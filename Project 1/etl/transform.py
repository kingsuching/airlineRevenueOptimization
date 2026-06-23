"""
transform.py
All data-wrangling logic for the Project 1 pipeline:

  1. Normalise FlightAware flight records
     - Extract IATA codes from nested dict strings (origin / destination)
     - Standardise timestamps to UTC
     - Map aircraft_type → aircraft_variant using AIRCRAFT_TYPE_MAP

  2. Flatten UAL aircraft seat-map JSON → clean DataFrame

  3. Compute flight_capacity table (flights × cabin_class → seats, ASM)

  4. Impute missing load factors from route-level averages (using BTS T-100)

  5. Flag load-factor outliers (> 3 σ from route mean)

  6. Build dim_time rows for any flight dates not yet in the table
"""

from __future__ import annotations

import ast
import logging
import re
from datetime import date

import numpy as np
import pandas as pd

from config import AIRCRAFT_TYPE_MAP, CABIN_TIER_MAP, FUEL_BURN_GPH

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Normalise flights DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def _parse_iata(cell) -> str | None:
    """Extract code_iata from a FlightAware airport dict (or stringified dict)."""
    if isinstance(cell, dict):
        return cell.get("code_iata")
    try:
        return ast.literal_eval(str(cell)).get("code_iata")
    except (ValueError, SyntaxError, AttributeError, TypeError):
        return None


_TS_COLS = [
    "scheduled_out", "estimated_out", "actual_out",
    "scheduled_off", "estimated_off", "actual_off",
    "scheduled_on",  "estimated_on",  "actual_on",
    "scheduled_in",  "estimated_in",  "actual_in",
]


def normalise_flights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw FlightAware flights DataFrame:
      - Rename fa_flight_id → flight_id
      - Parse origin/destination dicts → IATA codes
      - Coerce all timestamp columns to UTC-aware datetime
      - Map aircraft_type → aircraft_variant
      - Compute flight_date (UTC date of actual or scheduled departure)
      - Cast bool columns
    """
    out = df.copy()

    # ── Rename ID column ──────────────────────────────────────────────────
    if "fa_flight_id" in out.columns and "flight_id" not in out.columns:
        out = out.rename(columns={"fa_flight_id": "flight_id"})

    # ── Airport IATA codes ────────────────────────────────────────────────
    if "origin" in out.columns:
        out["origin_iata"] = out["origin"].map(_parse_iata)
    if "destination" in out.columns:
        out["destination_iata"] = out["destination"].map(_parse_iata)

    # ── Timestamps → UTC ──────────────────────────────────────────────────
    for col in _TS_COLS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")

    # ── Derived: flight_date (Zulu) ───────────────────────────────────────
    dep_col = next(
        (c for c in ["actual_off", "scheduled_off"] if c in out.columns), None
    )
    if dep_col:
        out["flight_date"] = out[dep_col].dt.date

    # ── Aircraft variant mapping ──────────────────────────────────────────
    if "aircraft_type" in out.columns:
        out["aircraft_variant"] = out["aircraft_type"].map(AIRCRAFT_TYPE_MAP)
        unmapped = set(
            out.loc[out["aircraft_variant"].isna(), "aircraft_type"]
            .dropna().unique()
        )
        if unmapped:
            logger.warning("Unmapped aircraft_type codes: %s", sorted(unmapped))

    # ── Boolean coercion ──────────────────────────────────────────────────
    for bool_col in ["cancelled", "diverted", "blocked", "position_only"]:
        if bool_col in out.columns:
            out[bool_col] = out[bool_col].map(
                lambda v: True if str(v).lower() in ("true", "1") else
                          False if str(v).lower() in ("false", "0") else None
            )

    # ── Numeric coercions ─────────────────────────────────────────────────
    for num_col in ["route_distance", "departure_delay", "arrival_delay",
                    "seats_cabin_first", "seats_cabin_business", "seats_cabin_coach"]:
        if num_col in out.columns:
            out[num_col] = pd.to_numeric(out[num_col], errors="coerce")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Flatten aircraft JSON → DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def flatten_aircraft_data(data: dict) -> pd.DataFrame:
    """
    Flatten the nested UAL_aircraft_data.json into a tidy DataFrame.
    Handles malformed nesting (where a variant key appears inside another variant).

    Adds:
      - cabin_tier  (standardised from CABIN_TIER_MAP)
      - fuel_gph    (from FUEL_BURN_GPH lookup)
    """
    rows: list[dict] = []

    def _process(aircraft_name: str, d: dict) -> None:
        for key, val in d.items():
            if not isinstance(val, dict):
                continue
            if any(isinstance(v, dict) for v in val.values()):
                _process(key, val)   # malformed nesting — recurse
            else:
                rows.append({"aircraft": aircraft_name, "cabin_class": key, **val})

    for aircraft, cabins in data.items():
        _process(aircraft, cabins)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Number of seats"] = pd.to_numeric(df["Number of seats"], errors="coerce")
    df["cabin_tier"]      = df["cabin_class"].map(CABIN_TIER_MAP).fillna("economy")
    df["fuel_gph"]        = df["aircraft"].map(FUEL_BURN_GPH)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Compute flight_capacity (flights × cabin → seats, ASM)
# ─────────────────────────────────────────────────────────────────────────────

def compute_flight_capacity(
    flights_df: pd.DataFrame,
    aircraft_df: pd.DataFrame,
) -> tuple[pd.DataFrame, set[str]]:
    """
    Explode flights by cabin class using the seat-map lookup.

    Parameters
    ----------
    flights_df  : normalised flights DataFrame (from normalise_flights)
    aircraft_df : flattened aircraft DataFrame (from flatten_aircraft_data)

    Returns
    -------
    capacity_df : (flight_id, cabin_class) level rows with seats + ASM
    unmapped    : set of aircraft_type codes we couldn't join
    """
    seats = (
        aircraft_df[["aircraft", "cabin_class", "cabin_tier", "Number of seats"]]
        .rename(columns={"Number of seats": "seats"})
        .copy()
    )
    seats["seats"] = pd.to_numeric(seats["seats"], errors="coerce")

    f = flights_df.copy()
    if "aircraft_variant" not in f.columns:
        f["aircraft_variant"] = f["aircraft_type"].map(AIRCRAFT_TYPE_MAP)

    have_seatmap = set(seats["aircraft"].unique())
    unmapped = set(
        f.loc[~f["aircraft_variant"].isin(have_seatmap), "aircraft_type"]
        .dropna().unique()
    )

    id_col = "flight_id" if "flight_id" in f.columns else "fa_flight_id"
    dist_col = next(
        (c for c in ["route_distance", "route_distance_mi"] if c in f.columns), None
    )

    merged = f.merge(
        seats, left_on="aircraft_variant", right_on="aircraft", how="inner"
    )

    if dist_col:
        merged["ASM"] = merged["seats"] * merged[dist_col]
        merged = merged.rename(columns={dist_col: "route_distance_mi"})
    else:
        merged["ASM"] = np.nan
        merged["route_distance_mi"] = np.nan

    keep_cols = [
        id_col, "ident", "aircraft_type", "aircraft_variant",
        "origin_iata", "destination_iata", "flight_date",
        "route_distance_mi", "cabin_class", "cabin_tier", "seats", "ASM",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    return merged[keep_cols].reset_index(drop=True), unmapped


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Load-factor imputation and outlier flagging
# ─────────────────────────────────────────────────────────────────────────────

def impute_load_factors(
    capacity_df: pd.DataFrame,
    t100_df: pd.DataFrame,
    carrier_col: str = "operator_iata",
) -> pd.DataFrame:
    """
    Attach load factors from BTS T-100 to the capacity table.

    Strategy:
      1. Compute route-level average LF from T-100 (carrier + O&D).
      2. Merge onto capacity_df (left join).
      3. Where LF is missing, fill with the overall carrier average.
    """
    if t100_df.empty or "load_factor" not in t100_df.columns:
        capacity_df["load_factor"]        = np.nan
        capacity_df["load_factor_source"] = "missing"
        return capacity_df

    route_lf = (
        t100_df.groupby(["carrier_code", "origin_iata", "destination_iata"])
        ["load_factor"].mean()
        .reset_index()
        .rename(columns={"load_factor": "route_avg_lf"})
    )

    carrier_lf = (
        t100_df.groupby("carrier_code")["load_factor"].mean()
        .rename("carrier_avg_lf")
        .reset_index()
    )

    out = capacity_df.copy()
    # Merge route LF
    out = out.merge(route_lf, on=["origin_iata", "destination_iata"], how="left")
    # Merge carrier LF
    if carrier_col in out.columns:
        out = out.merge(
            carrier_lf.rename(columns={"carrier_code": carrier_col}),
            on=carrier_col, how="left"
        )

    out["load_factor"] = out.get("route_avg_lf")
    out["load_factor_source"] = "t100_route"
    # Fill with carrier average where route is missing
    if "carrier_avg_lf" in out.columns:
        mask = out["load_factor"].isna()
        out.loc[mask, "load_factor"]        = out.loc[mask, "carrier_avg_lf"]
        out.loc[mask, "load_factor_source"] = "t100_carrier_avg"

    out["load_factor"] = out["load_factor"].clip(0, 1)
    return out.drop(columns=["route_avg_lf", "carrier_avg_lf"], errors="ignore")


def flag_lf_outliers(
    df: pd.DataFrame,
    lf_col: str = "load_factor",
    group_cols: list[str] | None = None,
    n_sigma: float = 3.0,
) -> pd.DataFrame:
    """
    Add a boolean column `lf_outlier` = True where load factor deviates
    more than n_sigma from the group mean.  Group defaults to (origin, destination).
    """
    group_cols = group_cols or ["origin_iata", "destination_iata"]
    group_cols = [c for c in group_cols if c in df.columns]

    out = df.copy()
    if not group_cols or lf_col not in out.columns:
        out["lf_outlier"] = False
        return out

    stats = (out.groupby(group_cols)[lf_col]
               .agg(["mean", "std"])
               .rename(columns={"mean": "_mu", "std": "_sigma"})
               .reset_index())
    out = out.merge(stats, on=group_cols, how="left")
    out["lf_outlier"] = (
        out[lf_col] - out["_mu"]).abs() > n_sigma * out["_sigma"].fillna(0)
    return out.drop(columns=["_mu", "_sigma"])


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Build dim_time rows for a set of dates
# ─────────────────────────────────────────────────────────────────────────────

_US_HOLIDAYS_2024_2026: set[date] = {
    # Federal holidays (approximate — extend as needed)
    date(2024, 1, 1),  date(2024, 5, 27), date(2024, 7, 4),  date(2024, 9, 2),
    date(2024, 11, 28),date(2024, 12, 25),
    date(2025, 1, 1),  date(2025, 5, 26), date(2025, 7, 4),  date(2025, 9, 1),
    date(2025, 11, 27),date(2025, 12, 25),
    date(2026, 1, 1),  date(2026, 5, 25), date(2026, 7, 4),  date(2026, 9, 7),
    date(2026, 11, 26),date(2026, 12, 25),
}


def build_dim_time(dates: pd.Series | list) -> pd.DataFrame:
    """
    Build dim_time rows for the given dates.
    """
    unique_dates = pd.to_datetime(pd.Series(dates).dropna().unique())
    rows = []
    for d in sorted(unique_dates):
        dt = d.date()
        rows.append({
            "time_id":      dt,
            "year":         d.year,
            "quarter":      d.quarter,
            "month":        d.month,
            "month_name":   d.strftime("%B"),
            "week_of_year": int(d.strftime("%V")),
            "day_of_week":  d.dayofweek,
            "day_name":     d.strftime("%A"),
            "is_weekend":   d.dayofweek >= 5,
            "is_holiday":   dt in _US_HOLIDAYS_2024_2026,
        })
    return pd.DataFrame(rows)
