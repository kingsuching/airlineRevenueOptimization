"""
fuel_costs.py
Compute per-route fuel costs and fuel CASM for Project 3.

Inputs:
  - route_demand_summary (Project 2): departures, distance_mi, aircraft_variant, asm
  - fuel_burn_rates.json (Project 1): BTS block-hour GPH by aircraft variant
  - fuel_prices (Project 1): weekly EIA Jet-A prices

Outputs:
  - fuel_cost_detail.csv
  - CSV columns: report_period, carrier_code, origin_iata, destination_iata,
                 aircraft_variant, departures, distance_mi, asm, block_hours,
                 gph_block_hour, fuel_gallons_est, jet_a_price_usd,
                 fuel_cost_usd, casm_fuel

Methodology:
  block_hours  = (distance_mi / CRUISE_SPEED_MPH + TAXI_HOURS) × departures
  fuel_gallons = gph_block_hour × block_hours
  fuel_cost    = fuel_gallons × jet_a_price
  casm_fuel    = fuel_cost / asm
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJECT1_ETL   = _HERE.parents[1] / "Project 1" / "etl"
_AIRCRAFT_DATA  = _HERE.parents[1] / "Project 1" / "aircraft_data"
if str(_PROJECT1_ETL) not in sys.path:
    sys.path.insert(0, str(_PROJECT1_ETL))

try:
    from load import get_connection
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

try:
    from extract_eia_fuel import run as _eia_run
    _EIA_AVAILABLE = True
except ImportError:
    _EIA_AVAILABLE = False

_OUTPUT_DIR    = _HERE.parent / "outputs"
_OUTPUT_DIR.mkdir(exist_ok=True)
_P2_OUTPUT_DIR = _HERE.parents[1] / "Project 2" / "outputs"
_P1_DATA_DIR   = _HERE.parents[1] / "Project 1" / "data"

logger = logging.getLogger(__name__)

# ── Physics / conversion constants ────────────────────────────────────────────
CRUISE_SPEED_MPH      = 515.0    # typical cruise groundspeed used for block-time estimate
TAXI_HOURS            = 0.50     # average taxi-out + taxi-in allowance (hours per departure)
FALLBACK_GPH          = 850.0    # used when aircraft variant is not in burn-rate table
FALLBACK_JET_A        = 2.90     # USD/gal fallback when no EIA price is available

# POH cruise GPH → block-hour GPH adjustment:
# Block hours include taxi, climb, and descent which burn more than cruise.
# Derived from fleet-average BTS/POH ratio across UA fleet (~+10%).
POH_TO_BLOCK_HOUR_FACTOR = 1.10


# =============================================================================
# Load reference data
# =============================================================================

def load_fuel_burn_rates() -> dict[str, float]:
    """
    Load fuel burn rates from fuel_burn_rates.json.

    Primary source: POH cruise GPH × POH_TO_BLOCK_HOUR_FACTOR (accounts for
    taxi, climb, and descent phases not captured in cruise-only figures).

    Fallback chain:
      1. poh_cruise_gph × 1.10  (FAA TCDS + FCOM-sourced cruise rate)
      2. bts_block_hour_gph      (BTS Form 41 actuals — block-hour rate)
      3. FALLBACK_GPH constant   (fleet-wide generic)

    Returns {aircraft_variant: gph_block_hour}.
    """
    path = _AIRCRAFT_DATA / "fuel_burn_rates.json"
    if not path.exists():
        logger.warning("fuel_burn_rates.json not found; falling back to config.FUEL_BURN_GPH.")
        try:
            from config import FUEL_BURN_GPH
            return dict(FUEL_BURN_GPH)
        except ImportError:
            return {}

    with path.open() as f:
        raw = json.load(f)

    lookup: dict[str, float] = {}
    poh_count = 0
    bts_count = 0

    for section, aircraft in raw.items():
        if section.startswith("_"):
            continue
        if not isinstance(aircraft, dict):
            continue
        for _, entry in aircraft.items():
            if not isinstance(entry, dict):
                continue
            variant = entry.get("aircraft_variant")
            if not variant:
                continue

            poh_cruise = entry.get("poh_cruise_gph")
            bts_block  = entry.get("bts_block_hour_gph")

            if poh_cruise:
                # Convert cruise GPH to block-hour equivalent
                gph = round(float(poh_cruise) * POH_TO_BLOCK_HOUR_FACTOR, 1)
                lookup[variant] = gph
                poh_count += 1
            elif bts_block:
                lookup[variant] = float(bts_block)
                bts_count += 1

    logger.info(
        "Loaded GPH rates for %d aircraft variants "
        "(POH-primary: %d, BTS-fallback: %d).",
        len(lookup), poh_count, bts_count,
    )
    return lookup


def load_demand_data(source: str = "csv") -> pd.DataFrame:
    if source == "db" and _DB_AVAILABLE:
        conn = get_connection()
        try:
            sql = """
                SELECT report_period, carrier_code, origin_iata, destination_iata,
                       aircraft_variant, departures, seats_available,
                       distance_mi, asm, rpm
                FROM route_demand_summary
                WHERE carrier_code = 'UA'
                ORDER BY report_period
            """
            return pd.read_sql(sql, conn, parse_dates=["report_period"])
        finally:
            conn.close()

    for candidate in [
        _P2_OUTPUT_DIR / "route_demand_summary.csv",
        _OUTPUT_DIR / "route_demand_summary.csv",
    ]:
        if candidate.exists():
            return pd.read_csv(candidate, parse_dates=["report_period"])

    logger.warning("No route_demand_summary.csv found.")
    return pd.DataFrame()


def load_fuel_prices(source: str = "csv") -> pd.DataFrame:
    """
    Load Jet-A weekly prices.

    Source priority:
      1. EIA API v2 (live, if EIA_API_KEY is set and cache is stale)
      2. Cached CSV at Project 1/data/fuel_prices.csv
      3. Fallback constant FALLBACK_JET_A
    """
    if source == "db" and _DB_AVAILABLE:
        conn = get_connection()
        try:
            return pd.read_sql(
                "SELECT price_date, jet_a_usd_per_gal FROM fuel_prices ORDER BY price_date",
                conn, parse_dates=["price_date"],
            )
        finally:
            conn.close()

    # Try live EIA fetch (auto-updates cache; returns cache if fresh or key absent)
    if _EIA_AVAILABLE:
        try:
            eia_df = _eia_run(offline=False, save_csv=True)
            if not eia_df.empty:
                eia_df["price_date"] = pd.to_datetime(eia_df["price_date"])
                logger.info("Using EIA fuel prices: %d weekly obs.", len(eia_df))
                return eia_df
        except Exception as exc:
            logger.warning("EIA fuel fetch failed (%s); falling back to CSV.", exc)

    # CSV fallback
    for candidate in [
        _P1_DATA_DIR / "fuel_prices.csv",
        _OUTPUT_DIR / "fuel_prices.csv",
    ]:
        if candidate.exists():
            return pd.read_csv(candidate, parse_dates=["price_date"])

    logger.warning("No fuel_prices data found; will use fallback $%.2f/gal.", FALLBACK_JET_A)
    return pd.DataFrame()


# =============================================================================
# Core computation
# =============================================================================

def compute_fuel_costs(
    demand_df:      pd.DataFrame,
    gph_lookup:     dict[str, float],
    fuel_price_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute fuel cost per route-period-aircraft.

    Steps:
      1. Estimate block hours from stage distance + taxi allowance.
      2. Look up BTS block-hour GPH by aircraft variant.
      3. Multiply gallons by backward-matched EIA Jet-A price.
      4. Divide by ASM to get casm_fuel.
    """
    df = demand_df.copy()
    df["report_period"] = pd.to_datetime(df["report_period"])

    # ── Block hours ────────────────────────────────────────────────────────────
    if "distance_mi" in df.columns:
        df["est_block_hours_per_dep"] = (
            df["distance_mi"] / CRUISE_SPEED_MPH + TAXI_HOURS
        ).clip(lower=0.3).round(3)
    else:
        df["est_block_hours_per_dep"] = 1.5

    df["block_hours"] = (
        df["est_block_hours_per_dep"] * df.get("departures", pd.Series(1, index=df.index))
    ).round(2)

    # ── GPH lookup ────────────────────────────────────────────────────────────
    df["gph_block_hour"] = df["aircraft_variant"].map(gph_lookup).fillna(FALLBACK_GPH)

    # ── Fuel gallons ──────────────────────────────────────────────────────────
    df["fuel_gallons_est"] = (df["gph_block_hour"] * df["block_hours"]).round(0)

    # ── Jet-A price (backward ASOF join to nearest weekly EIA price) ──────────
    if not fuel_price_df.empty:
        fp = (
            fuel_price_df
            .rename(columns={"price_date": "report_period",
                             "jet_a_usd_per_gal": "jet_a_price_usd"})
            .sort_values("report_period")
        )
        df = pd.merge_asof(
            df.sort_values("report_period"),
            fp[["report_period", "jet_a_price_usd"]],
            on="report_period",
            direction="backward",
        )
        df["jet_a_price_usd"] = df["jet_a_price_usd"].fillna(FALLBACK_JET_A)
    else:
        df["jet_a_price_usd"] = FALLBACK_JET_A

    # ── Fuel cost ─────────────────────────────────────────────────────────────
    df["fuel_cost_usd"] = (df["fuel_gallons_est"] * df["jet_a_price_usd"]).round(2)

    # ── CASM_fuel ─────────────────────────────────────────────────────────────
    asm = df.get("asm", pd.Series(0, index=df.index))
    df["casm_fuel"] = np.where(
        asm > 0,
        (df["fuel_cost_usd"] / asm).round(7),
        np.nan,
    )

    logger.info("Fuel cost computed for %d route-period rows.", len(df))
    return df


# =============================================================================
# Summary
# =============================================================================

def fuel_cost_summary(fuel_df: pd.DataFrame) -> dict:
    return {
        "total_fuel_gallons_est": round(float(fuel_df["fuel_gallons_est"].sum()), 0),
        "total_fuel_cost_usd":    round(float(fuel_df["fuel_cost_usd"].sum()), 2),
        "mean_casm_fuel":         round(float(fuel_df["casm_fuel"].mean()), 6),
        "mean_jet_a_price_usd":   round(float(fuel_df["jet_a_price_usd"].mean()), 4),
    }


# =============================================================================
# Entrypoint
# =============================================================================

def run(source: str = "csv", save_csv: bool = True) -> pd.DataFrame:
    demand_df = load_demand_data(source)
    if demand_df.empty:
        logger.warning("No demand data; cannot compute fuel costs.")
        return pd.DataFrame()

    gph_lookup    = load_fuel_burn_rates()
    fuel_price_df = load_fuel_prices(source)

    fuel_df = compute_fuel_costs(demand_df, gph_lookup, fuel_price_df)

    keep = [c for c in [
        "report_period", "carrier_code", "origin_iata", "destination_iata",
        "aircraft_variant", "departures", "distance_mi", "asm",
        "est_block_hours_per_dep", "block_hours",
        "gph_block_hour", "fuel_gallons_est", "jet_a_price_usd",
        "fuel_cost_usd", "casm_fuel",
    ] if c in fuel_df.columns]
    out = fuel_df[keep].copy()

    if save_csv:
        out.to_csv(_OUTPUT_DIR / "fuel_cost_detail.csv", index=False)
        logger.info("Saved fuel_cost_detail.csv → %s", _OUTPUT_DIR)

    logger.info("Fuel cost summary: %s", fuel_cost_summary(out))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    df = run(source="csv")
    if not df.empty:
        print(f"\nFuel cost detail: {len(df)} rows")
        print(df.head(10).to_string(index=False))
        print("\nSummary:")
        for k, v in fuel_cost_summary(df).items():
            print(f"  {k}: {v}")
