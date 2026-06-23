"""
extract_labor_rates.py
Loads and applies the AFA-CWA / ALPA labor rate tables to compute
per-flight crew costs.

Sources used:
  1. aircraft_data/labor_rates.json  — compiled from AFA-CWA and ALPA contracts
  2. SEC EDGAR 10-K                  — annual total labor validation
  3. BTS Schedule P-6               — reported carrier-level labor validation

Main output function:
  compute_flight_crew_cost(flights_df) → DataFrame with pilot_cost_usd,
                                          fa_cost_usd, total_crew_cost_usd

Usage:
    python extract_labor_rates.py      # print summary tables
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import DATA_DIR

logger = logging.getLogger(__name__)

_LABOR_JSON_PATH = (
    Path(__file__).resolve().parents[1]
    / "aircraft_data" / "labor_rates.json"
)

# Seniority year to use when computing blended crew costs
# (fleet-wide average seniority for active United pilots/FAs)
DEFAULT_SENIORITY_YEAR = 12


# ─────────────────────────────────────────────────────────────────────────────
# Load contracts
# ─────────────────────────────────────────────────────────────────────────────

def load_labor_rates(path: Path | None = None) -> dict[str, Any]:
    path = path or _LABOR_JSON_PATH
    with path.open() as f:
        return json.load(f)


def get_aircraft_to_equipment_group(rates: dict) -> dict[str, str]:
    """Return {aircraft_variant: equipment_group_key} mapping."""
    mapping: dict[str, str] = {}
    for group_key, group_data in rates["pilots"]["equipment_groups"].items():
        if group_key.startswith("_"):
            continue
        for variant in group_data.get("aircraft", []):
            mapping[variant] = group_key
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# Per-flight cost calculators
# ─────────────────────────────────────────────────────────────────────────────

def pilot_cost_per_flight(
    aircraft_variant: str,
    block_time_hr: float,
    rates: dict,
    seniority_year: int = DEFAULT_SENIORITY_YEAR,
    use_blended: bool = True,
) -> dict[str, float]:
    """
    Compute pilot crew cost for a single flight.

    Parameters
    ----------
    aircraft_variant : United variant name (e.g. "B777-300ER")
    block_time_hr    : flight block time in hours
    rates            : loaded labor_rates dict
    seniority_year   : specific year on pay scale (ignored if use_blended=True)
    use_blended      : if True, use blended fleet average rates

    Returns
    -------
    dict with keys:
        captain_rate_hr, fo_rate_hr, num_pilots,
        pilot_flight_pay_usd, pilot_total_cost_usd (with benefits)
    """
    pilot_data = rates["pilots"]
    eq_group_map = get_aircraft_to_equipment_group(rates)
    eq_group_key = eq_group_map.get(aircraft_variant)

    if eq_group_key is None:
        # Default to Group E (narrowbody classic) if unknown
        logger.debug("Unknown variant '%s'; defaulting to Group_E rates.", aircraft_variant)
        eq_group_key = "Group_E_narrowbody_classic"

    eq_group = pilot_data["equipment_groups"][eq_group_key]

    if use_blended:
        captain_rate = eq_group["captain_avg_blended"]
        fo_rate      = eq_group["fo_avg_blended"]
    else:
        yr = str(min(seniority_year, 17)) if seniority_year < 17 else "17+"
        captain_rate = eq_group["captain_rates_by_year"].get(yr, eq_group["captain_avg_blended"])
        fo_rate      = captain_rate * pilot_data["captain_fo_ratio"]

    # Determine number of pilots based on block time
    crew = pilot_data["crew_complement_by_aircraft"]
    if block_time_hr > 12 and aircraft_variant in crew.get("ultra_long_haul_gt_12h", {}).get("applicable_aircraft", []):
        num_pilots = 4
    elif block_time_hr > 8 and aircraft_variant in crew.get("long_haul_8_12h", {}).get("applicable_aircraft", []):
        num_pilots = 3
    else:
        num_pilots = 2  # standard 2-pilot crew

    # Cost: captain × block_time + (num_pilots - 1) × FO rate × block_time
    # (for augmented: 1 captain + N FOs)
    flight_pay = (captain_rate + (num_pilots - 1) * fo_rate) * block_time_hr
    total_cost = flight_pay * pilot_data["benefits_multiplier"]

    return {
        "captain_rate_hr":      captain_rate,
        "fo_rate_hr":           fo_rate,
        "num_pilots":           num_pilots,
        "pilot_flight_pay_usd": round(flight_pay, 2),
        "pilot_total_cost_usd": round(total_cost, 2),
    }


def fa_cost_per_flight(
    aircraft_variant: str,
    block_time_hr: float,
    rates: dict,
    is_international: bool = False,
    use_blended: bool = True,
    seniority_year: int = DEFAULT_SENIORITY_YEAR,
) -> dict[str, float]:
    """
    Compute flight attendant crew cost for a single flight.

    Returns dict with keys:
        num_fas, fa_rate_hr, fa_flight_pay_usd, fa_total_cost_usd
    """
    fa_data = rates["flight_attendants"]
    num_fas = fa_data["fa_complement_by_aircraft"].get(aircraft_variant, 3)

    if use_blended:
        base_rate = fa_data["avg_blended_rate"]
    else:
        yr = str(min(seniority_year, 19)) if seniority_year < 19 else "19+"
        base_rate = fa_data["base_rates_by_year"].get(yr, fa_data["avg_blended_rate"])

    intl_premium = fa_data["international_premium_pct"] if is_international else 0.0
    effective_rate = base_rate * (1 + intl_premium)

    flight_pay = effective_rate * block_time_hr * num_fas
    total_cost = flight_pay * fa_data["benefits_multiplier"]

    return {
        "num_fas":              num_fas,
        "fa_rate_hr":           round(effective_rate, 2),
        "fa_flight_pay_usd":    round(flight_pay, 2),
        "fa_total_cost_usd":    round(total_cost, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised: apply to a flights DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def compute_flight_crew_cost(
    flights_df: pd.DataFrame,
    block_time_col: str = "block_time_min",
    variant_col:    str = "aircraft_variant",
    is_intl_col:    str | None = None,
) -> pd.DataFrame:
    """
    Add crew cost columns to a flights DataFrame.

    Parameters
    ----------
    flights_df     : DataFrame with at least aircraft_variant and block_time_min
    block_time_col : column name for block time in minutes
    variant_col    : column name for United aircraft variant
    is_intl_col    : optional boolean column for international flag

    Returns
    -------
    flights_df with added columns:
        block_time_hr, num_pilots, num_fas,
        pilot_flight_pay_usd, pilot_total_cost_usd,
        fa_flight_pay_usd, fa_total_cost_usd,
        total_crew_cost_usd
    """
    rates = load_labor_rates()
    out   = flights_df.copy()

    # Block time in hours
    if block_time_col in out.columns:
        out["block_time_hr"] = pd.to_numeric(out[block_time_col], errors="coerce") / 60
    else:
        out["block_time_hr"] = np.nan

    # International flag
    if is_intl_col and is_intl_col in out.columns:
        is_intl_series = out[is_intl_col].fillna(False).astype(bool)
    else:
        # Heuristic: flag as international if block_time > 5h (crude proxy)
        is_intl_series = out.get("block_time_hr", pd.Series(dtype=float)) > 5.0

    # Apply row-by-row (vectorised via apply; can be sped up with lookup tables)
    pilot_cols = ["num_pilots", "pilot_flight_pay_usd", "pilot_total_cost_usd"]
    fa_cols    = ["num_fas", "fa_flight_pay_usd", "fa_total_cost_usd"]

    def _pilot_row(row):
        variant = row.get(variant_col, "unknown")
        bth     = row.get("block_time_hr", 0) or 0
        if pd.isna(bth) or bth <= 0:
            return pd.Series({"num_pilots": 2, "pilot_flight_pay_usd": np.nan,
                               "pilot_total_cost_usd": np.nan})
        res = pilot_cost_per_flight(variant, bth, rates)
        return pd.Series({k: res[k] for k in ["num_pilots", "pilot_flight_pay_usd",
                                               "pilot_total_cost_usd"]})

    def _fa_row(row):
        variant = row.get(variant_col, "unknown")
        bth     = row.get("block_time_hr", 0) or 0
        intl    = bool(is_intl_series.get(row.name, False))
        if pd.isna(bth) or bth <= 0:
            return pd.Series({"num_fas": 3, "fa_flight_pay_usd": np.nan,
                               "fa_total_cost_usd": np.nan})
        res = fa_cost_per_flight(variant, bth, rates, is_international=intl)
        return pd.Series({k: res[k] for k in ["num_fas", "fa_flight_pay_usd",
                                               "fa_total_cost_usd"]})

    out[pilot_cols] = out.apply(_pilot_row, axis=1)
    out[fa_cols]    = out.apply(_fa_row, axis=1)

    out["total_crew_cost_usd"] = (
        out["pilot_total_cost_usd"].fillna(0)
        + out["fa_total_cost_usd"].fillna(0)
    ).replace(0, np.nan).round(2)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables for reporting
# ─────────────────────────────────────────────────────────────────────────────

def build_crew_cost_library(rates: dict | None = None) -> pd.DataFrame:
    """
    Build a reference table of crew costs per block hour by aircraft variant.
    Useful for Project 3 cost benchmarking.
    """
    rates = rates or load_labor_rates()
    eq_map = get_aircraft_to_equipment_group(rates)

    rows = []
    for variant, group_key in sorted(eq_map.items()):
        g = rates["pilots"]["equipment_groups"][group_key]
        fa_count = rates["flight_attendants"]["fa_complement_by_aircraft"].get(variant, 3)
        fa_blended = rates["flight_attendants"]["avg_blended_rate"]
        fa_benefits = rates["flight_attendants"]["benefits_multiplier"]
        p_benefits  = rates["pilots"]["benefits_multiplier"]

        pilot_cost_hr = (
            g["captain_avg_blended"]
            + 1 * g["fo_avg_blended"]   # standard 2-crew
        ) * p_benefits
        fa_cost_hr = fa_blended * fa_count * fa_benefits

        rows.append({
            "aircraft_variant":         variant,
            "equipment_group":          group_key.replace("Group_", ""),
            "captain_rate_hr":          g["captain_avg_blended"],
            "fo_rate_hr":               g["fo_avg_blended"],
            "fa_count":                 fa_count,
            "fa_rate_hr":               round(fa_blended, 2),
            "pilot_cost_per_bh":        round(pilot_cost_hr, 2),
            "fa_cost_per_bh":           round(fa_cost_hr, 2),
            "total_crew_cost_per_bh":   round(pilot_cost_hr + fa_cost_hr, 2),
        })

    return pd.DataFrame(rows)


def load_bts_p6_validation(rates: dict | None = None) -> pd.DataFrame:
    """Load BTS P-6 reported labor totals from the labor_rates.json."""
    rates = rates or load_labor_rates()
    return pd.DataFrame(rates.get("bts_p6_validation", {}).get("data", []))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    rates = load_labor_rates()
    lib = build_crew_cost_library(rates)
    print("\n── Crew Cost Library ($/block hour, loaded including benefits) ──")
    print(lib.to_string(index=False))

    print("\n── BTS P-6 Labor Validation ──")
    p6 = load_bts_p6_validation(rates)
    print(p6.to_string(index=False))

    # Save crew cost library
    out = DATA_DIR / "crew_cost_library.csv"
    lib.to_csv(out, index=False)
    print(f"\nSaved crew cost library → {out}")
