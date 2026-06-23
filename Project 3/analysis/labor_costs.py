"""
labor_costs.py
Compute per-route flight crew labor costs for Project 3.

Inputs:
  - route_demand_summary (Project 2): departures, aircraft_variant
  - fuel_cost_detail.csv (Project 3 step 1): block_hours
  - labor_rates.json (Project 1): ALPA pilot and AFA FA contract rates

Outputs:
  - labor_cost_detail.csv
  - crew_cost_rates.csv (per-aircraft blended rates)

Methodology:
  Pilot cost/BH = (captain_avg + fo_avg) × benefits_multiplier
  FA cost/BH    = fa_avg_rate × n_fas × fa_benefits_multiplier
  Augmented crew rules applied for long-haul (>8h) and ultra-long-haul (>12h)
  casm_crew = crew_cost_usd / asm
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJECT1_ETL  = _HERE.parents[1] / "Project 1" / "etl"
_AIRCRAFT_DATA = _HERE.parents[1] / "Project 1" / "aircraft_data"
if str(_PROJECT1_ETL) not in sys.path:
    sys.path.insert(0, str(_PROJECT1_ETL))

try:
    from load import get_connection
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

_OUTPUT_DIR    = _HERE.parent / "outputs"
_OUTPUT_DIR.mkdir(exist_ok=True)
_P2_OUTPUT_DIR = _HERE.parents[1] / "Project 2" / "outputs"

logger = logging.getLogger(__name__)

FALLBACK_PILOT_COST_PER_BH = 1_400.0   # 2 pilots, benefits included
FALLBACK_FA_COST_PER_BH    =   280.0   # ~3.5 FAs average, benefits included
FALLBACK_CREW_COST_PER_BH  = FALLBACK_PILOT_COST_PER_BH + FALLBACK_FA_COST_PER_BH


# =============================================================================
# Build per-aircraft crew cost table from labor_rates.json
# =============================================================================

def _load_labor_json() -> dict:
    path = _AIRCRAFT_DATA / "labor_rates.json"
    if path.exists():
        with path.open() as f:
            return json.load(f)
    logger.warning("labor_rates.json not found; will use fallback rates.")
    return {}


def build_crew_cost_table(labor: dict) -> pd.DataFrame:
    """
    Derive a per-aircraft-variant crew cost table from labor_rates.json.

    Returns DataFrame with columns:
        aircraft_variant, equipment_group,
        captain_rate_per_fh, fo_rate_per_fh, n_fas, fa_rate_per_fh,
        pilot_cost_per_bh, fa_cost_per_bh, total_crew_cost_per_bh
    """
    if not labor:
        return pd.DataFrame()

    pilots = labor.get("pilots", {})
    fas    = labor.get("flight_attendants", {})

    # ── Pilot rates by equipment group ────────────────────────────────────────
    eq_groups   = pilots.get("equipment_groups", {})
    benefits_p  = pilots.get("benefits_multiplier", 1.38)
    capt_fo_ratio = pilots.get("captain_fo_ratio", 0.52)

    pilot_rows = []
    for group_name, gdata in eq_groups.items():
        if group_name.startswith("_"):
            continue
        capt_rate = gdata.get("captain_avg_blended", 350.0)
        fo_rate   = gdata.get("fo_avg_blended", capt_rate * capt_fo_ratio)
        # 2-pilot base cost (short-haul)
        pilot_cost_2p = (capt_rate + fo_rate) * benefits_p

        for variant in gdata.get("aircraft", []):
            pilot_rows.append({
                "aircraft_variant":     variant,
                "equipment_group":      group_name,
                "captain_rate_per_fh":  capt_rate,
                "fo_rate_per_fh":       fo_rate,
                "pilot_benefits_mult":  benefits_p,
                "pilot_cost_per_bh":    round(pilot_cost_2p, 2),
            })

    pilot_df = pd.DataFrame(pilot_rows)

    # ── FA rates and complement ───────────────────────────────────────────────
    fa_complement = fas.get("fa_complement_by_aircraft", {})
    fa_rate      = fas.get("avg_blended_rate", 55.0)
    benefits_fa  = fas.get("benefits_multiplier", 1.35)

    fa_rows = []
    for variant, n_fas in fa_complement.items():
        if variant.startswith("_"):
            continue
        fa_rows.append({
            "aircraft_variant": variant,
            "n_fas":            n_fas,
            "fa_rate_per_fh":   fa_rate,
            "fa_cost_per_bh":   round(fa_rate * n_fas * benefits_fa, 2),
        })

    fa_df = pd.DataFrame(fa_rows)

    if pilot_df.empty or fa_df.empty:
        return pd.DataFrame()

    crew = pilot_df.merge(fa_df, on="aircraft_variant", how="outer")
    crew["pilot_cost_per_bh"]      = crew["pilot_cost_per_bh"].fillna(FALLBACK_PILOT_COST_PER_BH)
    crew["fa_cost_per_bh"]         = crew["fa_cost_per_bh"].fillna(FALLBACK_FA_COST_PER_BH)
    crew["total_crew_cost_per_bh"] = (crew["pilot_cost_per_bh"] + crew["fa_cost_per_bh"]).round(2)

    logger.info("Crew cost table built: %d aircraft variants.", len(crew))
    return crew


# =============================================================================
# Load input data
# =============================================================================

def load_demand_data(source: str = "csv") -> pd.DataFrame:
    if source == "db" and _DB_AVAILABLE:
        conn = get_connection()
        try:
            return pd.read_sql(
                """SELECT report_period, carrier_code, origin_iata, destination_iata,
                          aircraft_variant, departures, distance_mi, asm
                   FROM route_demand_summary WHERE carrier_code = 'UA'""",
                conn, parse_dates=["report_period"],
            )
        finally:
            conn.close()

    for c in [_P2_OUTPUT_DIR / "route_demand_summary.csv",
              _OUTPUT_DIR / "route_demand_summary.csv"]:
        if c.exists():
            return pd.read_csv(c, parse_dates=["report_period"])
    return pd.DataFrame()


def load_fuel_cost_detail() -> pd.DataFrame:
    p = _OUTPUT_DIR / "fuel_cost_detail.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["report_period"])
    return pd.DataFrame()


# =============================================================================
# Augmented crew adjustment
# =============================================================================

def apply_augmented_crew_rules(df: pd.DataFrame, labor: dict) -> pd.DataFrame:
    """
    Multiply pilot cost by 1.5 for flights 8-12h block time (3 pilots)
    and by 2.0 for >12h (4 pilots), per United ALPA contract rules.
    Only applies to widebody-capable variants; narrowbodies cap at 2 pilots.
    """
    if "block_hours" not in df.columns or "departures" not in df.columns:
        return df

    rules     = labor.get("pilots", {}).get("crew_complement_by_aircraft", {})
    long_mult  = rules.get("long_haul_8_12h",     {}).get("pilots", 3) / 2.0
    ultra_mult = rules.get("ultra_long_haul_gt_12h", {}).get("pilots", 4) / 2.0

    # Hours per departure (to determine crew rule)
    df["bh_per_dep"] = np.where(
        df["departures"] > 0,
        df["block_hours"] / df["departures"],
        df["block_hours"],
    )

    # Only widebodies qualify for augmented crew (narrowbodies are always 2-pilot)
    widebody_variants = set(
        rules.get("long_haul_8_12h", {}).get("applicable_aircraft", [])
    ) | set(
        rules.get("ultra_long_haul_gt_12h", {}).get("applicable_aircraft", [])
    )

    is_widebody = df["aircraft_variant"].isin(widebody_variants)

    df["crew_multiplier"] = np.where(
        is_widebody & (df["bh_per_dep"] > 12),
        ultra_mult,
        np.where(
            is_widebody & (df["bh_per_dep"] > 8),
            long_mult,
            1.0,
        ),
    )

    if "pilot_cost_per_bh" in df.columns:
        df["pilot_cost_per_bh_adj"] = (
            df["pilot_cost_per_bh"] * df["crew_multiplier"]
        ).round(2)

    return df


# =============================================================================
# Core computation
# =============================================================================

def compute_labor_costs(
    demand_df:       pd.DataFrame,
    crew_cost_df:    pd.DataFrame,
    fuel_detail_df:  pd.DataFrame,
    labor:           dict,
) -> pd.DataFrame:
    df = demand_df.copy()
    df["report_period"] = pd.to_datetime(df["report_period"])

    # ── Block hours: prefer fuel_cost_detail where available ──────────────────
    key_cols = ["report_period", "carrier_code", "origin_iata",
                "destination_iata", "aircraft_variant"]

    if not fuel_detail_df.empty and "block_hours" in fuel_detail_df.columns:
        bh = fuel_detail_df[[*key_cols, "block_hours"]].copy()
        bh["report_period"] = pd.to_datetime(bh["report_period"])
        merge_on = [c for c in key_cols if c in df.columns and c in bh.columns]
        df = df.merge(bh, on=merge_on, how="left")

    if "block_hours" not in df.columns or df["block_hours"].isna().all():
        df["block_hours"] = (
            (df.get("distance_mi", pd.Series(0, index=df.index)) / 515.0 + 0.5)
            * df.get("departures", pd.Series(1, index=df.index))
        ).round(2)
    else:
        df["block_hours"] = df["block_hours"].fillna(
            (df.get("distance_mi", pd.Series(0, index=df.index)) / 515.0 + 0.5)
            * df.get("departures", pd.Series(1, index=df.index))
        )

    # ── Merge crew cost rates ──────────────────────────────────────────────────
    if not crew_cost_df.empty:
        rate_cols = ["aircraft_variant", "pilot_cost_per_bh", "fa_cost_per_bh",
                     "total_crew_cost_per_bh"]
        df = df.merge(
            crew_cost_df[[c for c in rate_cols if c in crew_cost_df.columns]],
            on="aircraft_variant", how="left",
        )
    else:
        df["pilot_cost_per_bh"]      = FALLBACK_PILOT_COST_PER_BH
        df["fa_cost_per_bh"]         = FALLBACK_FA_COST_PER_BH
        df["total_crew_cost_per_bh"] = FALLBACK_CREW_COST_PER_BH

    df["pilot_cost_per_bh"]      = df["pilot_cost_per_bh"].fillna(FALLBACK_PILOT_COST_PER_BH)
    df["fa_cost_per_bh"]         = df["fa_cost_per_bh"].fillna(FALLBACK_FA_COST_PER_BH)
    df["total_crew_cost_per_bh"] = df["total_crew_cost_per_bh"].fillna(FALLBACK_CREW_COST_PER_BH)

    # ── Augmented crew ─────────────────────────────────────────────────────────
    df = apply_augmented_crew_rules(df, labor)
    pilot_col = "pilot_cost_per_bh_adj" if "pilot_cost_per_bh_adj" in df.columns \
                else "pilot_cost_per_bh"

    # ── Cost computation ───────────────────────────────────────────────────────
    df["pilot_cost_usd"] = (df[pilot_col]         * df["block_hours"]).round(2)
    df["fa_cost_usd"]    = (df["fa_cost_per_bh"]  * df["block_hours"]).round(2)
    df["crew_cost_usd"]  = (df["pilot_cost_usd"]  + df["fa_cost_usd"]).round(2)

    asm = df.get("asm", pd.Series(0, index=df.index))
    df["casm_crew"] = np.where(asm > 0, (df["crew_cost_usd"] / asm).round(7), np.nan)

    logger.info("Labor cost computed for %d rows.", len(df))
    return df


# =============================================================================
# Summary
# =============================================================================

def labor_cost_summary(df: pd.DataFrame) -> dict:
    return {
        "total_pilot_cost_usd": round(float(df["pilot_cost_usd"].sum()), 2),
        "total_fa_cost_usd":    round(float(df["fa_cost_usd"].sum()),    2),
        "total_crew_cost_usd":  round(float(df["crew_cost_usd"].sum()),  2),
        "mean_casm_crew":       round(float(df["casm_crew"].mean()),     6),
    }


# =============================================================================
# Entrypoint
# =============================================================================

def run(source: str = "csv", save_csv: bool = True) -> pd.DataFrame:
    labor_json   = _load_labor_json()
    crew_cost_df = build_crew_cost_table(labor_json)

    demand_df      = load_demand_data(source)
    if demand_df.empty:
        logger.warning("No demand data; cannot compute labor costs.")
        return pd.DataFrame()

    fuel_detail_df = load_fuel_cost_detail()
    labor_df = compute_labor_costs(demand_df, crew_cost_df, fuel_detail_df, labor_json)

    keep = [c for c in [
        "report_period", "carrier_code", "origin_iata", "destination_iata",
        "aircraft_variant", "departures", "distance_mi", "asm", "block_hours",
        "pilot_cost_per_bh", "fa_cost_per_bh",
        "pilot_cost_usd", "fa_cost_usd", "crew_cost_usd", "casm_crew",
    ] if c in labor_df.columns]
    out = labor_df[keep].copy()

    if save_csv:
        out.to_csv(_OUTPUT_DIR / "labor_cost_detail.csv", index=False)
        if not crew_cost_df.empty:
            crew_cost_df.to_csv(_OUTPUT_DIR / "crew_cost_rates.csv", index=False)
        logger.info("Saved labor_cost_detail.csv, crew_cost_rates.csv → %s", _OUTPUT_DIR)

    logger.info("Labor cost summary: %s", labor_cost_summary(out))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    df = run(source="csv")
    if not df.empty:
        print(f"\nLabor cost detail: {len(df)} rows")
        print(df.head(10).to_string(index=False))
        print("\nSummary:")
        for k, v in labor_cost_summary(df).items():
            print(f"  {k}: {v}")
