"""
maintenance.py
Estimate aircraft Direct Maintenance Cost (DMC) per block hour for Project 3.

Source benchmarks:
  Oliver Wyman 2024 MRO Survey, ICF Aviation Cost Estimating Guide,
  CAPA Centre for Aviation fleet cost analyses.

Cost components (per block hour):
  - Airframe: scheduled checks (A/B/C/D), structural, cabin
  - Engine/APU: on-wing restoration, LLP replacement, shop visits
  - Components: LRUs, avionics, hydraulics, landing gear

Outputs:
  - maintenance_cost_detail.csv
  - aircraft_maintenance_rates.csv (benchmark table)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJECT1_ETL  = _HERE.parents[1] / "Project 1" / "etl"
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

FALLBACK_DMC_PER_BH = 400.0  # generic fallback USD/block-hour

# ── DMC benchmarks (USD / block hour at mature airline scale) ─────────────────
# Widebody rates are higher due to larger structure, more complex engines,
# and longer life-limited-part (LLP) replacement intervals.
MAINTENANCE_RATES: dict[str, dict] = {
    # ── Heavy widebodies (GE90 / GE9X / Trent engines) ───────────────────────
    "B777-300ER":               {"airframe": 650, "engine": 850, "components": 320},
    "B777-200":                 {"airframe": 620, "engine": 800, "components": 290},
    "B777-200ER Version 1":     {"airframe": 620, "engine": 800, "components": 290},
    "B777-200ER Version 2":     {"airframe": 620, "engine": 800, "components": 290},
    # ── Dreamliners (composite airframe, GEnx / Trent 1000) ─────────────────
    "B787-10":                  {"airframe": 380, "engine": 520, "components": 240},
    "B787-9 Version 1":         {"airframe": 360, "engine": 490, "components": 220},
    "B787-9 Version 2":         {"airframe": 360, "engine": 490, "components": 220},
    "B787-8":                   {"airframe": 340, "engine": 460, "components": 210},
    # ── Legacy widebodies (PW4000 / CF6) ────────────────────────────────────
    "767-400ER":                {"airframe": 480, "engine": 620, "components": 270},
    "767-300ER Version 1":      {"airframe": 460, "engine": 590, "components": 260},
    "767-300ER Version 2":      {"airframe": 460, "engine": 590, "components": 260},
    "757-300":                  {"airframe": 310, "engine": 380, "components": 170},
    "757-200":                  {"airframe": 290, "engine": 350, "components": 160},
    # ── Classic narrowbodies (CFM56-7B) ─────────────────────────────────────
    "737-900 Version 1":        {"airframe": 210, "engine": 240, "components": 140},
    "737-900 Version 2":        {"airframe": 210, "engine": 240, "components": 140},
    "737-900 Version 3":        {"airframe": 210, "engine": 240, "components": 140},
    "737-800 Version 1":        {"airframe": 200, "engine": 230, "components": 135},
    "737-800 Version 2":        {"airframe": 200, "engine": 230, "components": 135},
    "737-800 Version 3":        {"airframe": 200, "engine": 230, "components": 135},
    "737-700":                  {"airframe": 190, "engine": 220, "components": 130},
    "A320":                     {"airframe": 185, "engine": 215, "components": 132},
    "A319":                     {"airframe": 178, "engine": 202, "components": 128},
    # ── Next-gen / neo narrowbodies (LEAP-1B / CFM LEAP) ────────────────────
    "737 MAX 9 Version 1":      {"airframe": 165, "engine": 185, "components": 120},
    "737 MAX 9 Version 2":      {"airframe": 165, "engine": 185, "components": 120},
    "737 MAX 8 Version 1":      {"airframe": 155, "engine": 175, "components": 115},
    "737 MAX 8 Version 2":      {"airframe": 155, "engine": 175, "components": 115},
    "A321neo":                  {"airframe": 160, "engine": 188, "components": 118},
    # ── Regional jets (CF34 / PW1700G) ──────────────────────────────────────
    "CRJ550":                   {"airframe": 158, "engine": 192, "components":  98},
    "CRJ700":                   {"airframe": 152, "engine": 182, "components":  93},
    "CRJ200":                   {"airframe": 122, "engine": 148, "components":  78},
    "Embraer E175 Version 1":   {"airframe": 138, "engine": 158, "components":  85},
    "Embraer E175 Version 2":   {"airframe": 138, "engine": 158, "components":  85},
    "Embraer E170":             {"airframe": 132, "engine": 150, "components":  80},
}


# =============================================================================
# Reference table
# =============================================================================

def build_maintenance_rates_table() -> pd.DataFrame:
    rows = []
    for variant, rates in MAINTENANCE_RATES.items():
        af   = rates.get("airframe",   0)
        eng  = rates.get("engine",     0)
        comp = rates.get("components", 0)
        rows.append({
            "aircraft_variant":       variant,
            "airframe_per_bh_usd":    af,
            "engine_per_bh_usd":      eng,
            "components_per_bh_usd":  comp,
            "dmc_per_bh_usd":         af + eng + comp,
            "cost_basis":             "industry_benchmark",
        })
    return pd.DataFrame(rows)


# =============================================================================
# Load inputs
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


def load_block_hours() -> pd.DataFrame:
    p = _OUTPUT_DIR / "fuel_cost_detail.csv"
    if p.exists():
        return pd.read_csv(p, parse_dates=["report_period"])
    return pd.DataFrame()


# =============================================================================
# Core computation
# =============================================================================

def compute_maintenance_costs(
    demand_df:      pd.DataFrame,
    maint_rates_df: pd.DataFrame,
    block_hours_df: pd.DataFrame,
) -> pd.DataFrame:
    df = demand_df.copy()
    df["report_period"] = pd.to_datetime(df["report_period"])

    # ── Block hours from fuel_cost_detail or distance fallback ─────────────────
    key_cols = ["report_period", "carrier_code", "origin_iata",
                "destination_iata", "aircraft_variant"]

    if not block_hours_df.empty and "block_hours" in block_hours_df.columns:
        bh = block_hours_df[[*key_cols, "block_hours"]].copy()
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

    # ── Maintenance rates ──────────────────────────────────────────────────────
    rate_cols = ["aircraft_variant", "airframe_per_bh_usd",
                 "engine_per_bh_usd", "components_per_bh_usd", "dmc_per_bh_usd"]
    df = df.merge(
        maint_rates_df[[c for c in rate_cols if c in maint_rates_df.columns]],
        on="aircraft_variant", how="left",
    )
    df["dmc_per_bh_usd"] = df["dmc_per_bh_usd"].fillna(FALLBACK_DMC_PER_BH)

    # ── Cost ──────────────────────────────────────────────────────────────────
    df["maintenance_cost_usd"] = (df["dmc_per_bh_usd"] * df["block_hours"]).round(2)

    asm = df.get("asm", pd.Series(0, index=df.index))
    df["casm_maintenance"] = np.where(
        asm > 0, (df["maintenance_cost_usd"] / asm).round(7), np.nan
    )

    logger.info("Maintenance cost computed for %d rows.", len(df))
    return df


# =============================================================================
# Summary
# =============================================================================

def maintenance_cost_summary(df: pd.DataFrame) -> dict:
    return {
        "total_maintenance_cost_usd": round(float(df["maintenance_cost_usd"].sum()), 2),
        "mean_casm_maintenance":      round(float(df["casm_maintenance"].mean()), 6),
        "mean_dmc_per_bh_usd":        round(float(df["dmc_per_bh_usd"].mean()), 2),
    }


# =============================================================================
# Entrypoint
# =============================================================================

def run(source: str = "csv", save_csv: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    maint_rates_df = build_maintenance_rates_table()
    demand_df      = load_demand_data(source)

    if demand_df.empty:
        logger.warning("No demand data; cannot compute maintenance costs.")
        return pd.DataFrame(), maint_rates_df

    block_hours_df = load_block_hours()
    maint_df = compute_maintenance_costs(demand_df, maint_rates_df, block_hours_df)

    keep = [c for c in [
        "report_period", "carrier_code", "origin_iata", "destination_iata",
        "aircraft_variant", "departures", "distance_mi", "asm", "block_hours",
        "airframe_per_bh_usd", "engine_per_bh_usd", "components_per_bh_usd",
        "dmc_per_bh_usd", "maintenance_cost_usd", "casm_maintenance",
    ] if c in maint_df.columns]
    out = maint_df[keep].copy()

    if save_csv:
        out.to_csv(_OUTPUT_DIR / "maintenance_cost_detail.csv", index=False)
        maint_rates_df.to_csv(_OUTPUT_DIR / "aircraft_maintenance_rates.csv", index=False)
        logger.info("Saved maintenance_cost_detail.csv, aircraft_maintenance_rates.csv → %s",
                    _OUTPUT_DIR)

    logger.info("Maintenance cost summary: %s", maintenance_cost_summary(out))
    return out, maint_rates_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    df, rates = run(source="csv")
    if not df.empty:
        print(f"\nMaintenance cost detail: {len(df)} rows")
        print(df.head(10).to_string(index=False))
    print("\nMaintenance rates (sorted by DMC/BH):")
    print(rates.sort_values("dmc_per_bh_usd", ascending=False).to_string(index=False))
