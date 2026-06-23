"""
casm.py
Assemble full CASM cost structure from component modules and benchmark vs peers.

Cost components assembled here:
  1. Fuel             → fuel_cost_detail.csv  (fuel_costs.py)
  2. Crew labor       → labor_cost_detail.csv (labor_costs.py)
  3. Maintenance DMC  → maintenance_cost_detail.csv (maintenance.py)
  4. Airport / Navaid fees (estimated from stage-length bucket)
  5. Ownership        (depreciation + financing, per-block-hour fleet allocation)
  6. Overhead / G&A   (per-ASM fleet allocation)

Outputs:
  - route_cost_summary.csv  (full cost breakdown per route / period)
  - carrier_casm_benchmarks.csv  (BTS Form 41 actuals — UA vs AA, DL, WN, AS, B6)
  - casm_cost_feed.csv  (trimmed feed for Project 4 profitability model)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJECT1_ETL  = _HERE.parents[1] / "Project 1" / "etl"
_P1_DATA_DIR   = _HERE.parents[1] / "Project 1" / "data"
if str(_PROJECT1_ETL) not in sys.path:
    sys.path.insert(0, str(_PROJECT1_ETL))

try:
    from load import get_connection, _upsert
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

try:
    from extract_bts_form41 import load_form41, aggregate_form41
    _BTS_EXTRACTOR = True
except ImportError:
    _BTS_EXTRACTOR = False

try:
    from extract_sec_edgar import fetch_ownership_from_xbrl, derive_ownership_per_block_hour
    _EDGAR_AVAILABLE = True
except ImportError:
    _EDGAR_AVAILABLE = False

_OUTPUT_DIR    = _HERE.parent / "outputs"
_OUTPUT_DIR.mkdir(exist_ok=True)
_P2_OUTPUT_DIR = _HERE.parents[1] / "Project 2" / "outputs"

logger = logging.getLogger(__name__)

# ── Airport / Navigation fee model (USD per departure, by stage-length bucket) ─
AIRPORT_FEE_BY_STAGE: dict[str, float] = {
    "ultra_short":   750.0,   # < 250 mi  (commuter / regional)
    "short":        1_050.0,  # 250-500 mi
    "medium":       1_350.0,  # 500-1000 mi
    "long":         1_800.0,  # 1000-2000 mi
    "ultra_long":   2_500.0,  # > 2000 mi (transcon / international)
}

# ── Ownership costs (USD / block hour) — depreciation + financing ──────────────
# Derived from United's reported D&A ÷ block hours (Form 41 Schedule P-5.2 proxy)
OWNERSHIP_PER_BH: dict[str, float] = {
    "B777-300ER":             3_800.0,
    "B777-200":               3_400.0,
    "B777-200ER Version 1":   3_400.0,
    "B777-200ER Version 2":   3_400.0,
    "B787-10":                3_200.0,
    "B787-9 Version 1":       3_000.0,
    "B787-9 Version 2":       3_000.0,
    "B787-8":                 2_800.0,
    "767-400ER":              1_900.0,
    "767-300ER Version 1":    1_800.0,
    "767-300ER Version 2":    1_800.0,
    "757-300":                1_200.0,
    "757-200":                1_100.0,
    "737-900 Version 1":        950.0,
    "737-900 Version 2":        950.0,
    "737-900 Version 3":        950.0,
    "737-800 Version 1":        900.0,
    "737-800 Version 2":        900.0,
    "737-800 Version 3":        900.0,
    "737-700":                  850.0,
    "737 MAX 9 Version 1":    1_100.0,
    "737 MAX 9 Version 2":    1_100.0,
    "737 MAX 8 Version 1":    1_050.0,
    "737 MAX 8 Version 2":    1_050.0,
    "A321neo":                1_100.0,
    "A320":                     880.0,
    "A319":                     820.0,
    "CRJ550":                   480.0,
    "CRJ700":                   450.0,
    "CRJ200":                   380.0,
    "Embraer E175 Version 1":   420.0,
    "Embraer E175 Version 2":   420.0,
    "Embraer E170":             400.0,
}
FALLBACK_OWNERSHIP_PER_BH = 1_200.0

# ── Overhead / G&A (USD per ASM) — selling, G&A, distribution ────────────────
OVERHEAD_PER_ASM = 0.018   # ~1.8 ¢/ASM (typical network carrier benchmark)

CARRIER_NAMES = {
    "UA": "United Airlines",
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "WN": "Southwest Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
}

KEY_COLS = ["report_period", "carrier_code", "origin_iata", "destination_iata", "aircraft_variant"]


# =============================================================================
# Load component CSV outputs
# =============================================================================

def _load_csv(filename: str, date_cols: list[str] | None = None) -> pd.DataFrame:
    p = _OUTPUT_DIR / filename
    if p.exists():
        kw = {"parse_dates": date_cols} if date_cols else {}
        df = pd.read_csv(p, **kw)
        logger.info("Loaded %s: %d rows", filename, len(df))
        return df
    logger.warning("%s not found in outputs/ — run earlier pipeline steps first.", filename)
    return pd.DataFrame()


def load_fuel_detail()        -> pd.DataFrame: return _load_csv("fuel_cost_detail.csv",        ["report_period"])
def load_labor_detail()       -> pd.DataFrame: return _load_csv("labor_cost_detail.csv",       ["report_period"])
def load_maintenance_detail() -> pd.DataFrame: return _load_csv("maintenance_cost_detail.csv", ["report_period"])


def load_demand_summary() -> pd.DataFrame:
    p2 = _P2_OUTPUT_DIR / "route_demand_summary.csv"
    if p2.exists():
        return pd.read_csv(p2, parse_dates=["report_period"])
    return _load_csv("route_demand_summary.csv", ["report_period"])


# =============================================================================
# Airport / Navigation fees
# =============================================================================

def _stage_bucket(distance_mi) -> str:
    try:
        d = float(distance_mi)
    except (TypeError, ValueError):
        return "medium"
    if d < 250:   return "ultra_short"
    if d < 500:   return "short"
    if d < 1000:  return "medium"
    if d < 2000:  return "long"
    return "ultra_long"


def compute_airport_fees(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dist_col = "distance_mi" if "distance_mi" in df.columns else None
    if dist_col:
        df["stage_bucket"]        = df[dist_col].apply(_stage_bucket)
        df["airport_fee_per_dep"] = df["stage_bucket"].map(AIRPORT_FEE_BY_STAGE)
    else:
        df["stage_bucket"]        = "medium"
        df["airport_fee_per_dep"] = AIRPORT_FEE_BY_STAGE["medium"]

    deps = df.get("departures", pd.Series(1, index=df.index))
    df["airport_fees_usd"] = (df["airport_fee_per_dep"] * deps).round(2)

    asm = df.get("asm", pd.Series(0, index=df.index))
    df["casm_airport"] = np.where(asm > 0, (df["airport_fees_usd"] / asm).round(7), np.nan)
    return df


# =============================================================================
# Ownership costs
# =============================================================================

_DYNAMIC_OWNERSHIP: dict[str, float] = {}   # populated once from EDGAR on first use


def _load_dynamic_ownership(form41_df: pd.DataFrame | None = None) -> dict[str, float]:
    """
    Derive ownership $/BH from live EDGAR D&A if available; fall back to
    static OWNERSHIP_PER_BH constants.
    """
    global _DYNAMIC_OWNERSHIP
    if _DYNAMIC_OWNERSHIP:
        return _DYNAMIC_OWNERSHIP

    if _EDGAR_AVAILABLE:
        try:
            da_df = fetch_ownership_from_xbrl(years=3)
            if not da_df.empty:
                derived = derive_ownership_per_block_hour(da_df, form41_df)
                if derived:
                    _DYNAMIC_OWNERSHIP = derived
                    logger.info("Dynamic ownership rates loaded from EDGAR D&A.")
                    return _DYNAMIC_OWNERSHIP
        except Exception as exc:
            logger.warning("EDGAR ownership fetch failed (%s); using static table.", exc)

    logger.info("Using static OWNERSHIP_PER_BH table.")
    _DYNAMIC_OWNERSHIP = dict(OWNERSHIP_PER_BH)
    return _DYNAMIC_OWNERSHIP


def compute_ownership_costs(df: pd.DataFrame, form41_df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df.copy()
    ownership_rates = _load_dynamic_ownership(form41_df)
    df["ownership_per_bh"]   = df["aircraft_variant"].map(ownership_rates).fillna(FALLBACK_OWNERSHIP_PER_BH)
    df["ownership_cost_usd"] = (df["ownership_per_bh"] * df.get("block_hours", pd.Series(1.0, index=df.index))).round(2)

    asm = df.get("asm", pd.Series(0, index=df.index))
    df["casm_ownership"] = np.where(asm > 0, (df["ownership_cost_usd"] / asm).round(7), np.nan)
    return df


# =============================================================================
# Overhead / G&A
# =============================================================================

def compute_overhead(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    asm = df.get("asm", pd.Series(0, index=df.index))
    df["overhead_cost_usd"] = (asm * OVERHEAD_PER_ASM).round(2)
    df["casm_overhead"]     = np.where(asm > 0, OVERHEAD_PER_ASM, np.nan)
    return df


# =============================================================================
# Assemble route_cost_summary
# =============================================================================

def assemble_cost_summary(
    fuel_df:        pd.DataFrame,
    labor_df:       pd.DataFrame,
    maintenance_df: pd.DataFrame,
    demand_df:      pd.DataFrame,
) -> pd.DataFrame:
    """
    Join fuel, labor, and maintenance components; add airport fees,
    ownership, and overhead to produce a full per-route cost breakdown.
    """
    if fuel_df.empty and demand_df.empty:
        logger.warning("No fuel or demand data; cannot assemble cost summary.")
        return pd.DataFrame()

    # ── Start from fuel detail (richest base; has block_hours already) ─────────
    if not fuel_df.empty:
        base_cols = [c for c in [
            "report_period", "carrier_code", "origin_iata", "destination_iata",
            "aircraft_variant", "departures", "distance_mi", "asm",
            "block_hours", "fuel_gallons_est", "jet_a_price_usd",
            "fuel_cost_usd", "casm_fuel",
        ] if c in fuel_df.columns]
        base = fuel_df[base_cols].copy()
    else:
        base = demand_df.copy()

    base["report_period"] = pd.to_datetime(base["report_period"])

    # ── Merge labor ────────────────────────────────────────────────────────────
    if not labor_df.empty:
        labor_cols = [c for c in KEY_COLS + ["pilot_cost_usd", "fa_cost_usd",
                                              "crew_cost_usd", "casm_crew"]
                      if c in labor_df.columns]
        l = labor_df[labor_cols].copy()
        l["report_period"] = pd.to_datetime(l["report_period"])
        merge_on = [c for c in KEY_COLS if c in base.columns and c in l.columns]
        base = base.merge(l, on=merge_on, how="left")

    # ── Merge maintenance ──────────────────────────────────────────────────────
    if not maintenance_df.empty:
        maint_cols = [c for c in KEY_COLS + ["maintenance_cost_usd", "casm_maintenance"]
                      if c in maintenance_df.columns]
        m = maintenance_df[maint_cols].copy()
        m["report_period"] = pd.to_datetime(m["report_period"])
        merge_on = [c for c in KEY_COLS if c in base.columns and c in m.columns]
        base = base.merge(m, on=merge_on, how="left")

    # ── Airport fees, ownership, overhead ─────────────────────────────────────
    base = compute_airport_fees(base)
    base = compute_ownership_costs(base, form41_df=demand_df if not demand_df.empty else None)
    base = compute_overhead(base)

    # ── Totals ─────────────────────────────────────────────────────────────────
    direct_cols = ["fuel_cost_usd", "crew_cost_usd", "maintenance_cost_usd", "airport_fees_usd"]
    full_cols   = direct_cols + ["ownership_cost_usd", "overhead_cost_usd"]

    for col in full_cols:
        if col not in base.columns:
            base[col] = 0.0
        base[col] = base[col].fillna(0.0)

    base["total_direct_cost_usd"] = base[direct_cols].sum(axis=1).round(2)
    base["total_cost_usd"]        = base[full_cols].sum(axis=1).round(2)

    asm = base.get("asm", pd.Series(0, index=base.index))
    base["casm_direct"] = np.where(asm > 0, (base["total_direct_cost_usd"] / asm).round(7), np.nan)
    base["casm_total"]  = np.where(asm > 0, (base["total_cost_usd"]        / asm).round(7), np.nan)

    base["cost_basis"] = "estimated"
    logger.info("Route cost summary assembled: %d rows.", len(base))
    return base


# =============================================================================
# BTS Form 41 peer benchmarking
# =============================================================================

def load_form41_data() -> pd.DataFrame:
    if _BTS_EXTRACTOR:
        try:
            return load_form41()
        except Exception as e:
            logger.warning("extract_bts_form41 failed: %s", e)

    for candidate in [
        _P1_DATA_DIR / "bts_form41_summary.csv",
        _P1_DATA_DIR / "bts_form41_raw.csv",
        _OUTPUT_DIR  / "bts_form41_summary.csv",
    ]:
        if candidate.exists():
            logger.info("Loading Form 41 from %s", candidate)
            df = pd.read_csv(candidate, parse_dates=["report_quarter"])
            return df

    logger.warning("No BTS Form 41 data found; peer benchmarks will be empty.")
    return pd.DataFrame()


def build_carrier_casm_benchmarks(form41_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute quarterly cost actuals per carrier from BTS Form 41.
    ASM is estimated via a revenue-based proxy (RASM ≈ 12.5 ¢/ASM) since
    Form 41 does not directly report ASM.
    """
    if form41_df.empty:
        return pd.DataFrame()

    group_cols = ["report_quarter", "carrier_code"]
    if "carrier_name" in form41_df.columns:
        group_cols.append("carrier_name")

    agg_map = {
        col: "sum" for col in [
            "fuel_gallons", "fuel_cost_usd", "block_hours", "departures",
            "salary_pilots_usd", "salary_copilots_usd", "salary_fa_usd",
            "op_revenues_usd", "op_expenses_usd",
        ] if col in form41_df.columns
    }
    if not agg_map:
        logger.warning("Form 41 lacks expected cost columns.")
        return pd.DataFrame()

    bench = (
        form41_df
        .groupby(group_cols, dropna=False)
        .agg(agg_map)
        .reset_index()
    )

    bench["fuel_cost_per_gal"] = (
        bench.get("fuel_cost_usd", 0) /
        bench.get("fuel_gallons", pd.Series(np.nan, index=bench.index)).replace(0, np.nan)
    ).round(4)

    bench["gph_block_hour"] = (
        bench.get("fuel_gallons", 0) /
        bench.get("block_hours", pd.Series(np.nan, index=bench.index)).replace(0, np.nan)
    ).round(1)

    bench["total_salary_usd"] = (
        bench.get("salary_pilots_usd",   pd.Series(0, index=bench.index)).fillna(0)
        + bench.get("salary_copilots_usd", pd.Series(0, index=bench.index)).fillna(0)
        + bench.get("salary_fa_usd",       pd.Series(0, index=bench.index)).fillna(0)
    )

    # ASM estimated from revenue ÷ network-carrier RASM proxy
    RASM_PROXY = 0.125   # $/ASM (≈ 12.5 ¢)
    bench["asm_est"] = (
        bench.get("op_revenues_usd", pd.Series(np.nan, index=bench.index))
        / RASM_PROXY
    ).round(0).astype("Int64")

    for metric, cost_col in [
        ("casm_total_cents",  "op_expenses_usd"),
        ("casm_fuel_cents",   "fuel_cost_usd"),
        ("casm_labor_cents",  "total_salary_usd"),
        ("rasm_cents",        "op_revenues_usd"),
    ]:
        bench[metric] = np.where(
            bench["asm_est"].fillna(0) > 0,
            (bench.get(cost_col, 0) / bench["asm_est"] * 100).round(4),
            np.nan,
        )

    bench["carrier_name"] = bench.get("carrier_code", pd.Series(dtype=str)).map(CARRIER_NAMES)
    bench["source"] = "bts_form41"
    logger.info("Carrier CASM benchmarks: %d rows.", len(bench))
    return bench


# =============================================================================
# Assemble casm_cost_feed for Project 4
# =============================================================================

def assemble_casm_feed(
    cost_summary_df: pd.DataFrame,
    benchmarks_df:   pd.DataFrame,
) -> pd.DataFrame:
    if cost_summary_df.empty:
        return pd.DataFrame()

    df = cost_summary_df.copy()

    # Attach peer-average CASM benchmark by quarter
    if not benchmarks_df.empty and "casm_total_cents" in benchmarks_df.columns:
        peers = benchmarks_df[benchmarks_df["carrier_code"] != "UA"].copy()
        if not peers.empty:
            qcol = "report_quarter" if "report_quarter" in peers.columns else "report_period"
            peers["_q"] = pd.to_datetime(peers[qcol]).dt.to_period("Q").dt.to_timestamp()
            peer_avg = (
                peers.groupby("_q")["casm_total_cents"]
                .mean()
                .rename("industry_casm_total_cents")
                .reset_index()
                .rename(columns={"_q": "report_period_q"})
            )
            df["report_period_q"] = pd.to_datetime(df["report_period"]).dt.to_period("Q").dt.to_timestamp()
            df = df.merge(peer_avg, on="report_period_q", how="left")
            df["ua_vs_peer_casm_delta_pct"] = np.where(
                df["industry_casm_total_cents"].fillna(0) > 0,
                ((df["casm_total"] * 100 - df["industry_casm_total_cents"])
                 / df["industry_casm_total_cents"]).round(4),
                np.nan,
            )
            df = df.drop(columns=["report_period_q"], errors="ignore")

    feed_cols = [c for c in [
        "report_period", "carrier_code", "origin_iata", "destination_iata",
        "aircraft_variant",
        "casm_fuel", "casm_crew", "casm_maintenance", "casm_airport",
        "casm_ownership", "casm_overhead", "casm_direct", "casm_total",
        "total_cost_usd", "total_direct_cost_usd", "fuel_cost_usd", "crew_cost_usd",
        "block_hours", "asm",
        "industry_casm_total_cents", "ua_vs_peer_casm_delta_pct",
        "cost_basis",
    ] if c in df.columns]

    out = df[feed_cols].copy()
    logger.info("CASM cost feed assembled: %d rows.", len(out))
    return out


# =============================================================================
# Summary statistics
# =============================================================================

def casm_summary(cost_df: pd.DataFrame) -> dict:
    if cost_df.empty:
        return {}

    def _mean(col):
        return round(float(cost_df[col].mean()), 6) if col in cost_df.columns else None

    return {
        "n_routes":              cost_df.groupby(["origin_iata", "destination_iata"]).ngroups
                                 if "origin_iata" in cost_df else 0,
        "n_aircraft_variants":   cost_df["aircraft_variant"].nunique()
                                 if "aircraft_variant" in cost_df else 0,
        "mean_casm_total":       _mean("casm_total"),
        "mean_casm_fuel":        _mean("casm_fuel"),
        "mean_casm_crew":        _mean("casm_crew"),
        "mean_casm_maintenance": _mean("casm_maintenance"),
        "mean_casm_airport":     _mean("casm_airport"),
        "mean_casm_ownership":   _mean("casm_ownership"),
        "mean_total_cost_usd":   round(float(cost_df["total_cost_usd"].mean()), 2)
                                 if "total_cost_usd" in cost_df else None,
    }


# =============================================================================
# Entrypoint
# =============================================================================

def run(source: str = "csv", save_csv: bool = True) -> dict[str, pd.DataFrame]:
    """
    Full CASM assembly pipeline.

    Returns:
        dict with keys: 'cost_summary', 'benchmarks', 'casm_feed'
    """
    fuel_df        = load_fuel_detail()
    labor_df       = load_labor_detail()
    maintenance_df = load_maintenance_detail()
    demand_df      = load_demand_summary()

    cost_summary_df = assemble_cost_summary(fuel_df, labor_df, maintenance_df, demand_df)

    form41_df     = load_form41_data()
    benchmarks_df = build_carrier_casm_benchmarks(form41_df)

    # Fall back to a pre-generated carrier_casm_benchmarks.csv if Form 41 unavailable
    if benchmarks_df.empty:
        pre_built = _OUTPUT_DIR / "carrier_casm_benchmarks.csv"
        if pre_built.exists():
            benchmarks_df = pd.read_csv(pre_built, parse_dates=["report_quarter"])
            logger.info("Loaded pre-built carrier_casm_benchmarks.csv: %d rows.", len(benchmarks_df))

    casm_feed_df  = assemble_casm_feed(cost_summary_df, benchmarks_df)

    if save_csv:
        if not cost_summary_df.empty:
            cost_summary_df.to_csv(_OUTPUT_DIR / "route_cost_summary.csv", index=False)
            logger.info("Saved route_cost_summary.csv")
        if not benchmarks_df.empty:
            benchmarks_df.to_csv(_OUTPUT_DIR / "carrier_casm_benchmarks.csv", index=False)
            logger.info("Saved carrier_casm_benchmarks.csv")
        if not casm_feed_df.empty:
            casm_feed_df.to_csv(_OUTPUT_DIR / "casm_cost_feed.csv", index=False)
            logger.info("Saved casm_cost_feed.csv → %s", _OUTPUT_DIR)

    if not cost_summary_df.empty:
        logger.info("CASM summary: %s", casm_summary(cost_summary_df))

    return {
        "cost_summary": cost_summary_df,
        "benchmarks":   benchmarks_df,
        "casm_feed":    casm_feed_df,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    results = run(source="csv")
    for name, df in results.items():
        if not df.empty:
            print(f"\n── {name}: {len(df)} rows ──")
            print(df.head(5).to_string(index=False))
    if not results["cost_summary"].empty:
        print("\nCASM summary:")
        for k, v in casm_summary(results["cost_summary"]).items():
            print(f"  {k}: {v}")
