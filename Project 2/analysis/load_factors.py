"""
load_factors.py
Compute, validate, impute, and flag load factors for the route demand analysis.

Load Factor (LF) = passengers / seats_available

Inputs:
  - bts_t100_segments  (primary source — monthly route-level traffic)
  - flight_capacity    (for seat configs where T-100 is absent)

Outputs:
  - route_demand_summary table (upserted)
  - load_factor_imputed table  (upserted)
  - CSV export → outputs/load_factor_feed.csv

Imputation hierarchy:
  1. Direct T-100 value (preferred)
  2. Route × aircraft_variant rolling 6-month average
  3. Route × carrier rolling 12-month average
  4. Carrier-wide monthly average for that aircraft type
  5. Global fleet average for that month

Outlier definition: |LF - route_mean| > N_SIGMA * route_std  (default N=3)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Allow running standalone from the analysis/ directory
_HERE = Path(__file__).resolve().parent
_PROJECT1_ETL = _HERE.parents[1] / "Project 1" / "etl"
if str(_PROJECT1_ETL) not in sys.path:
    sys.path.insert(0, str(_PROJECT1_ETL))

try:
    from load import get_connection, _upsert
    from config import DATA_DIR as P1_DATA_DIR
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

_OUTPUT_DIR = _HERE.parent / "outputs"
_OUTPUT_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

# ── Tuning parameters ────────────────────────────────────────────────────────
N_SIGMA              = 3.0    # z-score threshold for outlier flagging
MIN_OBS_FOR_ROUTE    = 3      # minimum months of data to compute route average
ROLLING_WINDOW_SHORT = 6      # months for short rolling avg
ROLLING_WINDOW_LONG  = 12     # months for long rolling avg
BTS_CARRIER_CODE     = "UA"   # filter to United for route_demand_summary

# ── BTS aircraft type → United variant (subset of Project 1 map) ─────────────
BTS_VARIANT_MAP: dict[str, str] = {
    "77W": "B777-300ER",   "772": "B777-200",
    "788": "B787-8",        "789": "B787-9 Version 1",  "78X": "B787-10",
    "763": "767-300ER Version 1",                        "764": "767-400ER",
    "752": "757-200",       "753": "757-300",
    "737": "737-700",       "738": "737-800 Version 1",  "739": "737-900 Version 1",
    "7M8": "737 MAX 8 Version 1",                        "7M9": "737 MAX 9 Version 1",
    "319": "A319",          "320": "A320",               "32Q": "A321neo",
    "CRJ": "CRJ200",        "CR5": "CRJ550",             "CR7": "CRJ700",
    "E70": "Embraer E170",  "E75": "Embraer E175 Version 1",
}


# =============================================================================
# Load data
# =============================================================================

def load_t100_from_db(conn) -> pd.DataFrame:
    """Pull BTS T-100 data from Project 1 warehouse."""
    sql = """
        SELECT
            report_period,
            carrier_code,
            aircraft_type_bts,
            origin_iata,
            destination_iata,
            departures_performed AS departures,
            seats_available,
            passengers,
            load_factor,
            payload_lbs,
            distance_mi,
            asm,
            rpm
        FROM bts_t100_segments
        ORDER BY report_period, carrier_code, origin_iata, destination_iata
    """
    return pd.read_sql(sql, conn)


def load_t100_from_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """Fallback: load T-100 from CSV."""
    if path is None:
        # Check Project 1 data directory first
        candidates = [
            P1_DATA_DIR / "bts_t100_UA.csv" if _DB_AVAILABLE else None,
            _HERE.parent.parent / "Project 1" / "data" / "bts_t100_UA.csv",
            _HERE.parent / "outputs" / "bts_t100_UA.csv",
        ]
        for c in candidates:
            if c and c.exists():
                path = c
                break
    if path is None or not path.exists():
        logger.warning("No T-100 CSV found; returning empty DataFrame.")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["report_period"])
    logger.info("Loaded T-100 from %s: %d rows", path, len(df))
    return df


# =============================================================================
# Core computations
# =============================================================================

def compute_load_factors(t100_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute or validate load factors from T-100 data.

    T-100 reports seats_available and passengers directly.
    LF = passengers / seats_available.

    Parameters
    ----------
    t100_df : DataFrame from load_t100_from_db() or load_t100_from_csv()

    Returns
    -------
    DataFrame with added column: lf_computed (recomputed from raw values),
    lf_delta (difference from reported LF, for QA).
    """
    df = t100_df.copy()

    # Map BTS aircraft type codes → United variant names
    if "aircraft_type_bts" in df.columns:
        df["aircraft_variant"] = df["aircraft_type_bts"].map(BTS_VARIANT_MAP)

    # Recompute LF from raw values
    df["lf_computed"] = np.where(
        df["seats_available"] > 0,
        (df["passengers"] / df["seats_available"]).clip(0, 1),
        np.nan,
    )

    # Use recomputed; keep original for QA
    if "load_factor" in df.columns:
        df["lf_delta"] = (df["lf_computed"] - df["load_factor"]).abs()
    else:
        df["load_factor"] = df["lf_computed"]
        df["lf_delta"] = np.nan

    df["load_factor"] = df["lf_computed"]

    # ASM / RPM
    if "asm" not in df.columns:
        df["asm"] = df["seats_available"] * df["distance_mi"]
    if "rpm" not in df.columns:
        df["rpm"] = df["passengers"] * df["distance_mi"]

    return df


def flag_outliers(
    df: pd.DataFrame,
    lf_col: str = "load_factor",
    group_cols: tuple = ("origin_iata", "destination_iata", "aircraft_type_bts"),
    n_sigma: float = N_SIGMA,
) -> pd.DataFrame:
    """
    Flag load factors that are more than n_sigma standard deviations from
    the route-level mean.

    Adds columns:
        lf_route_mean, lf_route_std, lf_z_score, lf_outlier_flag
    """
    df = df.copy()

    # Compute route-level mean and std
    route_stats = (
        df.groupby(list(group_cols))[lf_col]
        .agg(lf_route_mean="mean", lf_route_std="std")
        .reset_index()
    )
    df = df.merge(route_stats, on=list(group_cols), how="left")

    # z-score
    df["lf_z_score"] = (
        (df[lf_col] - df["lf_route_mean"]) / df["lf_route_std"].replace(0, np.nan)
    )

    df["lf_outlier_flag"] = df["lf_z_score"].abs() > n_sigma
    df["lf_outlier_flag"] = df["lf_outlier_flag"].fillna(False)

    n_flagged = df["lf_outlier_flag"].sum()
    if n_flagged > 0:
        logger.info("Flagged %d load factor outliers (|z| > %.1f).", n_flagged, n_sigma)

    return df


def impute_load_factors(
    df: pd.DataFrame,
    lf_col: str = "load_factor",
) -> pd.DataFrame:
    """
    Impute missing (or outlier) load factors using a hierarchy of averages.

    Imputation hierarchy (applied in order):
      1. Direct T-100 value (pass-through)
      2. Rolling 6-month route × aircraft_variant average
      3. Rolling 12-month route (any aircraft) average
      4. Carrier-wide monthly average for that aircraft_variant
      5. Global fleet monthly average

    Adds columns:
        lf_imputed, imputation_method, lf_source
    """
    df = df.copy()
    if "report_period" in df.columns:
        df["report_period"] = pd.to_datetime(df["report_period"])
    df = df.sort_values("report_period").reset_index(drop=True)

    # ── Step 1: use direct value where not null and not flagged ──────────────
    df["lf_imputed"] = np.where(
        ~df["lf_outlier_flag"].fillna(False) & df[lf_col].notna(),
        df[lf_col],
        np.nan,
    )
    df["imputation_method"] = np.where(df["lf_imputed"].notna(), "direct", None)

    needs_imputation = df["lf_imputed"].isna()
    if needs_imputation.sum() == 0:
        df["lf_source"] = "T100"
        return df

    # ── Step 2: rolling 6-month route × variant average ──────────────────────
    route_variant_avg = (
        df[~df["lf_outlier_flag"].fillna(False)]
        .groupby(["origin_iata", "destination_iata", "aircraft_type_bts"])
        .apply(lambda g: g.set_index("report_period")[lf_col]
                          .rolling(f"{ROLLING_WINDOW_SHORT * 30}D", min_periods=MIN_OBS_FOR_ROUTE)
                          .mean()
               )
        .reset_index(name="lf_rolling_rv")
    )
    if "report_period" not in route_variant_avg.columns:
        route_variant_avg = route_variant_avg.rename(columns={"level_3": "report_period"})

    if not route_variant_avg.empty and "report_period" in route_variant_avg.columns:
        df = df.merge(
            route_variant_avg[["origin_iata", "destination_iata",
                               "aircraft_type_bts", "report_period", "lf_rolling_rv"]],
            on=["origin_iata", "destination_iata", "aircraft_type_bts", "report_period"],
            how="left",
        )
        mask2 = needs_imputation & df["lf_rolling_rv"].notna()
        df.loc[mask2, "lf_imputed"] = df.loc[mask2, "lf_rolling_rv"]
        df.loc[mask2, "imputation_method"] = "route_variant_rolling6m"
        needs_imputation = df["lf_imputed"].isna()

    # ── Step 3: route × period average across all aircraft types ─────────────
    route_avg = (
        df[~df["lf_outlier_flag"].fillna(False)]
        .groupby(["origin_iata", "destination_iata", "report_period"])[lf_col]
        .mean()
        .rename("lf_route_period_avg")
        .reset_index()
    )
    df = df.merge(route_avg, on=["origin_iata", "destination_iata", "report_period"], how="left")
    mask3 = needs_imputation & df["lf_route_period_avg"].notna()
    df.loc[mask3, "lf_imputed"] = df.loc[mask3, "lf_route_period_avg"]
    df.loc[mask3, "imputation_method"] = "route_avg_all_aircraft"
    needs_imputation = df["lf_imputed"].isna()

    # ── Step 4: carrier-wide aircraft_variant × period average ───────────────
    variant_avg = (
        df[~df["lf_outlier_flag"].fillna(False)]
        .groupby(["aircraft_type_bts", "report_period"])[lf_col]
        .mean()
        .rename("lf_variant_avg")
        .reset_index()
    )
    df = df.merge(variant_avg, on=["aircraft_type_bts", "report_period"], how="left")
    mask4 = needs_imputation & df["lf_variant_avg"].notna()
    df.loc[mask4, "lf_imputed"] = df.loc[mask4, "lf_variant_avg"]
    df.loc[mask4, "imputation_method"] = "carrier_variant_avg"
    needs_imputation = df["lf_imputed"].isna()

    # ── Step 5: global fleet monthly average ─────────────────────────────────
    global_avg = (
        df[~df["lf_outlier_flag"].fillna(False)]
        .groupby("report_period")[lf_col]
        .mean()
        .rename("lf_global_avg")
        .reset_index()
    )
    df = df.merge(global_avg, on="report_period", how="left")
    mask5 = needs_imputation & df["lf_global_avg"].notna()
    df.loc[mask5, "lf_imputed"] = df.loc[mask5, "lf_global_avg"]
    df.loc[mask5, "imputation_method"] = "global_monthly_avg"

    # Final fallback: fleet overall mean
    global_mean = df[~df["lf_outlier_flag"].fillna(False)][lf_col].mean()
    df["lf_imputed"] = df["lf_imputed"].fillna(global_mean)
    df.loc[df["imputation_method"].isna(), "imputation_method"] = "global_mean_fallback"

    # lf_source label
    df["lf_source"] = np.where(df["imputation_method"] == "direct", "T100", "imputed")

    n_imputed = (df["imputation_method"] != "direct").sum()
    logger.info(
        "Imputed %d / %d LF values.  Methods: %s",
        n_imputed,
        len(df),
        df["imputation_method"].value_counts().to_dict(),
    )
    return df


# =============================================================================
# Build route_demand_summary
# =============================================================================

def build_route_demand_summary(t100_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce the route_demand_summary DataFrame.

    1. Filter to United (carrier_code == 'UA')
    2. Compute LF
    3. Flag outliers
    4. Impute missing values
    """
    df = t100_df.copy()

    # Standardise period
    if "report_period" in df.columns:
        df["report_period"] = pd.to_datetime(df["report_period"])

    df = compute_load_factors(df)
    df = flag_outliers(df, lf_col="load_factor")
    df = impute_load_factors(df, lf_col="load_factor")

    # Map aircraft codes
    df["aircraft_variant"] = df.get("aircraft_type_bts", pd.Series(dtype=str)).map(BTS_VARIANT_MAP)

    out = df[[
        "report_period", "carrier_code",
        "origin_iata", "destination_iata",
        "aircraft_type_bts", "aircraft_variant",
        "departures", "seats_available", "passengers",
        "load_factor", "lf_imputed", "lf_source", "lf_outlier_flag",
        "payload_lbs", "distance_mi", "asm", "rpm",
    ]].rename(columns={"lf_imputed": "load_factor_imputed"})

    return out


# =============================================================================
# Build load_factor_imputed feed
# =============================================================================

def build_lf_imputed_feed(demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce the load_factor_imputed DataFrame (Project 4 input feed).
    """
    df = demand_df.copy()
    out = pd.DataFrame({
        "report_period":      df["report_period"],
        "carrier_code":       df["carrier_code"],
        "origin_iata":        df["origin_iata"],
        "destination_iata":   df["destination_iata"],
        "aircraft_variant":   df.get("aircraft_variant"),
        "lf_t100":            df["load_factor"],
        "lf_imputed":         df["load_factor_imputed"],
        "imputation_method":  df.get("lf_source", "direct"),
        "imputation_basis":   df.get("imputation_method", "direct"),
        "outlier_flag":       df["lf_outlier_flag"],
        "outlier_sigma":      df.get("lf_z_score"),
    })
    return out


# =============================================================================
# Summary statistics
# =============================================================================

def summarise_load_factors(demand_df: pd.DataFrame) -> dict:
    """
    Return a dict of summary statistics for the load factor analysis.
    """
    lf = demand_df["load_factor_imputed"].dropna()
    return {
        "n_routes":           demand_df.groupby(["origin_iata", "destination_iata"]).ngroups,
        "n_periods":          demand_df["report_period"].nunique(),
        "mean_lf":            round(float(lf.mean()), 4),
        "median_lf":          round(float(lf.median()), 4),
        "std_lf":             round(float(lf.std()), 4),
        "p10_lf":             round(float(lf.quantile(0.10)), 4),
        "p90_lf":             round(float(lf.quantile(0.90)), 4),
        "pct_outliers":       round(float(demand_df["lf_outlier_flag"].mean()), 4),
        "pct_imputed":        round(float((demand_df["lf_source"] != "T100").mean()), 4),
    }


# =============================================================================
# Entrypoint
# =============================================================================

def run(
    source: str = "db",
    csv_path: Optional[Path] = None,
    save_csv: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full load-factor pipeline.

    Parameters
    ----------
    source   : 'db' to load from PostgreSQL, 'csv' to load from file
    csv_path : path override for CSV source
    save_csv : whether to write output CSVs to outputs/

    Returns
    -------
    (demand_df, lf_imputed_df)
    """
    # ── Load raw data ─────────────────────────────────────────────────────────
    if source == "db" and _DB_AVAILABLE:
        conn = get_connection()
        try:
            t100_df = load_t100_from_db(conn)
        finally:
            conn.close()
    else:
        t100_df = load_t100_from_csv(csv_path)

    if t100_df.empty:
        logger.warning("No T-100 data available. Returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    # ── Filter to United ──────────────────────────────────────────────────────
    ua_df = t100_df[t100_df["carrier_code"] == BTS_CARRIER_CODE].copy()
    logger.info("T-100 rows for UA: %d", len(ua_df))

    # ── Compute demand summary ────────────────────────────────────────────────
    demand_df   = build_route_demand_summary(ua_df)
    lf_feed_df  = build_lf_imputed_feed(demand_df)
    stats       = summarise_load_factors(demand_df)

    logger.info("Load factor stats: %s", stats)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if save_csv:
        demand_df.to_csv(_OUTPUT_DIR / "route_demand_summary.csv", index=False)
        lf_feed_df.to_csv(_OUTPUT_DIR / "load_factor_feed.csv", index=False)
        logger.info("Saved route_demand_summary.csv and load_factor_feed.csv → %s", _OUTPUT_DIR)

    # ── Upsert to DB ──────────────────────────────────────────────────────────
    if _DB_AVAILABLE:
        conn = get_connection()
        try:
            n1 = _upsert(conn, demand_df, "route_demand_summary",
                         conflict_cols=["report_period", "carrier_code",
                                        "origin_iata", "destination_iata", "aircraft_type_bts"],
                         update_cols=["load_factor", "load_factor_imputed", "lf_source",
                                      "lf_outlier_flag", "seats_available", "passengers",
                                      "asm", "rpm"])
            n2 = _upsert(conn, lf_feed_df, "load_factor_imputed",
                         conflict_cols=["report_period", "carrier_code",
                                        "origin_iata", "destination_iata", "aircraft_variant"],
                         update_cols=["lf_t100", "lf_imputed", "imputation_method",
                                      "outlier_flag", "outlier_sigma"])
            logger.info("Upserted %d demand rows, %d LF imputed rows.", n1, n2)
        finally:
            conn.close()

    return demand_df, lf_feed_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    d, lf = run(source="csv")
    if not d.empty:
        print(f"\nRoute demand summary: {len(d)} rows")
        print(d.head(10).to_string(index=False))
        print(f"\nLoad factor stats:")
        for k, v in summarise_load_factors(d).items():
            print(f"  {k}: {v}")
