"""
demand.py
Route-level demand trend and seasonality analysis.

Outputs:
  - Monthly seasonality indices by route (Passenger and Load Factor)
  - Year-over-year growth rates
  - Top routes by demand and profitability potential
  - Demand volatility scores

Seasonality index = month_value / trailing_12m_monthly_avg
  > 1.0 → above-average month
  < 1.0 → below-average month

Inputs:
  - route_demand_summary (from load_factors.py) or bts_t100_segments DB table

Outputs:
  - route_seasonality table (upserted)
  - CSV → outputs/route_seasonality.csv
  - CSV → outputs/demand_trends.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

_HERE = Path(__file__).resolve().parent
_PROJECT1_ETL = _HERE.parents[1] / "Project 1" / "etl"
if str(_PROJECT1_ETL) not in sys.path:
    sys.path.insert(0, str(_PROJECT1_ETL))

try:
    from load import get_connection, _upsert
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

_OUTPUT_DIR = _HERE.parent / "outputs"
_OUTPUT_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

# ── Parameters ───────────────────────────────────────────────────────────────
MIN_MONTHS_FOR_SEASONALITY = 12   # need at least 12 months for a valid index
MIN_MONTHS_FOR_YOY          = 13  # need 13+ months for any YoY comparison
CARRIER_CODE                = "UA"


# =============================================================================
# Load data
# =============================================================================

def load_demand_from_db(conn) -> pd.DataFrame:
    """Pull route_demand_summary from Project 2 warehouse."""
    sql = """
        SELECT
            report_period, carrier_code,
            origin_iata, destination_iata,
            aircraft_type_bts, aircraft_variant,
            departures, seats_available, passengers,
            load_factor_imputed AS load_factor,
            lf_source, lf_outlier_flag,
            distance_mi, asm, rpm
        FROM route_demand_summary
        WHERE carrier_code = %s
        ORDER BY report_period
    """
    return pd.read_sql(sql, conn, params=(CARRIER_CODE,),
                       parse_dates=["report_period"])


def load_demand_from_csv(path: Optional[Path] = None) -> pd.DataFrame:
    candidates = [
        _HERE.parent / "outputs" / "route_demand_summary.csv",
        _HERE.parents[1] / "Project 1" / "data" / "bts_t100_UA.csv",
    ]
    if path:
        candidates.insert(0, path)
    for c in candidates:
        if c.exists():
            df = pd.read_csv(c, parse_dates=["report_period"])
            logger.info("Loaded demand data from %s: %d rows", c, len(df))
            return df
    logger.warning("No demand CSV found.")
    return pd.DataFrame()


# =============================================================================
# Seasonality index
# =============================================================================

def compute_seasonality_indices(
    demand_df: pd.DataFrame,
    base_years: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Compute monthly seasonality indices by (carrier, origin, destination).

    For each route, for each calendar month:
        seasonality_index_pax = avg_month_pax / (total_annual_pax / 12)
        seasonality_index_lf  = avg_month_lf  / annual_avg_lf

    Also computes average YoY growth rate using linear regression on log(passengers).

    Parameters
    ----------
    demand_df  : route_demand_summary DataFrame
    base_years : list of years to include in the calculation (None = all years)

    Returns
    -------
    DataFrame matching route_seasonality table schema
    """
    df = demand_df.copy()
    df["report_period"] = pd.to_datetime(df["report_period"])
    df["year"]          = df["report_period"].dt.year
    df["month"]         = df["report_period"].dt.month

    if base_years:
        df = df[df["year"].isin(base_years)]

    # Use imputed LF where available
    lf_col = "load_factor" if "load_factor" in df.columns else "load_factor_imputed"
    pax_col = "passengers"

    # Aggregate across aircraft types per route-month (sum pax, avg LF)
    route_month = (
        df.groupby(["carrier_code", "origin_iata", "destination_iata", "year", "month"])
        .agg(
            monthly_pax     =(pax_col, "sum"),
            monthly_seats   =("seats_available", "sum"),
            monthly_lf      =(lf_col, "mean"),
        )
        .reset_index()
    )

    # Annual average monthly pax per route
    route_annual_avg = (
        route_month.groupby(["carrier_code", "origin_iata", "destination_iata", "year"])
        ["monthly_pax"].mean()
        .rename("annual_avg_monthly_pax")
        .reset_index()
    )
    route_month = route_month.merge(
        route_annual_avg,
        on=["carrier_code", "origin_iata", "destination_iata", "year"],
    )

    # Annual average monthly LF per route
    route_annual_lf = (
        route_month.groupby(["carrier_code", "origin_iata", "destination_iata", "year"])
        ["monthly_lf"].mean()
        .rename("annual_avg_lf")
        .reset_index()
    )
    route_month = route_month.merge(
        route_annual_lf,
        on=["carrier_code", "origin_iata", "destination_iata", "year"],
    )

    # Seasonality indices
    route_month["si_pax"] = (
        route_month["monthly_pax"] /
        route_month["annual_avg_monthly_pax"].replace(0, np.nan)
    )
    route_month["si_lf"] = (
        route_month["monthly_lf"] /
        route_month["annual_avg_lf"].replace(0, np.nan)
    )

    # Average across all years per route-month
    season = (
        route_month.groupby(["carrier_code", "origin_iata", "destination_iata", "month"])
        .agg(
            avg_monthly_pax         =("monthly_pax",  "mean"),
            avg_monthly_seats       =("monthly_seats","mean"),
            avg_load_factor         =("monthly_lf",   "mean"),
            seasonality_index_pax   =("si_pax",       "mean"),
            seasonality_index_lf    =("si_lf",        "mean"),
            n_years                 =("year",          "nunique"),
        )
        .round(4)
        .reset_index()
        .rename(columns={"month": "month_of_year"})
    )

    # Filter to routes with enough data
    enough_data = season["n_years"] >= 1  # at least 1 year complete
    season = season[enough_data].copy()

    # YoY growth rate (linear regression on log pax by year, per route)
    yoy_rates = _compute_yoy_growth(route_month)
    season = season.merge(yoy_rates, on=["carrier_code", "origin_iata", "destination_iata"],
                          how="left")

    # base_year: the most recent year in the dataset
    season["base_year"] = df["year"].max()
    season["computed_at"] = pd.Timestamp.utcnow()

    logger.info("Computed seasonality for %d route-month combinations.", len(season))
    return season


def _compute_yoy_growth(route_month_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate year-over-year passenger growth rate per route using
    linear regression on ln(annual_pax) ~ year.

    Returns a DataFrame with columns:
        carrier_code, origin_iata, destination_iata, yoy_growth_rate
    """
    rows = []
    for (carrier, orig, dest), grp in route_month_df.groupby(
        ["carrier_code", "origin_iata", "destination_iata"]
    ):
        annual = grp.groupby("year")["monthly_pax"].sum().reset_index()
        annual = annual[annual["monthly_pax"] > 0]
        if len(annual) < 2:
            rows.append({
                "carrier_code": carrier,
                "origin_iata": orig,
                "destination_iata": dest,
                "yoy_growth_rate": np.nan,
            })
            continue
        log_pax = np.log(annual["monthly_pax"].values)
        years   = annual["year"].values.astype(float)
        try:
            slope, _, _, _, _ = scipy_stats.linregress(years, log_pax)
            # slope ≈ ln(1 + growth_rate) for small rates
            yoy = float(np.exp(slope) - 1)
        except Exception:
            yoy = np.nan
        rows.append({
            "carrier_code": carrier,
            "origin_iata": orig,
            "destination_iata": dest,
            "yoy_growth_rate": round(yoy, 4),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Demand trends
# =============================================================================

def compute_demand_trends(demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling averages and trend statistics per route.

    Adds:
        rolling_3m_pax, rolling_6m_pax, rolling_12m_pax
        rolling_3m_lf,  rolling_6m_lf
        demand_trend  ('growing'|'stable'|'declining')
        demand_volatility (coefficient of variation of monthly pax)
    """
    df = demand_df.copy()
    df["report_period"] = pd.to_datetime(df["report_period"])

    lf_col  = "load_factor" if "load_factor" in df.columns else "load_factor_imputed"
    pax_col = "passengers"

    # Route-month aggregation
    route_monthly = (
        df.groupby(["carrier_code", "origin_iata", "destination_iata", "report_period"])
        .agg(pax=(pax_col, "sum"), lf=(lf_col, "mean"), seats=("seats_available", "sum"))
        .reset_index()
        .sort_values("report_period")
    )

    results = []
    for (carrier, orig, dest), grp in route_monthly.groupby(
        ["carrier_code", "origin_iata", "destination_iata"]
    ):
        grp = grp.set_index("report_period").sort_index()

        # Rolling averages (approximate window periods)
        grp["rolling_3m_pax"]  = grp["pax"].rolling(3,  min_periods=1).mean()
        grp["rolling_6m_pax"]  = grp["pax"].rolling(6,  min_periods=1).mean()
        grp["rolling_12m_pax"] = grp["pax"].rolling(12, min_periods=1).mean()
        grp["rolling_3m_lf"]   = grp["lf"].rolling(3,  min_periods=1).mean()
        grp["rolling_6m_lf"]   = grp["lf"].rolling(6,  min_periods=1).mean()

        # Trend: compare last 3m avg to prior 3m avg
        if len(grp) >= 6:
            recent = grp["pax"].iloc[-3:].mean()
            prior  = grp["pax"].iloc[-6:-3].mean()
            if prior > 0:
                change = (recent - prior) / prior
                if change > 0.05:
                    trend = "growing"
                elif change < -0.05:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        # Volatility: coefficient of variation of monthly pax
        cv = float(grp["pax"].std() / grp["pax"].mean()) if grp["pax"].mean() > 0 else np.nan

        grp["carrier_code"]       = carrier
        grp["origin_iata"]        = orig
        grp["destination_iata"]   = dest
        grp["demand_trend"]       = trend
        grp["demand_volatility"]  = round(cv, 4)

        results.append(grp.reset_index())

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].round(2)
    return out


# =============================================================================
# Route ranking
# =============================================================================

def rank_routes_by_demand(
    demand_df: pd.DataFrame,
    n_months: int = 12,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Rank routes by average monthly passenger volume and load factor
    over the most recent n_months.

    Returns
    -------
    Top top_n routes with demand stats.
    """
    df = demand_df.copy()
    df["report_period"] = pd.to_datetime(df["report_period"])
    cutoff = df["report_period"].max() - pd.DateOffset(months=n_months)
    recent = df[df["report_period"] > cutoff].copy()

    lf_col  = "load_factor" if "load_factor" in df.columns else "load_factor_imputed"

    ranked = (
        recent.groupby(["origin_iata", "destination_iata"])
        .agg(
            avg_monthly_pax     =("passengers",      "mean"),
            avg_lf              =(lf_col,            "mean"),
            avg_seats           =("seats_available", "mean"),
            avg_asm             =("asm",              "mean"),
            n_periods           =("report_period",   "nunique"),
        )
        .round(2)
        .reset_index()
        .sort_values("avg_monthly_pax", ascending=False)
        .head(top_n)
    )
    ranked["pax_rank"] = range(1, len(ranked) + 1)
    return ranked


# =============================================================================
# Entrypoint
# =============================================================================

def run(
    source: str = "db",
    csv_path: Optional[Path] = None,
    save_csv: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full demand analysis pipeline.

    Returns
    -------
    (seasonality_df, trends_df)
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    if source == "db" and _DB_AVAILABLE:
        conn = get_connection()
        try:
            demand_df = load_demand_from_db(conn)
        finally:
            conn.close()
    else:
        demand_df = load_demand_from_csv(csv_path)

    if demand_df.empty:
        logger.warning("No demand data. Returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    # ── Compute ───────────────────────────────────────────────────────────────
    seasonality_df = compute_seasonality_indices(demand_df)
    trends_df      = compute_demand_trends(demand_df)
    top_routes     = rank_routes_by_demand(demand_df)

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_csv:
        seasonality_df.to_csv(_OUTPUT_DIR / "route_seasonality.csv", index=False)
        trends_df.to_csv(     _OUTPUT_DIR / "demand_trends.csv",     index=False)
        top_routes.to_csv(    _OUTPUT_DIR / "top_routes_by_demand.csv", index=False)
        logger.info("Saved demand analysis CSVs → %s", _OUTPUT_DIR)

    # ── Upsert to DB ──────────────────────────────────────────────────────────
    if _DB_AVAILABLE and not seasonality_df.empty:
        conn = get_connection()
        try:
            n = _upsert(
                conn, seasonality_df, "route_seasonality",
                conflict_cols=["carrier_code", "origin_iata", "destination_iata",
                               "month_of_year", "base_year"],
                update_cols=["avg_monthly_pax", "avg_monthly_seats", "avg_load_factor",
                             "seasonality_index_pax", "seasonality_index_lf",
                             "yoy_growth_rate"],
            )
            logger.info("Upserted %d seasonality rows.", n)
        finally:
            conn.close()

    return seasonality_df, trends_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    s, t = run(source="csv")
    if not s.empty:
        print(f"\nSeasonality: {len(s)} rows")
        print(s.head(12).to_string(index=False))
    if not t.empty:
        print(f"\nDemand trends: {len(t)} rows")
        top = rank_routes_by_demand(t)
        print("\n── Top Routes by Demand ──")
        print(top.head(20).to_string(index=False))
