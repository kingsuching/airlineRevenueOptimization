"""
kpis.py
Assemble route-level KPIs for the Project 4 profitability model input feed.

KPIs computed:
  - Load Factor (from load_factors.py output, imputed where needed)
  - Estimated Revenue (LF × seats × blended_fare from DB1B)
  - Estimated Operational Costs (fuel + crew — full CASM in Project 3)
  - Operational Profit per Block Hour = (est_rev - op_cost) / block_hours
  - Seasonality Index (from demand.py)
  - OTP % and average delay (from operational.py)

This module joins outputs from load_factors.py, demand.py, operational.py
together with DB1B fare data and Project 1 cost tables to build the
route_kpi_feed table.

Output: outputs/route_kpi_feed.csv  (and upserted to DB)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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

# ── Cabin revenue allocation fractions ───────────────────────────────────────
# Used to blend a single revenue estimate when per-cabin fare data is absent.
# Based on typical United revenue mix (approximate).
CABIN_REVENUE_MIX = {
    "first":          0.12,   # ~12% of PAX revenue from first
    "business":       0.28,   # ~28% from business/Polaris
    "premium_plus":   0.08,
    "economy_plus":   0.15,
    "economy":        0.37,
}

# Blended fare markup vs economy baseline (DB1B)
CABIN_FARE_MULTIPLIER = {
    "first":          4.5,
    "business":       3.8,
    "premium_plus":   2.2,
    "economy_plus":   1.35,
    "economy":        1.0,
}

# Default blended fare if DB1B missing ($/passenger)
FALLBACK_BLENDED_FARE = 280.0


# =============================================================================
# Load input data
# =============================================================================

def _load_csv_if_exists(filename: str) -> pd.DataFrame:
    p = _OUTPUT_DIR / filename
    if p.exists():
        df = pd.read_csv(p, parse_dates=["report_period"] if "period" in filename or "demand" in filename else [])
        logger.info("Loaded %s: %d rows", filename, len(df))
        return df
    logger.warning("%s not found in outputs/", filename)
    return pd.DataFrame()


def load_demand_summary() -> pd.DataFrame:
    return _load_csv_if_exists("route_demand_summary.csv")


def load_operational_perf() -> pd.DataFrame:
    return _load_csv_if_exists("route_operational_perf.csv")


def load_seasonality() -> pd.DataFrame:
    return _load_csv_if_exists("route_seasonality.csv")


def load_db1b_fares_from_db(conn) -> pd.DataFrame:
    sql = """
        SELECT
            report_quarter,
            carrier_code,
            origin_iata,
            destination_iata,
            cabin_class_bts,
            avg_fare_usd,
            passengers,
            yield_per_mile
        FROM bts_db1b_fares
        WHERE carrier_code = 'UA'
    """
    return pd.read_sql(sql, conn, parse_dates=["report_quarter"])


def load_fuel_burn_from_db(conn) -> pd.DataFrame:
    sql = """
        SELECT aircraft_variant, gph_cruise
        FROM aircraft_fuel_burn
    """
    return pd.read_sql(sql, conn)


def load_crew_costs_from_db(conn) -> pd.DataFrame:
    sql = """
        SELECT aircraft_variant, pilot_cost_per_bh, fa_cost_per_bh, total_crew_cost_per_bh
        FROM labor_rates_ref
    """
    return pd.read_sql(sql, conn)


def load_fuel_prices_from_db(conn) -> pd.DataFrame:
    sql = """
        SELECT price_date, jet_a_usd_per_gal
        FROM fuel_prices
        ORDER BY price_date
    """
    return pd.read_sql(sql, conn, parse_dates=["price_date"])


def load_db1b_from_csv() -> pd.DataFrame:
    candidates = [
        _OUTPUT_DIR / "bts_db1b.csv",
        _HERE.parents[1] / "Project 1" / "data" / "bts_db1b.csv",
    ]
    for c in candidates:
        if c.exists():
            return pd.read_csv(c, parse_dates=["report_quarter"])
    return pd.DataFrame()


# =============================================================================
# Fare blending
# =============================================================================

def build_blended_fares(db1b_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a blended per-passenger fare per route-quarter from DB1B data.

    Also derive per-cabin fares by mapping BTS cabin labels to United tiers.
    """
    if db1b_df.empty:
        return pd.DataFrame()

    cabin_map = {
        "Coach":    "economy",
        "Business": "business",
        "First":    "first",
        "Premium":  "premium_plus",
    }
    df = db1b_df.copy()
    df["cabin_tier"] = df.get("cabin_class_bts", pd.Series(dtype=str)).map(cabin_map).fillna("economy")

    # Pivot to wide: one row per route-quarter with per-cabin fares
    pivot = df.pivot_table(
        index=["carrier_code", "origin_iata", "destination_iata", "report_quarter"],
        columns="cabin_tier",
        values="avg_fare_usd",
        aggfunc="mean",
    ).reset_index().rename_axis(None, axis=1)

    # Rename to schema columns
    col_map = {
        "economy":      "avg_fare_economy_usd",
        "business":     "avg_fare_business_usd",
        "first":        "avg_fare_first_usd",
        "premium_plus": "avg_fare_premium_plus_usd",
    }
    pivot = pivot.rename(columns=col_map)

    # Blended fare = weighted average across available cabins
    def _blend(row):
        total_weight = 0.0
        total_fare   = 0.0
        for cabin, weight in CABIN_REVENUE_MIX.items():
            col = f"avg_fare_{cabin}_usd"
            if col in row and pd.notna(row.get(col)):
                total_fare   += row[col] * weight
                total_weight += weight
        if total_weight > 0:
            return total_fare / total_weight
        return FALLBACK_BLENDED_FARE

    pivot["avg_fare_blended_usd"] = pivot.apply(_blend, axis=1)

    # Yield per mile (economy as proxy for overall)
    if "avg_fare_economy_usd" in pivot.columns:
        pivot["yield_per_mile_economy"] = df.groupby(
            ["carrier_code", "origin_iata", "destination_iata", "report_quarter"]
        ).apply(
            lambda g: g.loc[g["cabin_tier"] == "economy", "yield_per_mile"].mean()
            if not g.empty else np.nan
        ).reset_index(drop=True) if not df.empty else np.nan

    # Map report_quarter → report_period (first month of quarter)
    if "report_quarter" in pivot.columns:
        pivot["report_period"] = pd.to_datetime(pivot["report_quarter"])

    return pivot


# =============================================================================
# Cost estimation
# =============================================================================

def estimate_flight_costs(
    demand_df: pd.DataFrame,
    fuel_burn_df: pd.DataFrame,
    crew_cost_df: pd.DataFrame,
    fuel_price_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate fuel and crew costs per route-period-aircraft.

    Uses:
      - fuel_gph × block_hours × jet_a_price = fuel cost
      - crew_cost_per_bh × block_hours = crew cost
    """
    df = demand_df.copy()

    # Merge fuel burn rates
    if not fuel_burn_df.empty and "aircraft_variant" in df.columns:
        df = df.merge(fuel_burn_df[["aircraft_variant", "gph_cruise"]],
                      on="aircraft_variant", how="left")
    else:
        df["gph_cruise"] = np.nan

    # Merge crew costs
    if not crew_cost_df.empty and "aircraft_variant" in df.columns:
        df = df.merge(crew_cost_df[["aircraft_variant", "total_crew_cost_per_bh"]],
                      on="aircraft_variant", how="left")
    else:
        df["total_crew_cost_per_bh"] = np.nan

    # Match fuel price to report period (nearest weekly price, backward)
    if not fuel_price_df.empty and "report_period" in df.columns:
        fuel_sorted = fuel_price_df.sort_values("price_date")
        df["report_period"] = pd.to_datetime(df["report_period"])
        df = pd.merge_asof(
            df.sort_values("report_period"),
            fuel_sorted.rename(columns={"price_date": "report_period",
                                        "jet_a_usd_per_gal": "jet_a_price"}),
            on="report_period",
            direction="backward",
        )
    else:
        df["jet_a_price"] = 2.90  # fallback: approximate historical Jet-A price

    # Approximate block hours from average block_time_min (we don't have per-route
    # scheduled times in T-100; use distance as proxy: avg 450 knots cruise + 30 min taxi)
    if "distance_mi" in df.columns:
        # Cruise speed ≈ 450 knots = 518 mph; taxi allowance = 0.5h
        df["est_block_hours"] = (df["distance_mi"] / 518.0 + 0.5).round(2)
    else:
        df["est_block_hours"] = 1.5   # default

    # Cost calculations (per departure)
    df["est_fuel_cost_per_dep"] = (
        df["gph_cruise"].fillna(850) * df["est_block_hours"] * df["jet_a_price"].fillna(2.90)
    ).round(2)
    df["est_crew_cost_per_dep"] = (
        df["total_crew_cost_per_bh"].fillna(1200) * df["est_block_hours"]
    ).round(2)

    # Scale to total monthly departures
    df["est_fuel_cost_usd"] = (df["est_fuel_cost_per_dep"] * df.get("departures", 1)).round(2)
    df["est_crew_cost_usd"] = (df["est_crew_cost_per_dep"] * df.get("departures", 1)).round(2)
    df["est_op_cost_usd"]   = (df["est_fuel_cost_usd"] + df["est_crew_cost_usd"]).round(2)

    return df


# =============================================================================
# Revenue estimation
# =============================================================================

def estimate_revenue(
    demand_df: pd.DataFrame,
    fare_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate monthly route revenue = LF × seats × blended_fare × departures.

    Merges DB1B blended fares; falls back to FALLBACK_BLENDED_FARE where absent.
    """
    df = demand_df.copy()

    if not fare_df.empty and "origin_iata" in fare_df.columns:
        df["report_period"] = pd.to_datetime(df["report_period"])
        fare_df["report_period"] = pd.to_datetime(fare_df.get("report_period",
                                                               fare_df.get("report_quarter")))
        df = pd.merge_asof(
            df.sort_values("report_period"),
            fare_df[["origin_iata", "destination_iata", "report_period",
                     "avg_fare_blended_usd",
                     "avg_fare_economy_usd", "avg_fare_business_usd",
                     "avg_fare_first_usd"]].sort_values("report_period"),
            on="report_period",
            by=["origin_iata", "destination_iata"],
            direction="backward",
        )
    else:
        df["avg_fare_blended_usd"]  = FALLBACK_BLENDED_FARE
        df["avg_fare_economy_usd"]  = np.nan
        df["avg_fare_business_usd"] = np.nan
        df["avg_fare_first_usd"]    = np.nan

    df["avg_fare_blended_usd"] = df["avg_fare_blended_usd"].fillna(FALLBACK_BLENDED_FARE)

    lf_col = "load_factor_imputed" if "load_factor_imputed" in df.columns else "load_factor"
    df["est_revenue_usd"] = (
        df[lf_col].fillna(0.83)
        * df.get("seats_available", 150)
        * df.get("departures", 1)
        * df["avg_fare_blended_usd"]
    ).round(2)

    return df


# =============================================================================
# Assemble KPI feed
# =============================================================================

def assemble_kpi_feed(
    demand_df:   pd.DataFrame,
    perf_df:     pd.DataFrame,
    season_df:   pd.DataFrame,
    fare_df:     pd.DataFrame,
    fuel_burn_df: pd.DataFrame,
    crew_cost_df: pd.DataFrame,
    fuel_price_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join all inputs and compute final KPIs.
    Returns route_kpi_feed DataFrame.
    """
    if demand_df.empty:
        logger.warning("No demand data; cannot assemble KPI feed.")
        return pd.DataFrame()

    df = demand_df.copy()
    df["report_period"] = pd.to_datetime(df["report_period"])

    # ── Costs ─────────────────────────────────────────────────────────────────
    df = estimate_flight_costs(df, fuel_burn_df, crew_cost_df, fuel_price_df)

    # ── Revenue ───────────────────────────────────────────────────────────────
    blended_fares = build_blended_fares(fare_df) if not fare_df.empty else pd.DataFrame()
    df = estimate_revenue(df, blended_fares)

    # ── Operational perf join ─────────────────────────────────────────────────
    if not perf_df.empty and "origin_iata" in perf_df.columns:
        perf_df["report_period"] = pd.to_datetime(perf_df["report_period"])
        df = df.merge(
            perf_df[["report_period", "origin_iata", "destination_iata",
                     "aircraft_variant", "otp_pct", "avg_arrival_delay_min",
                     "avg_block_time_min"]],
            on=["report_period", "origin_iata", "destination_iata", "aircraft_variant"],
            how="left",
        )

    # ── Seasonality join ──────────────────────────────────────────────────────
    if not season_df.empty and "month_of_year" in season_df.columns:
        df["month_of_year"] = pd.to_datetime(df["report_period"]).dt.month
        df = df.merge(
            season_df[["carrier_code", "origin_iata", "destination_iata",
                       "month_of_year", "seasonality_index_lf"]].rename(
                columns={"seasonality_index_lf": "seasonality_index"}),
            on=["carrier_code", "origin_iata", "destination_iata", "month_of_year"],
            how="left",
        )

    # ── Profit per block hour ─────────────────────────────────────────────────
    if "avg_block_time_min" not in df.columns:
        df["avg_block_time_min"] = df.get("est_block_hours", 1.5) * 60
    df["block_hours"] = (df["avg_block_time_min"].fillna(df.get("est_block_hours", 1.5) * 60) / 60).round(2)

    df["op_profit_per_bh_usd"] = np.where(
        df["block_hours"] > 0,
        ((df["est_revenue_usd"].fillna(0) - df["est_op_cost_usd"].fillna(0))
         / df["block_hours"]).round(2),
        np.nan,
    )

    # ── Imputation flags ──────────────────────────────────────────────────────
    lf_col = "load_factor_imputed" if "load_factor_imputed" in df.columns else "load_factor"
    df["lf_imputed_flag"]  = df.get("lf_source", "T100") != "T100"
    df["lf_outlier_flag"]  = df.get("lf_outlier_flag", False).fillna(False)

    # ── Passengers estimate ───────────────────────────────────────────────────
    df["passengers_est"] = df.get("passengers", (df[lf_col] * df.get("seats_available", 150)).round(0))

    # ── Yield per mile (economy) ──────────────────────────────────────────────
    yield_col = "yield_per_mile_economy"
    if yield_col not in df.columns:
        df[yield_col] = np.where(
            df.get("distance_mi", 0) > 0,
            df.get("avg_fare_economy_usd", np.nan) / df.get("distance_mi", 1),
            np.nan,
        )

    # ── Select final columns ──────────────────────────────────────────────────
    keep = [
        "report_period", "carrier_code", "origin_iata", "destination_iata",
        "aircraft_variant",
        lf_col,
        "seats_available", "passengers_est", "asm", "rpm",
        "avg_fare_economy_usd", "avg_fare_business_usd", "avg_fare_first_usd",
        yield_col,
        "est_fuel_cost_usd", "est_crew_cost_usd", "est_op_cost_usd",
        "est_revenue_usd",
        "block_hours", "op_profit_per_bh_usd",
        "seasonality_index",
        "otp_pct", "avg_arrival_delay_min",
        "lf_imputed_flag", "lf_outlier_flag",
    ]
    available = [c for c in keep if c in df.columns]
    out = df[available].copy()

    # Rename load_factor_imputed → load_factor in output
    out = out.rename(columns={lf_col: "load_factor",
                               "lf_imputed_flag": "lf_imputed"})

    num_cols = out.select_dtypes(include="number").columns
    out[num_cols] = out[num_cols].round(4)

    logger.info("KPI feed assembled: %d rows, %d columns", len(out), len(out.columns))
    return out


# =============================================================================
# Summary and validation
# =============================================================================

def kpi_summary(kpi_df: pd.DataFrame) -> dict:
    """Return high-level KPI summary statistics."""
    if kpi_df.empty:
        return {}
    return {
        "n_routes":               kpi_df.groupby(["origin_iata", "destination_iata"]).ngroups,
        "n_aircraft_variants":    kpi_df["aircraft_variant"].nunique(),
        "n_periods":              kpi_df["report_period"].nunique(),
        "mean_load_factor":       round(float(kpi_df["load_factor"].mean()), 4),
        "mean_op_profit_per_bh":  round(float(kpi_df["op_profit_per_bh_usd"].mean()), 2),
        "mean_otp_pct":           round(float(kpi_df["otp_pct"].mean()), 4) if "otp_pct" in kpi_df else None,
        "pct_lf_imputed":         round(float(kpi_df["lf_imputed"].mean()), 4),
        "pct_lf_outlier":         round(float(kpi_df["lf_outlier_flag"].mean()), 4),
        "mean_est_revenue_usd":   round(float(kpi_df["est_revenue_usd"].mean()), 2),
        "mean_est_op_cost_usd":   round(float(kpi_df["est_op_cost_usd"].mean()), 2),
    }


# =============================================================================
# Entrypoint
# =============================================================================

def run(source: str = "db", save_csv: bool = True) -> pd.DataFrame:
    """
    Full KPI assembly pipeline.

    Loads all intermediate outputs (from load_factors, operational, demand)
    and DB tables, then assembles and saves route_kpi_feed.
    """
    # ── Load intermediate CSVs ────────────────────────────────────────────────
    demand_df  = load_demand_summary()
    perf_df    = load_operational_perf()
    season_df  = load_seasonality()

    # ── Load cost/fare tables ─────────────────────────────────────────────────
    if source == "db" and _DB_AVAILABLE:
        conn = get_connection()
        try:
            fare_df       = load_db1b_fares_from_db(conn)
            fuel_burn_df  = load_fuel_burn_from_db(conn)
            crew_cost_df  = load_crew_costs_from_db(conn)
            fuel_price_df = load_fuel_prices_from_db(conn)
        finally:
            conn.close()
    else:
        fare_df       = load_db1b_from_csv()
        fuel_burn_df  = pd.DataFrame()   # will use defaults
        crew_cost_df  = pd.DataFrame()
        fuel_price_df = pd.DataFrame()

    # ── Assemble ──────────────────────────────────────────────────────────────
    kpi_df = assemble_kpi_feed(
        demand_df, perf_df, season_df,
        fare_df, fuel_burn_df, crew_cost_df, fuel_price_df,
    )

    if kpi_df.empty:
        return kpi_df

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_csv:
        kpi_df.to_csv(_OUTPUT_DIR / "route_kpi_feed.csv", index=False)
        logger.info("Saved route_kpi_feed.csv → %s", _OUTPUT_DIR)

    # ── Upsert to DB ──────────────────────────────────────────────────────────
    if _DB_AVAILABLE:
        conn = get_connection()
        try:
            n = _upsert(
                conn, kpi_df, "route_kpi_feed",
                conflict_cols=["report_period", "carrier_code",
                               "origin_iata", "destination_iata", "aircraft_variant"],
                update_cols=[c for c in kpi_df.columns
                             if c not in ("report_period", "carrier_code",
                                          "origin_iata", "destination_iata",
                                          "aircraft_variant")],
            )
            logger.info("Upserted %d KPI feed rows.", n)
        finally:
            conn.close()

    summary = kpi_summary(kpi_df)
    logger.info("KPI summary: %s", summary)
    return kpi_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    kpi = run(source="csv")
    if not kpi.empty:
        print(f"\nKPI feed: {len(kpi)} rows")
        print(kpi.head(10).to_string(index=False))
        print("\n── KPI Summary ──")
        for k, v in kpi_summary(kpi).items():
            print(f"  {k}: {v}")
