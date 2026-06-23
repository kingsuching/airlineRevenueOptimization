"""
operational.py
Route-level operational performance analysis.

Computes OTP, block time variance, delay statistics, and cancellation rates
by (route, aircraft_variant, month) from FlightAware flight data.

Key metrics:
  - OTP (On-Time Performance): % flights with arrival_delay ≤ 14 min (DOT A14)
  - Average / median / P75 / P95 arrival delay
  - Block time mean, median, std dev vs scheduled
  - Block time padding: scheduled - actual (positive = buffer built in)
  - Cancellation rate
  - Schedule reliability index (composite OTP × (1 - cancel_rate))

Inputs (from Project 1 warehouse):
  - flights table  (FlightAware)

Outputs:
  - route_operational_perf table (upserted)
  - CSV → outputs/route_operational_perf.csv
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

# ── DOT on-time threshold ────────────────────────────────────────────────────
OTP_THRESHOLD_MIN = 14          # A14 standard: ≤ 14 min late = on-time
MIN_FLIGHTS_FOR_STATS = 5       # minimum flights per route-month to compute stats


# =============================================================================
# Load raw flight data
# =============================================================================

def load_flights_from_db(conn) -> pd.DataFrame:
    """Pull UAL flights from Project 1 warehouse."""
    sql = """
        SELECT
            flight_id,
            ident,
            operator_icao,
            aircraft_type,
            aircraft_variant,
            origin_iata,
            destination_iata,
            route_distance_mi,
            flight_date,
            scheduled_off,
            scheduled_on,
            actual_off,
            actual_on,
            block_time_min,
            departure_delay_min,
            arrival_delay_min,
            status,
            cancelled,
            diverted
        FROM flights
        WHERE operator_icao = 'UAL'
        ORDER BY flight_date, origin_iata, destination_iata
    """
    return pd.read_sql(sql, conn, parse_dates=["flight_date", "scheduled_off",
                                                "scheduled_on", "actual_off", "actual_on"])


def load_flights_from_csv(path: Optional[Path] = None) -> pd.DataFrame:
    """Fallback: load flights from Project 1 CSV snapshot."""
    if path is None:
        candidates = [
            _HERE.parents[1] / "Project 1" / "data" / "UAL_flights.csv",
            _HERE.parent / "outputs" / "UAL_flights.csv",
        ]
        for c in candidates:
            if c.exists():
                path = c
                break
    if path is None or not path.exists():
        logger.warning("No flights CSV found.")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["flight_date"])
    logger.info("Loaded flights from %s: %d rows", path, len(df))
    return df


# =============================================================================
# Feature engineering
# =============================================================================

def compute_block_times(flights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute scheduled and actual block times in minutes.
    block_time_min is already in the flights table (GENERATED ALWAYS AS);
    here we also compute scheduled_block_time_min from timestamps.
    """
    df = flights_df.copy()

    # Scheduled block time (ramp-to-ramp from scheduled timestamps)
    if "scheduled_off" in df.columns and "scheduled_on" in df.columns:
        df["scheduled_block_time_min"] = (
            (pd.to_datetime(df["scheduled_on"]) - pd.to_datetime(df["scheduled_off"]))
            .dt.total_seconds() / 60
        ).clip(lower=0)
    else:
        df["scheduled_block_time_min"] = np.nan

    # Actual block time from actual timestamps (or fall back to stored column)
    if "actual_off" in df.columns and "actual_on" in df.columns:
        actual_bt = (
            (pd.to_datetime(df["actual_on"]) - pd.to_datetime(df["actual_off"]))
            .dt.total_seconds() / 60
        ).clip(lower=0)
        df["actual_block_time_min"] = actual_bt.where(actual_bt > 0, df.get("block_time_min"))
    else:
        df["actual_block_time_min"] = df.get("block_time_min")

    # Padding = scheduled − actual (positive = airline built in buffer)
    df["block_time_padding_min"] = (
        df["scheduled_block_time_min"] - df["actual_block_time_min"]
    )

    # Infer arrival delay from block times if not stored
    if "arrival_delay_min" not in df.columns or df["arrival_delay_min"].isna().all():
        df["arrival_delay_min"] = -df["block_time_padding_min"]   # rough proxy

    # On-time flag
    df["is_on_time"] = (
        df["arrival_delay_min"].fillna(0) <= OTP_THRESHOLD_MIN
    ) & (~df["cancelled"].fillna(False))

    return df


# =============================================================================
# Aggregate to route × aircraft × month
# =============================================================================

def aggregate_operational_perf(flights_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate flight-level operational data to
    (report_period, operator_icao, origin, destination, aircraft_variant).

    report_period = first day of the flight month.
    """
    df = compute_block_times(flights_df)

    if "flight_date" in df.columns:
        df["flight_date"] = pd.to_datetime(df["flight_date"])
        df["report_period"] = df["flight_date"].dt.to_period("M").dt.to_timestamp()
    else:
        logger.error("No flight_date column; cannot aggregate.")
        return pd.DataFrame()

    # Fill missing aircraft_variant from aircraft_type
    if "aircraft_variant" not in df.columns:
        df["aircraft_variant"] = df.get("aircraft_type", "Unknown")
    df["aircraft_variant"] = df["aircraft_variant"].fillna(df.get("aircraft_type", "Unknown"))

    group_cols = [
        "report_period", "operator_icao",
        "origin_iata", "destination_iata", "aircraft_variant",
    ]

    def percentile(n):
        def _p(x): return float(np.nanpercentile(x, n)) if len(x) > 0 else np.nan
        _p.__name__ = f"p{n}"
        return _p

    agg = df.groupby(group_cols).agg(
        total_flights               =("flight_id",              "count"),
        cancelled_flights           =("cancelled",              "sum"),
        diverted_flights            =("diverted",               "sum"),
        avg_block_time_min          =("actual_block_time_min",  "mean"),
        median_block_time_min       =("actual_block_time_min",  "median"),
        stddev_block_time_min       =("actual_block_time_min",  "std"),
        scheduled_block_time_min    =("scheduled_block_time_min","mean"),
        block_time_padding_min      =("block_time_padding_min", "mean"),
        otp_pct                     =("is_on_time",             "mean"),
        avg_arrival_delay_min       =("arrival_delay_min",      "mean"),
        avg_departure_delay_min     =("departure_delay_min",    "mean"),
        p75_arrival_delay_min       =("arrival_delay_min",      percentile(75)),
        p95_arrival_delay_min       =("arrival_delay_min",      percentile(95)),
    ).reset_index()

    # Round numeric columns
    num_cols = agg.select_dtypes(include="number").columns
    agg[num_cols] = agg[num_cols].round(2)

    # Exclude routes with fewer than MIN_FLIGHTS_FOR_STATS (unreliable stats)
    low_volume = agg["total_flights"] < MIN_FLIGHTS_FOR_STATS
    if low_volume.sum() > 0:
        logger.info(
            "Excluding %d route-month combos with fewer than %d flights.",
            low_volume.sum(), MIN_FLIGHTS_FOR_STATS,
        )
        agg = agg[~low_volume].copy()

    logger.info("Operational perf: %d route-month rows aggregated.", len(agg))
    return agg


# =============================================================================
# Aircraft type comparisons
# =============================================================================

def aircraft_otp_comparison(perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise OTP, average delay, and cancellation rate by aircraft variant.
    Useful for the dashboard and Project 4 feature engineering.
    """
    if perf_df.empty:
        return pd.DataFrame()

    perf_df = perf_df.copy()
    perf_df["cancellation_rate"] = (
        perf_df["cancelled_flights"] / perf_df["total_flights"].replace(0, np.nan)
    )

    summary = (
        perf_df.groupby("aircraft_variant")
        .agg(
            total_routes        =("origin_iata",           "count"),
            total_flights       =("total_flights",          "sum"),
            avg_otp_pct         =("otp_pct",                "mean"),
            avg_arrival_delay   =("avg_arrival_delay_min",  "mean"),
            avg_p95_delay       =("p95_arrival_delay_min",  "mean"),
            avg_cancel_rate     =("cancellation_rate",      "mean"),
            avg_block_padding   =("block_time_padding_min", "mean"),
        )
        .round(4)
        .reset_index()
        .sort_values("avg_otp_pct", ascending=False)
    )
    return summary


def route_reliability_index(perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite reliability index per route:
        reliability = OTP × (1 − cancellation_rate)

    Values near 1.0 = reliable; near 0.0 = unreliable.
    """
    df = perf_df.copy()
    df["cancellation_rate"] = (
        df["cancelled_flights"] / df["total_flights"].replace(0, np.nan)
    ).fillna(0)
    df["reliability_index"] = df["otp_pct"].fillna(0) * (1 - df["cancellation_rate"])

    out = (
        df.groupby(["origin_iata", "destination_iata", "aircraft_variant"])
        .agg(
            avg_reliability     =("reliability_index",     "mean"),
            avg_otp             =("otp_pct",               "mean"),
            avg_cancel_rate     =("cancellation_rate",     "mean"),
            n_months            =("report_period",         "count"),
        )
        .round(4)
        .reset_index()
        .sort_values("avg_reliability", ascending=False)
    )
    return out


# =============================================================================
# Entrypoint
# =============================================================================

def run(
    source: str = "db",
    csv_path: Optional[Path] = None,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Full operational performance pipeline.

    Returns
    -------
    route_perf_df : aggregated operational performance DataFrame
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    if source == "db" and _DB_AVAILABLE:
        conn = get_connection()
        try:
            flights_df = load_flights_from_db(conn)
        finally:
            conn.close()
    else:
        flights_df = load_flights_from_csv(csv_path)

    if flights_df.empty:
        logger.warning("No flight data available.")
        return pd.DataFrame()

    # ── Aggregate ─────────────────────────────────────────────────────────────
    perf_df = aggregate_operational_perf(flights_df)

    if perf_df.empty:
        return pd.DataFrame()

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_csv:
        perf_df.to_csv(_OUTPUT_DIR / "route_operational_perf.csv", index=False)
        aircraft_otp_comparison(perf_df).to_csv(
            _OUTPUT_DIR / "aircraft_otp_summary.csv", index=False
        )
        route_reliability_index(perf_df).to_csv(
            _OUTPUT_DIR / "route_reliability_index.csv", index=False
        )
        logger.info("Saved operational perf CSVs → %s", _OUTPUT_DIR)

    # ── Upsert to DB ──────────────────────────────────────────────────────────
    if _DB_AVAILABLE:
        conn = get_connection()
        try:
            n = _upsert(
                conn, perf_df, "route_operational_perf",
                conflict_cols=["report_period", "operator_icao",
                               "origin_iata", "destination_iata", "aircraft_variant"],
                update_cols=[
                    "total_flights", "cancelled_flights", "diverted_flights",
                    "avg_block_time_min", "median_block_time_min", "stddev_block_time_min",
                    "scheduled_block_time_min", "block_time_padding_min",
                    "otp_pct", "avg_arrival_delay_min", "avg_departure_delay_min",
                    "p75_arrival_delay_min", "p95_arrival_delay_min",
                ],
            )
            logger.info("Upserted %d operational perf rows.", n)
        finally:
            conn.close()

    return perf_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    perf = run(source="csv")
    if not perf.empty:
        print(f"\nRoute operational perf: {len(perf)} rows")
        print(perf.head(10).to_string(index=False))
        print("\n── Aircraft OTP Comparison ──")
        print(aircraft_otp_comparison(perf).to_string(index=False))
