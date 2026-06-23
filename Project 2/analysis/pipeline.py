"""
pipeline.py  —  Project 2 orchestrator
Route Traffic, Demand & Operational Performance Analysis

Run modes:
  --load-factors   Compute LF, impute missing, flag outliers
  --operational    Aggregate OTP / delay / block-time stats from FlightAware
  --demand         Seasonality indices and YoY growth rates
  --kpis           Assemble route_kpi_feed (all inputs required)
  --init-db        Apply Project 2 schema.sql
  --all            Run every step in sequence

Source options:
  --db             Pull from PostgreSQL (requires Project 1 DB to be populated)
  --csv            Use CSV files in outputs/ and Project 1/data/ (default)

Usage:
    python pipeline.py --all --csv
    python pipeline.py --all --db
    python pipeline.py --kpis --db
    python pipeline.py --init-db
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT1_ETL = _HERE.parents[1] / "Project 1" / "etl"
if str(_PROJECT1_ETL) not in sys.path:
    sys.path.insert(0, str(_PROJECT1_ETL))

# Project 2 analysis modules
sys.path.insert(0, str(_HERE))
import load_factors as lf_mod
import operational  as op_mod
import demand       as dem_mod
import kpis         as kpi_mod

try:
    from load import get_connection
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

_OUTPUT_DIR = _HERE.parent / "outputs"
_LOG_DIR    = _HERE.parent / "logs"
_SQL_DIR    = _HERE.parent / "sql"
_LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)


# =============================================================================
# DB initialisation
# =============================================================================

def init_database() -> None:
    """Apply Project 2 schema.sql to PostgreSQL."""
    if not _DB_AVAILABLE:
        logger.error("DB not available — install psycopg2 and configure DB_CONFIG.")
        return
    schema_path = _SQL_DIR / "schema.sql"
    logger.info("Applying Project 2 schema from %s …", schema_path)
    ddl = schema_path.read_text()
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    conn.close()
    logger.info("Project 2 schema applied.")


# =============================================================================
# Pipeline steps
# =============================================================================

def run_load_factors(source: str) -> dict:
    logger.info("=== STEP 1: LOAD FACTORS ===")
    started = datetime.now(timezone.utc)
    demand_df, lf_df = lf_mod.run(source=source, save_csv=True)
    stats = {
        "demand_rows":      len(demand_df),
        "lf_imputed_rows":  len(lf_df),
        "elapsed_s":        round((datetime.now(timezone.utc) - started).total_seconds(), 1),
    }
    if not demand_df.empty:
        stats.update(lf_mod.summarise_load_factors(demand_df))
    logger.info("Load factor step complete: %s", stats)
    return stats


def run_operational(source: str) -> dict:
    logger.info("=== STEP 2: OPERATIONAL PERFORMANCE ===")
    started = datetime.now(timezone.utc)
    perf_df = op_mod.run(source=source, save_csv=True)
    stats = {
        "perf_rows":   len(perf_df),
        "elapsed_s":   round((datetime.now(timezone.utc) - started).total_seconds(), 1),
    }
    if not perf_df.empty:
        otp_summary = op_mod.aircraft_otp_comparison(perf_df)
        stats["aircraft_types"] = len(otp_summary)
        stats["mean_otp_pct"]   = round(float(otp_summary["avg_otp_pct"].mean()), 4) \
                                   if not otp_summary.empty else None
    logger.info("Operational step complete: %s", stats)
    return stats


def run_demand(source: str) -> dict:
    logger.info("=== STEP 3: DEMAND & SEASONALITY ===")
    started = datetime.now(timezone.utc)
    season_df, trends_df = dem_mod.run(source=source, save_csv=True)
    stats = {
        "seasonality_rows": len(season_df),
        "trend_rows":       len(trends_df),
        "elapsed_s":        round((datetime.now(timezone.utc) - started).total_seconds(), 1),
    }
    logger.info("Demand step complete: %s", stats)
    return stats


def run_kpis(source: str) -> dict:
    logger.info("=== STEP 4: KPI ASSEMBLY ===")
    started = datetime.now(timezone.utc)
    kpi_df = kpi_mod.run(source=source, save_csv=True)
    stats  = {
        "kpi_rows":  len(kpi_df),
        "elapsed_s": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
    }
    if not kpi_df.empty:
        stats.update(kpi_mod.kpi_summary(kpi_df))
    logger.info("KPI step complete: %s", stats)
    return stats


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    log_path = _LOG_DIR / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )

    parser = argparse.ArgumentParser(
        description="Project 2 — Route Traffic, Demand & Operational Performance Pipeline"
    )
    parser.add_argument("--load-factors",  action="store_true",
                        help="Compute and impute load factors from BTS T-100")
    parser.add_argument("--operational",   action="store_true",
                        help="Aggregate OTP / delay stats from FlightAware flights")
    parser.add_argument("--demand",        action="store_true",
                        help="Compute seasonality indices and YoY growth rates")
    parser.add_argument("--kpis",          action="store_true",
                        help="Assemble route_kpi_feed for Project 4")
    parser.add_argument("--init-db",       action="store_true",
                        help="Apply Project 2 schema.sql to PostgreSQL")
    parser.add_argument("--all",           action="store_true",
                        help="Run all steps in sequence")
    parser.add_argument("--db",            action="store_true",
                        help="Source data from PostgreSQL (default: CSV)")
    parser.add_argument("--csv",           action="store_true",
                        help="Source data from CSV files (default)")
    args = parser.parse_args()

    source = "db" if args.db else "csv"

    if args.init_db:
        init_database()
        return

    all_stats: dict[str, dict] = {}
    started_total = datetime.now(timezone.utc)

    if args.all or args.load_factors:
        all_stats["load_factors"] = run_load_factors(source)

    if args.all or args.operational:
        all_stats["operational"] = run_operational(source)

    if args.all or args.demand:
        all_stats["demand"] = run_demand(source)

    if args.all or args.kpis:
        # KPIs depend on all prior steps having written their CSVs
        all_stats["kpis"] = run_kpis(source)

    if not any(vars(args).values()):
        parser.print_help()
        return

    elapsed = round((datetime.now(timezone.utc) - started_total).total_seconds(), 1)
    logger.info("=== PROJECT 2 COMPLETE (%.1fs) ===", elapsed)
    for step, stats in all_stats.items():
        logger.info("  [%s] %s", step, stats)

    print(f"\nProject 2 pipeline complete in {elapsed}s.")
    print(f"Outputs written to: {_OUTPUT_DIR}")
    for step, stats in all_stats.items():
        print(f"  {step}: {stats}")


if __name__ == "__main__":
    main()
