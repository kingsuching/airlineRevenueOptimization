"""
pipeline.py  —  Project 3 orchestrator
Cost Structure & CASM Analysis

Run modes:
  --fuel           Compute fuel costs (GPH × block hours × Jet-A price)
  --labor          Compute crew labor costs (pilot + FA, from contract rates)
  --maintenance    Estimate aircraft DMC (Direct Maintenance Cost per block hour)
  --casm           Assemble full CASM and benchmark vs peer carriers
  --init-db        Apply Project 3 schema.sql to PostgreSQL
  --all            Run every step in sequence

Source options:
  --db             Pull from PostgreSQL (requires Project 1 & 2 DBs populated)
  --csv            Use CSV files in outputs/ and Project 2/outputs/ (default)

Usage:
    python pipeline.py --all --csv
    python pipeline.py --all --db
    python pipeline.py --casm --csv
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
sys.path.insert(0, str(_HERE))

import fuel_costs  as fuel_mod
import labor_costs as labor_mod
import maintenance as maint_mod
import casm        as casm_mod

try:
    from load import get_connection
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False

_OUTPUT_DIR = _HERE.parent / "outputs"
_LOG_DIR    = _HERE.parent / "logs"
_SQL_DIR    = _HERE.parent / "sql"
_LOG_DIR.mkdir(exist_ok=True)
_OUTPUT_DIR.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)


# =============================================================================
# DB initialisation
# =============================================================================

def init_database() -> None:
    if not _DB_AVAILABLE:
        logger.error("DB not available — install psycopg2 and configure DB_CONFIG.")
        return
    schema_path = _SQL_DIR / "schema.sql"
    logger.info("Applying Project 3 schema from %s …", schema_path)
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(schema_path.read_text())
    conn.commit()
    conn.close()
    logger.info("Project 3 schema applied.")


# =============================================================================
# Pipeline steps
# =============================================================================

def run_fuel(source: str) -> dict:
    logger.info("=== STEP 1: FUEL COSTS ===")
    started = datetime.now(timezone.utc)
    fuel_df = fuel_mod.run(source=source, save_csv=True)
    stats = {
        "rows":      len(fuel_df),
        "elapsed_s": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
    }
    if not fuel_df.empty:
        stats.update(fuel_mod.fuel_cost_summary(fuel_df))
    logger.info("Fuel step complete: %s", stats)
    return stats


def run_labor(source: str) -> dict:
    logger.info("=== STEP 2: LABOR COSTS ===")
    started = datetime.now(timezone.utc)
    labor_df = labor_mod.run(source=source, save_csv=True)
    stats = {
        "rows":      len(labor_df),
        "elapsed_s": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
    }
    if not labor_df.empty:
        stats.update(labor_mod.labor_cost_summary(labor_df))
    logger.info("Labor step complete: %s", stats)
    return stats


def run_maintenance(source: str) -> dict:
    logger.info("=== STEP 3: MAINTENANCE COSTS ===")
    started = datetime.now(timezone.utc)
    maint_df, rates_df = maint_mod.run(source=source, save_csv=True)
    stats = {
        "rows":            len(maint_df),
        "aircraft_types":  len(rates_df),
        "elapsed_s":       round((datetime.now(timezone.utc) - started).total_seconds(), 1),
    }
    if not maint_df.empty:
        stats.update(maint_mod.maintenance_cost_summary(maint_df))
    logger.info("Maintenance step complete: %s", stats)
    return stats


def run_casm(source: str) -> dict:
    logger.info("=== STEP 4: CASM ASSEMBLY ===")
    started = datetime.now(timezone.utc)
    results = casm_mod.run(source=source, save_csv=True)
    cost_df = results["cost_summary"]
    stats = {
        "cost_summary_rows": len(cost_df),
        "benchmark_rows":    len(results["benchmarks"]),
        "casm_feed_rows":    len(results["casm_feed"]),
        "elapsed_s":         round((datetime.now(timezone.utc) - started).total_seconds(), 1),
    }
    if not cost_df.empty:
        stats.update(casm_mod.casm_summary(cost_df))
    logger.info("CASM step complete: %s", stats)
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
        description="Project 3 — Cost Structure & CASM Analysis Pipeline"
    )
    parser.add_argument("--fuel",        action="store_true", help="Compute fuel costs")
    parser.add_argument("--labor",       action="store_true", help="Compute crew labor costs")
    parser.add_argument("--maintenance", action="store_true", help="Estimate aircraft DMC")
    parser.add_argument("--casm",        action="store_true", help="Assemble full CASM and peer benchmarks")
    parser.add_argument("--init-db",     action="store_true", help="Apply Project 3 schema.sql")
    parser.add_argument("--all",         action="store_true", help="Run all steps in sequence")
    parser.add_argument("--db",          action="store_true", help="Pull from PostgreSQL")
    parser.add_argument("--csv",         action="store_true", help="Use CSV files (default)")
    args = parser.parse_args()

    source = "db" if args.db else "csv"

    if args.init_db:
        init_database()
        return

    if not any(vars(args).values()):
        parser.print_help()
        return

    all_stats: dict[str, dict] = {}
    started_total = datetime.now(timezone.utc)

    if args.all or args.fuel:
        all_stats["fuel"] = run_fuel(source)

    if args.all or args.labor:
        all_stats["labor"] = run_labor(source)

    if args.all or args.maintenance:
        all_stats["maintenance"] = run_maintenance(source)

    if args.all or args.casm:
        all_stats["casm"] = run_casm(source)

    elapsed = round((datetime.now(timezone.utc) - started_total).total_seconds(), 1)
    logger.info("=== PROJECT 3 COMPLETE (%.1fs) ===", elapsed)
    for step, stats in all_stats.items():
        logger.info("  [%s] %s", step, stats)

    print(f"\nProject 3 pipeline complete in {elapsed}s.")
    print(f"Outputs written to: {_OUTPUT_DIR}")
    for step, stats in all_stats.items():
        print(f"  {step}: {stats}")


if __name__ == "__main__":
    main()
