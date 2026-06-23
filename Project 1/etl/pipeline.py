"""
pipeline.py
Main orchestrator for the Project 1 daily ETL pipeline.

Run modes:
  --live        Pull fresh FlightAware data via API (requires FA_API_KEY)
  --from-csv    Use saved CSVs in data/ (default, for dev / replay)
  --fuel-only   Refresh only EIA fuel prices
  --bts         Load BTS T-100 / DB1B files found in data/
  --form41      Load BTS Form 41 P-12(a) from data/ (CSV already present)
  --form41-dl   Download latest Form 41 ZIP then load
  --labor       Load AFA-CWA / ALPA crew cost library + SEC EDGAR 10-K
  --init-db     Apply schema.sql to create tables (first-time setup)
  --all         Run every pipeline end-to-end (live FA data)

Usage:
    python pipeline.py --init-db           # first-time DB setup
    python pipeline.py --from-csv          # dev / CI
    python pipeline.py --live              # daily cron job
    python pipeline.py --bts               # after dropping BTS CSV files in data/
    python pipeline.py --form41-dl         # download + validate Form 41
    python pipeline.py --labor             # refresh crew cost library
    python pipeline.py --all               # full end-to-end run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Local modules (adjust sys.path so config is importable when run directly) ─
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import SQL_DIR, DATA_DIR, AIRCRAFT_TYPE_MAP
from extract_flightaware import fetch_all_operators, load_flights_from_csv
from extract_eia_fuel import fetch_jet_fuel_prices
from extract_bts import auto_load_bts
from extract_bts_form41 import load_form41, aggregate_form41, validate_against_poh
from extract_sec_edgar import fetch_and_save as fetch_10k_labor
from extract_labor_rates import build_crew_cost_library, load_bts_p6_validation
from transform import (
    normalise_flights,
    flatten_aircraft_data,
    compute_flight_capacity,
    impute_load_factors,
    flag_lf_outliers,
    build_dim_time,
)
from load import (
    get_connection,
    load_dim_time,
    load_aircraft_configs,
    load_flights,
    load_flight_capacity,
    load_fuel_prices,
    load_aircraft_fuel_burn,
    load_t100,
    load_db1b,
    _upsert,
    log_etl_run,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Init: create tables from schema.sql
# ─────────────────────────────────────────────────────────────────────────────

def init_database() -> None:
    schema_path = SQL_DIR / "schema.sql"
    logger.info("Applying schema from %s …", schema_path)
    ddl = schema_path.read_text()
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    conn.close()
    logger.info("Schema applied.")


# ─────────────────────────────────────────────────────────────────────────────
# Load aircraft seat-map into DB (static, run once or on update)
# ─────────────────────────────────────────────────────────────────────────────

def run_aircraft_configs() -> None:
    json_path = Path(__file__).resolve().parents[1] / "aircraft_data" / "UAL_aircraft_data.json"
    with json_path.open() as f:
        raw = json.load(f)
    aircraft_df = flatten_aircraft_data(raw)
    conn = get_connection()
    try:
        load_aircraft_configs(conn, aircraft_df)
        load_aircraft_fuel_burn(conn)
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main ETL: flights → capacity → dim_time
# ─────────────────────────────────────────────────────────────────────────────

def run_flights_pipeline(live: bool = False) -> dict:
    started = datetime.now(timezone.utc)
    stats: dict[str, int] = {}

    # ── 1. Extract ────────────────────────────────────────────────────────────
    logger.info("=== EXTRACT ===")
    if live:
        logger.info("Pulling live data from FlightAware …")
        operator_dfs = fetch_all_operators(save_csv=True)
    else:
        logger.info("Loading from CSV snapshots …")
        operator_dfs = load_flights_from_csv()

    if not operator_dfs:
        logger.error("No flight data extracted. Aborting.")
        return stats

    # ── 2. Transform ──────────────────────────────────────────────────────────
    logger.info("=== TRANSFORM ===")
    json_path = Path(__file__).resolve().parents[1] / "aircraft_data" / "UAL_aircraft_data.json"
    with json_path.open() as f:
        raw_aircraft = json.load(f)
    aircraft_df = flatten_aircraft_data(raw_aircraft)

    all_flights:  list[pd.DataFrame] = []
    all_capacity: list[pd.DataFrame] = []

    for op, df in operator_dfs.items():
        norm = normalise_flights(df)
        all_flights.append(norm)

        cap, unmapped = compute_flight_capacity(norm, aircraft_df)
        if unmapped:
            logger.warning("[%s] Unmapped aircraft codes: %s", op, sorted(unmapped))
        all_capacity.append(cap)

    flights_df  = pd.concat(all_flights,  ignore_index=True).drop_duplicates("flight_id")
    capacity_df = pd.concat(all_capacity, ignore_index=True).drop_duplicates(
        subset=["flight_id", "cabin_class"]
    )

    # dim_time
    dim_time_df = build_dim_time(flights_df.get("flight_date", pd.Series(dtype="object")))

    stats["flights_extracted"]  = len(flights_df)
    stats["capacity_rows"]      = len(capacity_df)

    # ── 3. Load ───────────────────────────────────────────────────────────────
    logger.info("=== LOAD ===")
    conn = get_connection()
    try:
        stats["dim_time_loaded"]   = load_dim_time(conn, dim_time_df)
        stats["flights_loaded"]    = load_flights(conn, flights_df)
        stats["capacity_loaded"]   = load_flight_capacity(conn, capacity_df)
        log_etl_run(conn, "flights_pipeline", "flightaware", "success",
                    rows_extracted=stats["flights_extracted"],
                    rows_loaded=stats["flights_loaded"],
                    started_at=started)
    except Exception as exc:
        logger.exception("Load failed: %s", exc)
        log_etl_run(conn, "flights_pipeline", "flightaware", "failed",
                    error_message=str(exc), started_at=started)
        raise
    finally:
        conn.close()

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Fuel price pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_fuel_pipeline(start_date: str = "2020-01-01") -> int:
    started = datetime.now(timezone.utc)
    logger.info("=== FUEL PRICES ===")
    fuel_df = fetch_jet_fuel_prices(start_date=start_date, save_csv=True)
    if fuel_df.empty:
        logger.warning("No fuel prices fetched.")
        return 0
    conn = get_connection()
    try:
        n = load_fuel_prices(conn, fuel_df)
        log_etl_run(conn, "fuel_pipeline", "eia", "success",
                    rows_extracted=len(fuel_df), rows_loaded=n,
                    started_at=started)
    finally:
        conn.close()
    return n


# ─────────────────────────────────────────────────────────────────────────────
# BTS pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_bts_pipeline() -> dict:
    started = datetime.now(timezone.utc)
    logger.info("=== BTS T-100 / DB1B ===")
    bts_dfs = auto_load_bts()
    stats: dict[str, int] = {}

    conn = get_connection()
    try:
        if "t100" in bts_dfs:
            stats["t100_loaded"] = load_t100(conn, bts_dfs["t100"])
        if "db1b" in bts_dfs:
            stats["db1b_loaded"] = load_db1b(conn, bts_dfs["db1b"])
        log_etl_run(conn, "bts_pipeline", "bts", "success",
                    rows_loaded=sum(stats.values()), started_at=started)
    finally:
        conn.close()
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Form 41 fuel/labor pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_form41_pipeline(download: bool = False) -> dict:
    started = datetime.now(timezone.utc)
    logger.info("=== BTS FORM 41 P-12(a) ===")
    stats: dict = {}

    if download:
        from extract_bts_form41 import download_form41
        zip_path = download_form41()
        raw_df = load_form41(zip_path)
    else:
        raw_df = load_form41()

    if raw_df.empty:
        logger.warning("No Form 41 data — skipping.")
        return stats

    agg_df = aggregate_form41(raw_df)
    stats["form41_rows"] = len(agg_df)

    # POH cross-validation
    val_df = validate_against_poh(raw_df)
    stats["poh_flags"] = int(val_df["flag"].sum()) if not val_df.empty else 0

    conn = get_connection()
    try:
        # Load aggregated Form 41
        n = _upsert(conn, agg_df, "form41_fuel_labor",
                    conflict_cols=["report_quarter", "carrier_code", "aircraft_type_bts"],
                    update_cols=["fuel_gallons", "fuel_cost_usd", "block_hours",
                                 "gph_block_hour", "fuel_cost_per_gal",
                                 "salary_pilots_usd", "salary_fa_usd"])
        stats["form41_loaded"] = n

        # Load POH validation results
        if not val_df.empty:
            val_keep = ["aircraft_variant", "engine", "poh_cruise_gph",
                        "poh_cruise_gph_low", "poh_cruise_gph_high",
                        "bts_block_hour_gph_poh", "bts_weighted_gph",
                        "divergence_pct", "flag", "poh_source"]
            val_load = val_df[[c for c in val_keep if c in val_df.columns]].rename(
                columns={"bts_weighted_gph": "bts_actual_gph", "flag": "flagged"})
            _upsert(conn, val_load, "poh_fuel_validation",
                    conflict_cols=["aircraft_variant"],
                    update_cols=["bts_actual_gph", "divergence_pct", "flagged"])

        log_etl_run(conn, "form41_pipeline", "bts_form41", "success",
                    rows_extracted=len(raw_df), rows_loaded=n, started_at=started)
    finally:
        conn.close()

    if stats.get("poh_flags", 0) > 0:
        logger.warning("⚠  %d aircraft variant(s) flagged: POH vs BTS divergence >10%%.",
                       stats["poh_flags"])
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Labor rates pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_labor_pipeline() -> dict:
    started = datetime.now(timezone.utc)
    logger.info("=== LABOR RATES (AFA-CWA / ALPA / SEC EDGAR) ===")
    stats: dict = {}

    # 1. Crew cost library from contract JSON
    crew_lib = build_crew_cost_library()
    stats["crew_variants"] = len(crew_lib)

    # 2. United 10-K labor from SEC EDGAR
    labor_10k = fetch_10k_labor(years=10, save_csv=True)
    stats["10k_years"] = len(labor_10k)

    conn = get_connection()
    try:
        # Load crew cost library
        n = _upsert(conn, crew_lib, "labor_rates_ref",
                    conflict_cols=["aircraft_variant"],
                    update_cols=["pilot_cost_per_bh", "fa_cost_per_bh",
                                 "total_crew_cost_per_bh", "fa_count"])
        stats["labor_rates_loaded"] = n

        # Load 10-K data
        if not labor_10k.empty:
            n2 = _upsert(conn, labor_10k, "ual_10k_labor",
                         conflict_cols=["fiscal_year"],
                         update_cols=[c for c in labor_10k.columns if c != "fiscal_year"])
            stats["10k_loaded"] = n2

        log_etl_run(conn, "labor_pipeline", "afa_alpa_edgar", "success",
                    rows_loaded=stats.get("labor_rates_loaded", 0), started_at=started)
    finally:
        conn.close()

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                Path(__file__).resolve().parents[1] / "logs" / "pipeline.log"
            ),
        ],
    )

    parser = argparse.ArgumentParser(description="Project 1 ETL pipeline")
    parser.add_argument("--live",        action="store_true", help="Pull live FlightAware data")
    parser.add_argument("--from-csv",    action="store_true", help="Use CSV snapshots (default)")
    parser.add_argument("--fuel-only",   action="store_true", help="Refresh EIA fuel prices only")
    parser.add_argument("--bts",         action="store_true", help="Load BTS T-100/DB1B from data/")
    parser.add_argument("--form41",      action="store_true", help="Load BTS Form 41 P-12(a)")
    parser.add_argument("--form41-dl",   action="store_true", help="Download + load BTS Form 41")
    parser.add_argument("--labor",       action="store_true", help="Load AFA-CWA/ALPA rates + 10-K")
    parser.add_argument("--init-db",     action="store_true", help="Apply schema.sql (first-time setup)")
    parser.add_argument("--all",         action="store_true", help="Run all pipelines (live)")
    args = parser.parse_args()

    if args.init_db:
        init_database()
        run_aircraft_configs()
        return

    if args.fuel_only:
        n = run_fuel_pipeline()
        print(f"Loaded {n} fuel price rows.")
        return

    if args.bts:
        stats = run_bts_pipeline()
        print(f"BTS T-100/DB1B stats: {stats}")
        return

    if args.form41 or args.form41_dl:
        stats = run_form41_pipeline(download=args.form41_dl)
        print(f"Form 41 stats: {stats}")
        return

    if args.labor:
        stats = run_labor_pipeline()
        print(f"Labor pipeline stats: {stats}")
        return

    if args.all:
        run_aircraft_configs()
        stats = run_flights_pipeline(live=True)
        run_fuel_pipeline()
        run_bts_pipeline()
        run_form41_pipeline()
        run_labor_pipeline()
        print(f"All pipelines complete: {stats}")
        return

    # Default: from-csv (dev mode)
    stats = run_flights_pipeline(live=args.live)
    print(f"Pipeline complete: {stats}")


if __name__ == "__main__":
    main()
