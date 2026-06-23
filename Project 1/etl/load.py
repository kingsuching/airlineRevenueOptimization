"""
load.py
Write transformed DataFrames into the PostgreSQL data warehouse.

Uses psycopg2 for direct COPY-based bulk inserts (fast) with upsert
semantics on conflict (idempotent re-runs).

Prerequisites:
    pip install psycopg2-binary pandas sqlalchemy
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone

import pandas as pd
import psycopg2
import psycopg2.extras
from psycopg2 import sql

from config import DB_CONFIG

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────────────────────────────────────

def get_connection() -> psycopg2.extensions.connection:
    """Return a new psycopg2 connection using DB_CONFIG."""
    return psycopg2.connect(**DB_CONFIG)


# ─────────────────────────────────────────────────────────────────────────────
# Generic upsert  (COPY → temp table → INSERT ON CONFLICT DO UPDATE)
# ─────────────────────────────────────────────────────────────────────────────

def _upsert(
    conn,
    df: pd.DataFrame,
    table: str,
    conflict_cols: list[str],
    update_cols: list[str] | None = None,
) -> int:
    """
    Bulk-upsert `df` into `table`.

    Uses a temp table + COPY for speed, then merges with ON CONFLICT.
    Returns the number of rows inserted/updated.
    """
    if df.empty:
        return 0

    cols = list(df.columns)
    tmp  = f"_tmp_{table}"
    update_cols = update_cols or [c for c in cols if c not in conflict_cols]

    col_ids      = [sql.Identifier(c) for c in cols]
    conflict_ids = [sql.Identifier(c) for c in conflict_cols]
    update_set   = sql.SQL(", ").join(
        sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(c))
        for c in update_cols
    )

    with conn.cursor() as cur:
        # Create temp table mirroring target
        cur.execute(
            sql.SQL("CREATE TEMP TABLE {tmp} (LIKE {tbl} INCLUDING DEFAULTS) ON COMMIT DROP")
            .format(tmp=sql.Identifier(tmp), tbl=sql.Identifier(table))
        )

        # COPY into temp table
        buf = io.StringIO()
        df.to_csv(buf, index=False, header=False, na_rep="\\N")
        buf.seek(0)
        cur.copy_expert(
            sql.SQL("COPY {tmp} ({cols}) FROM STDIN WITH (FORMAT CSV, NULL '\\N')")
            .format(tmp=sql.Identifier(tmp),
                    cols=sql.SQL(", ").join(col_ids)),
            buf
        )

        # Upsert from temp into target
        if update_cols:
            upsert_sql = sql.SQL("""
                INSERT INTO {tbl} ({cols})
                SELECT {cols} FROM {tmp}
                ON CONFLICT ({conflict}) DO UPDATE SET {upd}
            """).format(
                tbl=sql.Identifier(table),
                cols=sql.SQL(", ").join(col_ids),
                tmp=sql.Identifier(tmp),
                conflict=sql.SQL(", ").join(conflict_ids),
                upd=update_set,
            )
        else:
            upsert_sql = sql.SQL("""
                INSERT INTO {tbl} ({cols})
                SELECT {cols} FROM {tmp}
                ON CONFLICT ({conflict}) DO NOTHING
            """).format(
                tbl=sql.Identifier(table),
                cols=sql.SQL(", ").join(col_ids),
                tmp=sql.Identifier(tmp),
                conflict=sql.SQL(", ").join(conflict_ids),
            )
        cur.execute(upsert_sql)
        rows_affected = cur.rowcount

    conn.commit()
    return rows_affected


# ─────────────────────────────────────────────────────────────────────────────
# Table-specific loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_dim_time(conn, df: pd.DataFrame) -> int:
    keep = ["time_id", "year", "quarter", "month", "month_name",
            "week_of_year", "day_of_week", "day_name", "is_weekend", "is_holiday"]
    df = df[[c for c in keep if c in df.columns]].drop_duplicates("time_id")
    n = _upsert(conn, df, "dim_time", conflict_cols=["time_id"], update_cols=[])
    logger.info("dim_time: %d rows upserted.", n)
    return n


def load_aircraft_configs(conn, df: pd.DataFrame) -> int:
    """
    Load the flattened aircraft seat-map into aircraft_configs.
    """
    rename = {
        "aircraft":          "aircraft_variant",
        "cabin_class":       "cabin_class",
        "cabin_tier":        "cabin_tier",
        "Number of seats":   "seat_count",
        "Seat configuration":"seat_configuration",
        "Wi-Fi":             "has_wifi",
        "Power outlets":     "has_power",
    }
    out = df.rename(columns=rename)

    # Derive seat_pitch_in from "Standard seat pitch" (first numeric value)
    if "Standard seat pitch" in df.columns:
        out["seat_pitch_in"] = (
            df["Standard seat pitch"].str.extract(r'(\d+\.?\d*)')[0]
            .apply(pd.to_numeric, errors="coerce")
        )
    # Derive seat_width_in
    if "Seat width" in df.columns:
        out["seat_width_in"] = (
            df["Seat width"].str.extract(r'(\d+\.?\d*)')[0]
            .apply(pd.to_numeric, errors="coerce")
        )

    keep = ["aircraft_variant", "cabin_class", "cabin_tier", "seat_count",
            "seat_configuration", "seat_pitch_in", "seat_width_in",
            "has_wifi", "has_power"]
    out = out[[c for c in keep if c in out.columns]].drop_duplicates(
        subset=["aircraft_variant", "cabin_class"]
    )
    # Boolean normalisation
    for b in ["has_wifi", "has_power"]:
        if b in out.columns:
            out[b] = out[b].map(
                lambda v: True if str(v).lower() in ("true", "yes", "1") else
                          False if str(v).lower() in ("false", "no", "0") else None
            )

    n = _upsert(conn, out, "aircraft_configs",
                conflict_cols=["aircraft_variant", "cabin_class"],
                update_cols=["seat_count", "seat_configuration",
                             "seat_pitch_in", "seat_width_in",
                             "has_wifi", "has_power"])
    logger.info("aircraft_configs: %d rows upserted.", n)
    return n


def load_flights(conn, df: pd.DataFrame) -> int:
    keep = [
        "flight_id", "ident", "ident_iata", "operator_icao", "operator_iata",
        "flight_number", "registration", "aircraft_type", "aircraft_variant",
        "origin_iata", "destination_iata", "route_distance",
        "scheduled_out", "scheduled_off", "scheduled_on", "scheduled_in",
        "actual_out", "actual_off", "actual_on", "actual_in",
        "departure_delay", "arrival_delay", "status",
        "cancelled", "diverted",
        "seats_cabin_first", "seats_cabin_business", "seats_cabin_coach",
    ]
    col_renames = {
        "route_distance":        "route_distance_mi",
        "departure_delay":       "departure_delay_min",
        "arrival_delay":         "arrival_delay_min",
        "seats_cabin_first":     "fa_seats_first",
        "seats_cabin_business":  "fa_seats_business",
        "seats_cabin_coach":     "fa_seats_coach",
    }
    out = df[[c for c in keep if c in df.columns]].rename(columns=col_renames)
    out = out.drop_duplicates("flight_id")

    n = _upsert(conn, out, "flights",
                conflict_cols=["flight_id"],
                update_cols=["status", "actual_out", "actual_off",
                             "actual_on", "actual_in",
                             "departure_delay_min", "arrival_delay_min",
                             "aircraft_variant"])
    logger.info("flights: %d rows upserted.", n)
    return n


def load_flight_capacity(conn, df: pd.DataFrame) -> int:
    keep = ["flight_id", "aircraft_variant", "cabin_class", "cabin_tier",
            "seats", "route_distance_mi"]
    out = df[[c for c in keep if c in df.columns]].drop_duplicates(
        subset=["flight_id", "cabin_class"]
    )
    n = _upsert(conn, out, "flight_capacity",
                conflict_cols=["flight_id", "cabin_class"],
                update_cols=["seats"])
    logger.info("flight_capacity: %d rows upserted.", n)
    return n


def load_fuel_prices(conn, df: pd.DataFrame) -> int:
    keep = ["price_date", "jet_a_usd_per_gal"]
    out = df[[c for c in keep if c in df.columns]].drop_duplicates("price_date")
    n = _upsert(conn, out, "fuel_prices",
                conflict_cols=["price_date"],
                update_cols=["jet_a_usd_per_gal"])
    logger.info("fuel_prices: %d rows upserted.", n)
    return n


def load_aircraft_fuel_burn(conn) -> int:
    """Insert static fuel-burn rates from config."""
    from config import FUEL_BURN_GPH
    rows = [{"aircraft_variant": k, "gph_cruise": v, "gph_source": "POH/BTS-P12a"}
            for k, v in FUEL_BURN_GPH.items()]
    df = pd.DataFrame(rows)
    n = _upsert(conn, df, "aircraft_fuel_burn",
                conflict_cols=["aircraft_variant"],
                update_cols=["gph_cruise"])
    logger.info("aircraft_fuel_burn: %d rows upserted.", n)
    return n


def load_t100(conn, df: pd.DataFrame) -> int:
    keep = ["report_period", "carrier_code", "carrier_name", "aircraft_type_bts",
            "origin_iata", "destination_iata", "departures_performed",
            "seats_available", "passengers", "payload_lbs", "distance_mi"]
    out = df[[c for c in keep if c in df.columns]]
    n = _upsert(conn, out, "bts_t100_segments",
                conflict_cols=["report_period", "carrier_code",
                               "origin_iata", "destination_iata",
                               "aircraft_type_bts"],
                update_cols=["departures_performed", "seats_available",
                             "passengers", "payload_lbs", "distance_mi"])
    logger.info("bts_t100_segments: %d rows upserted.", n)
    return n


def load_db1b(conn, df: pd.DataFrame) -> int:
    keep = ["report_quarter", "carrier_code", "origin_iata", "destination_iata",
            "cabin_class_bts", "avg_fare_usd", "passengers", "miles"]
    out = df[[c for c in keep if c in df.columns]]
    n = _upsert(conn, out, "bts_db1b_fares",
                conflict_cols=["report_quarter", "carrier_code",
                               "origin_iata", "destination_iata",
                               "cabin_class_bts"],
                update_cols=["avg_fare_usd", "passengers", "miles"])
    logger.info("bts_db1b_fares: %d rows upserted.", n)
    return n


def log_etl_run(
    conn,
    pipeline_name: str,
    data_source: str,
    status: str,
    rows_extracted: int = 0,
    rows_loaded: int = 0,
    error_message: str | None = None,
    started_at: datetime | None = None,
) -> None:
    started_at = started_at or datetime.now(timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO etl_log
               (pipeline_name, data_source, status, rows_extracted,
                rows_loaded, error_message, started_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (pipeline_name, data_source, status, rows_extracted,
             rows_loaded, error_message, started_at)
        )
    conn.commit()
