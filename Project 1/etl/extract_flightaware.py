"""
extract_flightaware.py
Pulls scheduled/operated flight data from FlightAware AeroAPI.

Key endpoint used:
    GET /operators/{operator_id}/flights
    Returns up to `max_pages * page_size` flights per operator.

Outputs:
    - Returns a list of raw flight dicts (one per flight leg).
    - Optionally saves a dated CSV snapshot to DATA_DIR.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import (
    AEROAPI_BASE_URL,
    FA_API_KEY,
    TRACKED_OPERATORS,
    DATA_DIR,
)

logger = logging.getLogger(__name__)

# ── Tuning ────────────────────────────────────────────────────────────────────
DEFAULT_PAGE_SIZE = 15   # AeroAPI default
MAX_PAGES         = 10   # pull up to 150 flights per operator per run
RETRY_LIMIT       = 3
RETRY_BACKOFF_S   = 5


def _get(url: str, params: dict | None = None, headers: dict | None = None) -> dict:
    """HTTP GET with retry / back-off."""
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else "?"
            if status == 429:
                wait = RETRY_BACKOFF_S * attempt
                logger.warning("Rate-limited (%s). Waiting %ss …", status, wait)
                time.sleep(wait)
            elif attempt == RETRY_LIMIT:
                raise
            else:
                logger.warning("HTTP %s on attempt %d/%d. Retrying …", status, attempt, RETRY_LIMIT)
                time.sleep(RETRY_BACKOFF_S)
    raise RuntimeError(f"Failed after {RETRY_LIMIT} attempts: {url}")


def fetch_operator_flights(
    operator_id: str,
    max_pages: int = MAX_PAGES,
) -> list[dict[str, Any]]:
    """
    Paginate through /operators/{operator_id}/flights and return a flat list
    of flight dicts.  FlightAware returns 'links.next' when more pages exist.
    """
    url = f"{AEROAPI_BASE_URL}/operators/{operator_id}/flights"
    headers = {"x-apikey": FA_API_KEY}
    flights: list[dict] = []

    for page_num in range(1, max_pages + 1):
        logger.info("  Fetching %s flights — page %d …", operator_id, page_num)
        data = _get(url, headers=headers)
        batch = data.get("scheduled", data.get("flights", []))
        flights.extend(batch)

        cursor = data.get("links", {}).get("next")
        if not cursor:
            break
        # AeroAPI returns a relative path like "/operators/UAL/flights?cursor=…"
        url = AEROAPI_BASE_URL + cursor if cursor.startswith("/") else cursor

    logger.info("  %s: %d flights fetched.", operator_id, len(flights))
    return flights


def fetch_all_operators(
    operators: list[str] | None = None,
    save_csv: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Pull flights for every operator in `operators` (defaults to TRACKED_OPERATORS).

    Returns a dict of {operator_id: DataFrame}.
    If save_csv=True, each DataFrame is saved to DATA_DIR/{op}_flights.csv.
    """
    operators = operators or TRACKED_OPERATORS
    results: dict[str, pd.DataFrame] = {}

    for op in operators:
        logger.info("Fetching flights for operator: %s", op)
        try:
            raw = fetch_operator_flights(op)
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", op, exc)
            continue

        df = pd.DataFrame(raw)
        if df.empty:
            logger.warning("%s returned 0 flights.", op)
            continue

        # Normalise timestamps to UTC (they come as ISO-8601 strings with Z suffix)
        for ts_col in [
            "scheduled_out", "estimated_out", "actual_out",
            "scheduled_off", "estimated_off", "actual_off",
            "scheduled_on",  "estimated_on",  "actual_on",
            "scheduled_in",  "estimated_in",  "actual_in",
        ]:
            if ts_col in df.columns:
                df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

        if save_csv:
            out_path = DATA_DIR / f"{op}_flights.csv"
            df.to_csv(out_path, index=False)
            logger.info("  Saved → %s", out_path)

        results[op] = df

    return results


def load_flights_from_csv(
    operators: list[str] | None = None,
    data_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load previously-saved CSV snapshots (avoids a live API call during dev/test).
    Returns the same {operator_id: DataFrame} structure as fetch_all_operators().
    """
    operators  = operators or TRACKED_OPERATORS
    data_dir   = data_dir or DATA_DIR
    results: dict[str, pd.DataFrame] = {}

    for op in operators:
        path = data_dir / f"{op}_flights.csv"
        if not path.exists():
            logger.warning("CSV not found for %s at %s — skipping.", op, path)
            continue
        df = pd.read_csv(path)
        # Re-parse timestamp columns
        for ts_col in [
            "scheduled_out", "actual_out", "scheduled_off",
            "actual_off", "scheduled_on", "actual_on",
            "scheduled_in", "actual_in",
        ]:
            if ts_col in df.columns:
                df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        results[op] = df
        logger.info("Loaded %d rows for %s from %s.", len(df), op, path)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    dfs = fetch_all_operators()
    total = sum(len(d) for d in dfs.values())
    print(f"Fetched {total} flights across {len(dfs)} operators.")
