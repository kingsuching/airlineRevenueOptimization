"""
extract_eia_fuel.py
Pulls weekly U.S. Gulf Coast Kerosene-Type Jet Fuel spot prices from the EIA API v2.

Series: PET.EER_EPJK_PF4_RGC_DPG.W
Units:  Dollars per Gallon

EIA API v2 docs: https://www.eia.gov/opendata/documentation.php
Register for a free key at: https://www.eia.gov/opendata/register.php

Outputs:
    DataFrame with columns [price_date (DATE), jet_a_usd_per_gal (float)]
    Optionally saves to DATA_DIR/eia_fuel_prices.csv
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import requests

from config import EIA_API_KEY, EIA_API_URL, EIA_SERIES_ID, DATA_DIR

logger = logging.getLogger(__name__)

# EIA caps a single request at 5 000 rows; jet fuel weekly → ~96 yrs; we're safe.
_DEFAULT_START = "2010-01-01"


def fetch_jet_fuel_prices(
    start_date: str = _DEFAULT_START,
    end_date: str | None = None,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Fetch weekly Jet-A spot prices from EIA API v2.

    Parameters
    ----------
    start_date : ISO date string, e.g. "2020-01-01"
    end_date   : ISO date string (defaults to today)
    save_csv   : if True, saves result to DATA_DIR/eia_fuel_prices.csv

    Returns
    -------
    DataFrame with columns: price_date, jet_a_usd_per_gal
    """
    end_date = end_date or date.today().isoformat()

    params = {
        "api_key":  EIA_API_KEY,
        "frequency": "weekly",
        "data[0]":   "value",
        "facets[series][]": EIA_SERIES_ID,
        "start":     start_date,
        "end":       end_date,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }

    logger.info("Fetching EIA Jet-A prices %s → %s …", start_date, end_date)
    try:
        resp = requests.get(EIA_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.error("EIA fetch failed: %s", exc)
        return _load_from_csv()

    rows = payload.get("response", {}).get("data", [])
    if not rows:
        logger.warning("EIA returned 0 rows — falling back to CSV cache.")
        return _load_from_csv()

    df = pd.DataFrame(rows)
    df = df.rename(columns={"period": "price_date", "value": "jet_a_usd_per_gal"})
    df["price_date"]          = pd.to_datetime(df["price_date"]).dt.date
    df["jet_a_usd_per_gal"]   = pd.to_numeric(df["jet_a_usd_per_gal"], errors="coerce")
    df = df[["price_date", "jet_a_usd_per_gal"]].dropna().sort_values("price_date")

    if save_csv:
        out = DATA_DIR / "eia_fuel_prices.csv"
        df.to_csv(out, index=False)
        logger.info("Saved %d weekly prices → %s", len(df), out)

    return df


def _load_from_csv() -> pd.DataFrame:
    """Load cached EIA prices from disk if available."""
    path = DATA_DIR / "eia_fuel_prices.csv"
    if path.exists():
        logger.info("Loading EIA prices from cache: %s", path)
        df = pd.read_csv(path, parse_dates=["price_date"])
        df["price_date"] = df["price_date"].dt.date
        return df
    logger.warning("No EIA cache at %s — returning empty DataFrame.", path)
    return pd.DataFrame(columns=["price_date", "jet_a_usd_per_gal"])


def match_fuel_price_to_flight(
    flights_df: pd.DataFrame,
    fuel_df: pd.DataFrame,
    flight_date_col: str = "flight_date",
) -> pd.DataFrame:
    """
    Attach the most-recent weekly fuel price that is ≤ each flight date.

    Uses pandas merge_asof (backward match on sorted date).

    Parameters
    ----------
    flights_df     : DataFrame with a date column (flight_date_col)
    fuel_df        : DataFrame with price_date and jet_a_usd_per_gal
    flight_date_col: name of the date column in flights_df

    Returns
    -------
    flights_df with an added column 'jet_a_usd_per_gal'
    """
    if fuel_df.empty:
        flights_df["jet_a_usd_per_gal"] = None
        return flights_df

    f = flights_df.copy()
    p = fuel_df.copy()

    f["_date_key"] = pd.to_datetime(f[flight_date_col])
    p["_date_key"] = pd.to_datetime(p["price_date"])
    p = p.sort_values("_date_key")
    f = f.sort_values("_date_key")

    merged = pd.merge_asof(
        f, p[["_date_key", "jet_a_usd_per_gal"]],
        on="_date_key", direction="backward"
    ).drop(columns=["_date_key"])

    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    df = fetch_jet_fuel_prices(start_date="2020-01-01")
    print(df.tail())
    print(f"\nLatest price: ${df['jet_a_usd_per_gal'].iloc[-1]:.4f}/gal on {df['price_date'].iloc[-1]}")
