"""
extract_bts.py
Downloads and parses BTS (Bureau of Transportation Statistics) data:

1. T-100 Segment Data  — route-level traffic / load factor (monthly, carrier-level)
   URL: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FMF&QO_fu146_anzpbqr=N&rnd=0

2. DB1B Coupon/Market  — 10% itinerary ticket sample with fares by cabin (quarterly)
   URL: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FIM

Both datasets are bulk downloads (ZIP → CSV).  This module handles:
  - Downloading the zip file (or loading from local cache)
  - Extracting and parsing the CSV
  - Filtering to the carriers in TRACKED_OPERATORS
  - Returning a clean DataFrame

Because BTS zip downloads require form interactions, this module supports
two paths:
  a. Automated download via the BTS "direct download" API endpoint (recent months)
  b. Manual drop-in: place the unzipped CSV in DATA_DIR and call load_*_from_csv()
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from config import DATA_DIR, TRACKED_OPERATORS

logger = logging.getLogger(__name__)

# ── BTS operator IATA codes (T-100 uses 2-letter IATA) ──────────────────────
BTS_CARRIER_MAP = {
    "UAL": "UA",
    "AAL": "AA",
    "DL":  "DL",
    "WN":  "WN",
    "AS":  "AS",
    "B6":  "B6",
}
_IATA_CARRIERS = list(BTS_CARRIER_MAP.values())


# ─────────────────────────────────────────────────────────────────────────────
# T-100 Segment
# ─────────────────────────────────────────────────────────────────────────────

# BTS direct-download URL for T-100 Domestic Segment
# Replace {YEAR} and {MONTH} with e.g. 2024, 1
_T100_URL = (
    "https://www.transtats.bts.gov/Download_Lookup.asp"
    "?Lookup=L_CARRIER_HISTORY"   # placeholder — real download needs POST
)

# Columns we care about from T-100 Segment
T100_COLUMNS = {
    "UNIQUE_CARRIER":        "carrier_code",
    "UNIQUE_CARRIER_NAME":   "carrier_name",
    "AIRCRAFT_TYPE":         "aircraft_type_bts",
    "ORIGIN":                "origin_iata",
    "DEST":                  "destination_iata",
    "YEAR":                  "year",
    "MONTH":                 "month",
    "DEPARTURES_PERFORMED":  "departures_performed",
    "SEATS":                 "seats_available",
    "PASSENGERS":            "passengers",
    "PAYLOAD":               "payload_lbs",
    "DISTANCE":              "distance_mi",
}


def load_t100_from_csv(path: str | Path, filter_carriers: bool = True) -> pd.DataFrame:
    """
    Parse a T-100 Segment CSV downloaded manually from BTS.

    Parameters
    ----------
    path            : path to the CSV (or ZIP containing a single CSV)
    filter_carriers : if True, keep only carriers in _IATA_CARRIERS

    Returns
    -------
    Cleaned DataFrame with standardised column names + report_period DATE column.
    """
    path = Path(path)
    logger.info("Loading T-100 data from %s …", path)

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV found inside {path}")
            with zf.open(csv_names[0]) as f:
                raw = pd.read_csv(f)
    else:
        raw = pd.read_csv(path)

    # Keep only columns we need (case-insensitive header match)
    raw.columns = [c.strip().upper() for c in raw.columns]
    present = {k: v for k, v in T100_COLUMNS.items() if k in raw.columns}
    df = raw[list(present.keys())].rename(columns=present)

    # Numeric coercion
    for col in ["departures_performed", "seats_available", "passengers", "payload_lbs", "distance_mi"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build report_period (first day of the month)
    if "year" in df.columns and "month" in df.columns:
        df["report_period"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
        ).dt.date
        df = df.drop(columns=["year", "month"])

    if filter_carriers and "carrier_code" in df.columns:
        df = df[df["carrier_code"].isin(_IATA_CARRIERS)]

    logger.info("T-100: %d rows loaded (%d carriers).", len(df),
                df["carrier_code"].nunique() if "carrier_code" in df.columns else 0)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# DB1B Coupon/Market
# ─────────────────────────────────────────────────────────────────────────────

# Columns from DB1B Market dataset that we use
DB1B_COLUMNS = {
    "REPORTING_CARRIER":  "carrier_code",
    "YEAR":               "year",
    "QUARTER":            "quarter",
    "ORIGIN":             "origin_iata",
    "DEST":               "destination_iata",
    "MARKET_FARE":        "avg_fare_usd",
    "PASSENGERS":         "passengers",
    "MARKET_MILES_FLOWN": "miles",
    "CABIN_CLASS":        "cabin_class_bts",    # if present in the coupon table
}


def load_db1b_from_csv(path: str | Path, filter_carriers: bool = True) -> pd.DataFrame:
    """
    Parse a DB1B Market or Coupon CSV downloaded from BTS.

    Parameters
    ----------
    path            : path to the CSV (or ZIP)
    filter_carriers : if True, keep only carriers in _IATA_CARRIERS

    Returns
    -------
    DataFrame with standardized columns + report_quarter DATE column.
    """
    path = Path(path)
    logger.info("Loading DB1B data from %s …", path)

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV found inside {path}")
            with zf.open(csv_names[0]) as f:
                raw = pd.read_csv(f, low_memory=False)
    else:
        raw = pd.read_csv(path, low_memory=False)

    raw.columns = [c.strip().upper() for c in raw.columns]
    present = {k: v for k, v in DB1B_COLUMNS.items() if k in raw.columns}
    df = raw[list(present.keys())].rename(columns=present)

    for col in ["avg_fare_usd", "passengers", "miles"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build report_quarter (first day of the quarter)
    if "year" in df.columns and "quarter" in df.columns:
        month_start = {1: "01", 2: "04", 3: "07", 4: "10"}
        df["report_quarter"] = pd.to_datetime(
            df["year"].astype(str) + "-"
            + df["quarter"].map(month_start) + "-01"
        ).dt.date
        df = df.drop(columns=["year", "quarter"])

    # Aggregate to route/carrier/cabin level (DB1B is individual itineraries)
    group_cols = [c for c in
                  ["report_quarter", "carrier_code", "origin_iata",
                   "destination_iata", "cabin_class_bts"]
                  if c in df.columns]
    if group_cols:
        df = (df.groupby(group_cols, dropna=False)
                .agg(avg_fare_usd=("avg_fare_usd", "mean"),
                     passengers=("passengers", "sum"),
                     miles=("miles", "mean"))
                .reset_index())
        df["avg_fare_usd"] = df["avg_fare_usd"].round(2)

    if filter_carriers and "carrier_code" in df.columns:
        df = df[df["carrier_code"].isin(_IATA_CARRIERS)]

    logger.info("DB1B: %d route-quarter rows loaded.", len(df))
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: try to auto-detect and load any BTS file dropped in DATA_DIR
# ─────────────────────────────────────────────────────────────────────────────

def auto_load_bts(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Scan DATA_DIR for any BTS file and return a dict:
      {'t100': df, 'db1b': df}
    Files must match patterns:
      T-100 → filename contains 'T_100' or 'T100'
      DB1B  → filename contains 'DB1B'
    """
    data_dir = data_dir or DATA_DIR
    result: dict[str, pd.DataFrame] = {}

    for p in sorted(data_dir.iterdir()):
        name = p.name.upper()
        if "T100" in name or "T_100" in name:
            result["t100"] = load_t100_from_csv(p)
        elif "DB1B" in name:
            result["db1b"] = load_db1b_from_csv(p)

    if not result:
        logger.info(
            "No BTS files found in %s. "
            "Download T-100 and DB1B from https://www.transtats.bts.gov/ "
            "and place in the data/ folder.", data_dir
        )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    dfs = auto_load_bts()
    for name, df in dfs.items():
        print(f"\n── {name.upper()} ({len(df)} rows) ──")
        print(df.head(3).to_string())
