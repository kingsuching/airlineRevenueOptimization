"""
config.py
Centralised configuration for the Project 1 ETL pipeline.

Reads API keys from ../../apiKeys/apiKeys.json and environment variables for
database credentials.  All other constants live here so they're easy to tune
without touching pipeline logic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parents[2]   # airlineRevenueOptimization/
PROJECT_DIR = Path(__file__).resolve().parent.parent # Project 1/
DATA_DIR    = PROJECT_DIR / "data"
SQL_DIR     = PROJECT_DIR / "sql"
LOG_DIR     = PROJECT_DIR / "logs"

LOG_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
_KEY_FILE = REPO_ROOT / "apiKeys" / "apiKeys.json"
with _KEY_FILE.open() as _f:
    _KEYS = json.load(_f)

FA_API_KEY  = _KEYS.get("FA_API_KEY", "")
EIA_API_KEY = _KEYS.get("EIA_API_KEY", "")   # register free at https://www.eia.gov/opendata/

# ---------------------------------------------------------------------------
# FlightAware AeroAPI
# ---------------------------------------------------------------------------
AEROAPI_BASE_URL = "https://aeroapi.flightaware.com/aeroapi"

# Airlines to track (ICAO codes)
TRACKED_OPERATORS = ["UAL", "AAL", "DL", "WN", "AS", "B6"]

# Number of scheduled flights to pull per operator per call
FA_MAX_FLIGHTS = 200   # AeroAPI default page size is 15; set ident_filter for bulk

# ---------------------------------------------------------------------------
# EIA Jet-A fuel price series
# ---------------------------------------------------------------------------
EIA_SERIES_ID = "PET.EER_EPJK_PF4_RGC_DPG.W"  # Weekly U.S. Gulf Coast Kerosene-Type Jet Fuel
EIA_API_URL   = "https://api.eia.gov/v2/petroleum/pri/spt/data/"

# ---------------------------------------------------------------------------
# Aircraft type code → United seat-map variant mapping
# (FlightAware ICAO aircraft code → key in UAL_aircraft_data.json)
# ---------------------------------------------------------------------------
AIRCRAFT_TYPE_MAP: dict[str, str] = {
    # ── Wide-bodies ────────────────────────────────────────────────────────
    "B77L": "B777-200",           # 777-200LR (same interior as -200)
    "B772": "B777-200",           # 777-200
    "B77W": "B777-300ER",         # 777-300ER
    "B788": "B787-8",             # 787-8 Dreamliner
    "B789": "B787-9 Version 1",   # 787-9 (default to V1)
    "B78X": "B787-10",            # 787-10
    "B763": "767-300ER Version 1",# 767-300ER (default to V1)
    "B764": "767-400ER",          # 767-400ER
    "B752": "757-200",            # 757-200
    "B753": "757-300",            # 757-300
    # ── Narrow-bodies ──────────────────────────────────────────────────────
    "B737": "737-700",            # 737-700
    "B738": "737-800 Version 1",  # 737-800 (default to V1)
    "B739": "737-900 Version 1",  # 737-900/ER (default to V1)
    "B38M": "737 MAX 8 Version 1",# 737 MAX 8 (default to V1)
    "B39M": "737 MAX 9 Version 1",# 737 MAX 9 (default to V1)
    "A319": "A319",
    "A320": "A320",
    "A21N": "A321neo",            # A321neo XLR (UA calls it A321XLR)
    # ── Regional jets ──────────────────────────────────────────────────────
    "CRJ2": "CRJ200",
    "CL65": "CRJ550",             # CRJ-550 uses CL-65 type cert
    "CRJ7": "CRJ700",
    "E170": "Embraer E170",
    "E75L": "Embraer E175 Version 1",  # E175 long (default to V1)
    "E75S": "Embraer E175 Version 1",  # E175 short (same config)
}

# Canonical cabin-tier labels used in the warehouse
CABIN_TIER_MAP: dict[str, str] = {
    "United Polaris® Business Class":  "business",
    "United Polaris® business class":  "business",
    "Polaris® Business Class":         "business",
    "United First®":                   "first",
    "United Business®":                "first",    # domestic first
    "United® Premium Plus":            "premium_plus",
    "Premium Plus®":                   "premium_plus",
    "United Economy Plus®":            "economy_plus",
    "Economy Plus®":                   "economy_plus",
    "United Economy®":                 "economy",
}

# ---------------------------------------------------------------------------
# Aircraft cruise fuel burn (gal/hr) — from POH / BTS P-12(a)
# ---------------------------------------------------------------------------
FUEL_BURN_GPH: dict[str, float] = {
    "B777-200":           1550.0,
    "B777-200ER Version 1": 1550.0,
    "B777-200ER Version 2": 1550.0,
    "B777-300ER":         1650.0,
    "B787-8":              700.0,
    "B787-9 Version 1":    780.0,
    "B787-9 Version 2":    780.0,
    "B787-10":             830.0,
    "767-300ER Version 1": 950.0,
    "767-300ER Version 2": 950.0,
    "767-400ER":          1000.0,
    "757-200":             590.0,
    "757-300":             620.0,
    "737-700":             480.0,
    "737-800 Version 1":   520.0,
    "737-800 Version 2":   520.0,
    "737-800 Version 3":   520.0,
    "737-900 Version 1":   550.0,
    "737-900 Version 2":   550.0,
    "737-900 Version 3":   550.0,
    "737 MAX 8 Version 1": 430.0,
    "737 MAX 8 Version 2": 430.0,
    "737 MAX 9 Version 1": 460.0,
    "737 MAX 9 Version 2": 460.0,
    "A319":                480.0,
    "A320":                510.0,
    "A321neo":             440.0,
    "CRJ200":              195.0,
    "CRJ550":              220.0,
    "CRJ700":              280.0,
    "Embraer E170":        260.0,
    "Embraer E175 Version 1": 280.0,
    "Embraer E175 Version 2": 280.0,
}

# ---------------------------------------------------------------------------
# PostgreSQL connection (override via environment variables)
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host":     os.getenv("PG_HOST",     "localhost"),
    "port":     int(os.getenv("PG_PORT", "5432")),
    "dbname":   os.getenv("PG_DB",       "airline_revenue"),
    "user":     os.getenv("PG_USER",     "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}
DB_DSN = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)
