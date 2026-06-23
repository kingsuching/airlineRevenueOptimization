"""
extract_bts_form41.py
Downloads and parses BTS Form 41 Schedule P-12(a) — carrier-reported fuel and
labor expenses per block hour.

BTS Form 41 Schedule P-12(a)
  - Reports quarterly: total fuel gallons, fuel cost ($), block hours, employee
    count, and total operating expenses by airline.
  - Table ID: 216  (Air Carrier Financial — Form 41)
  - URL pattern: https://www.transtats.bts.gov/Download_Lookup.asp?...

Two access paths:
  A. BTS custom download (POST form) → ZIP → CSV  [implemented below]
  B. Pre-downloaded CSV in data/ (auto-detected)

After loading, this module cross-validates the BTS-reported gal/block-hour
against the POH-modelled values in fuel_burn_rates.json, flagging any
aircraft variant where the two differ by more than DIVERGENCE_THRESHOLD.

Usage:
    python extract_bts_form41.py                # load from data/ + validate
    python extract_bts_form41.py --download     # fetch latest from BTS
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import zipfile
from pathlib import Path

import pandas as pd
import requests

from config import DATA_DIR, TRACKED_OPERATORS

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
DIVERGENCE_THRESHOLD = 0.10   # flag if |POH - BTS| / BTS > 10%

# BTS Form 41 P-12(a) download — try candidates in order
_BTS_FORM41_URLS = [
    # Current BTS bulk download (as of 2025)
    "https://www.transtats.bts.gov/ftproot/PublicDL/Form41.zip",
    "https://www.transtats.bts.gov/ftproot/PublicDL/AirCarrier.zip",
    "https://www.transtats.bts.gov/ftproot/PublicDL/ACarrier.zip",       # legacy
]

# BTS TranStats custom-download form — used if bulk ZIP is unavailable
_BTS_FORM_URL = "https://www.transtats.bts.gov/DL_SelectFields.aspx"
_BTS_TABLE_ID = "216"   # Air Carrier Financial — Form 41 P-12(a)

# Field names used in the BTS Form 41 P-12(a) CSV
_P12A_COLUMNS = {
    "UNIQUE_CARRIER":       "carrier_code",
    "UNIQUE_CARRIER_NAME":  "carrier_name",
    "YEAR":                 "year",
    "QUARTER":              "quarter",
    "AIRCRAFT_CONFIG":      "aircraft_config_bts",  # BTS aircraft group code
    "AIRCRAFT_GROUP":       "aircraft_group_bts",
    "AIRCRAFT_TYPE":        "aircraft_type_bts",
    "FUEL_FLY":             "fuel_gallons",          # gallons consumed in flight
    "FUEL_COST_FLY":        "fuel_cost_usd",         # $ fuel cost
    "HOURS_AIRBORNE":       "hours_airborne",         # airborne hours
    "HOURS_RAMP_TO_RAMP":   "block_hours",            # ramp-to-ramp (block) hours
    "DEPARTURES_PERFORMED": "departures",
    "EMP_FT_PILOTS":        "emp_ft_pilots",
    "EMP_FT_COPILOTS":      "emp_ft_copilots",
    "EMP_FT_FLIGHT_ATT":    "emp_ft_flight_att",
    "SALARY_FT_PILOTS":     "salary_pilots_usd",     # total quarterly pilot salary
    "SALARY_FT_COPILOTS":   "salary_copilots_usd",
    "SALARY_FT_FLIGHT_ATT": "salary_fa_usd",
    "OP_REVENUES":          "op_revenues_usd",
    "OP_EXPENSES":          "op_expenses_usd",
}

# Map BTS aircraft_type_bts codes to United variants
# BTS type codes differ from FA ICAO codes; this is a separate crosswalk
BTS_AIRCRAFT_TYPE_MAP = {
    "77W": "B777-300ER",
    "772": "B777-200",
    "77L": "B777-200",
    "788": "B787-8",
    "789": "B787-9 Version 1",
    "78X": "B787-10",
    "763": "767-300ER Version 1",
    "764": "767-400ER",
    "752": "757-200",
    "753": "757-300",
    "737": "737-700",
    "738": "737-800 Version 1",
    "739": "737-900 Version 1",
    "7M8": "737 MAX 8 Version 1",
    "7M9": "737 MAX 9 Version 1",
    "319": "A319",
    "320": "A320",
    "32Q": "A321neo",
    "CRJ": "CRJ200",
    "CR5": "CRJ550",
    "CR7": "CRJ700",
    "E70": "Embraer E170",
    "E75": "Embraer E175 Version 1",
}

_IATA_CARRIERS = {"UA", "AA", "DL", "WN", "AS", "B6"}


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

def download_form41(save_path: Path | None = None) -> Path:
    """
    Download the BTS Form 41 carrier financial ZIP and save to disk.

    Tries multiple BTS bulk-download URLs in sequence, then falls back to
    BTS TranStats custom form download.

    Returns the path to the saved ZIP/CSV.
    """
    save_path = save_path or (DATA_DIR / "bts_form41_raw.zip")
    headers   = {"User-Agent": "UAL-Revenue-Research/1.0 research@example.com"}

    # ── Try known bulk-download ZIP URLs ─────────────────────────────────────
    for url in _BTS_FORM41_URLS:
        logger.info("Trying BTS bulk download: %s …", url)
        try:
            resp = requests.get(url, headers=headers, timeout=120, stream=True)
            if resp.status_code == 200 and int(resp.headers.get("Content-Length", 1)) > 1000:
                with open(save_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        f.write(chunk)
                logger.info("Saved → %s (%d bytes)", save_path, save_path.stat().st_size)
                return save_path
            logger.warning("URL %s returned HTTP %s — skipping.", url, resp.status_code)
        except requests.RequestException as exc:
            logger.warning("URL %s failed: %s — trying next.", url, exc)

    # ── Fallback: BTS TranStats form-based download ───────────────────────────
    logger.info("Falling back to BTS TranStats form download (table %s) …", _BTS_TABLE_ID)
    try:
        session = requests.Session()
        session.headers.update(headers)

        # Step 1: load the download page to get the form token
        page = session.get(
            f"https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FHK",
            timeout=30,
        )
        page.raise_for_status()

        # Step 2: extract hidden form fields (ViewState etc.)
        import re as _re
        vs    = _re.search(r'id="__VIEWSTATE"\s+value="([^"]+)"', page.text)
        vsg   = _re.search(r'id="__VIEWSTATEGENERATOR"\s+value="([^"]+)"', page.text)
        ev    = _re.search(r'id="__EVENTVALIDATION"\s+value="([^"]+)"', page.text)

        form_data = {
            "__VIEWSTATE":          vs.group(1)  if vs  else "",
            "__VIEWSTATEGENERATOR": vsg.group(1) if vsg else "",
            "__EVENTVALIDATION":    ev.group(1)  if ev  else "",
            "__EVENTTARGET":        "DL_URL",
            "UserTableName":        "Form 41 Financial Data - Schedule P 12(a)",
            "DBShortName":          "Air Carriers",
            "RawDataForm":          "216",
            "selectAll":            "on",
        }

        csv_save = save_path.with_suffix(".csv")
        resp2 = session.post(
            "https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FHK",
            data=form_data, timeout=180, stream=True,
        )
        resp2.raise_for_status()

        with open(csv_save, "wb") as f:
            for chunk in resp2.iter_content(chunk_size=1 << 16):
                f.write(chunk)

        logger.info("Form download saved → %s (%d bytes)", csv_save, csv_save.stat().st_size)
        return csv_save

    except Exception as exc:
        logger.error(
            "All BTS Form 41 download attempts failed: %s\n"
            "Manual download: https://www.transtats.bts.gov/databases.asp?"
            "Z_INT_Transportaton_Mode_ID=1&Z_INT_Group_ID=3\n"
            "Select 'Air Carrier Financial: Form 41 Schedules' → Schedule P-12(a) → Download ZIP.",
            exc,
        )
        raise RuntimeError("BTS Form 41 download failed — see log for manual download URL.") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Parse
# ─────────────────────────────────────────────────────────────────────────────

def load_form41(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load BTS Form 41 P-12(a) data.

    If `path` is None, auto-scans DATA_DIR for a file matching
    'form41' or 'ACarrier' (case-insensitive).
    """
    if path is None:
        candidates = [
            p for p in DATA_DIR.iterdir()
            if any(kw in p.name.upper()
                   for kw in ("FORM41", "ACARRIER", "P12A", "P_12A"))
        ]
        if not candidates:
            logger.warning(
                "No Form 41 file found in %s. "
                "Run with --download or place a CSV/ZIP in data/.", DATA_DIR
            )
            return pd.DataFrame()
        path = sorted(candidates)[-1]   # most recent
        logger.info("Auto-detected Form 41 file: %s", path)

    path = Path(path)
    logger.info("Loading BTS Form 41 from %s …", path)

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise ValueError(f"No CSV inside {path}")
            with zf.open(csv_names[0]) as f:
                raw = pd.read_csv(f, low_memory=False)
    else:
        raw = pd.read_csv(path, low_memory=False)

    raw.columns = [c.strip().upper() for c in raw.columns]
    present = {k: v for k, v in _P12A_COLUMNS.items() if k in raw.columns}
    df = raw[list(present.keys())].rename(columns=present)

    # Numeric coercion
    num_cols = [
        "fuel_gallons", "fuel_cost_usd", "hours_airborne",
        "block_hours", "departures",
        "emp_ft_pilots", "emp_ft_copilots", "emp_ft_flight_att",
        "salary_pilots_usd", "salary_copilots_usd", "salary_fa_usd",
        "op_revenues_usd", "op_expenses_usd",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Build report_quarter DATE
    if "year" in df.columns and "quarter" in df.columns:
        q_to_month = {1: "01", 2: "04", 3: "07", 4: "10"}
        df["report_quarter"] = pd.to_datetime(
            df["year"].astype(str) + "-"
            + df["quarter"].astype(int).map(q_to_month) + "-01",
            errors="coerce",
        ).dt.date
        df = df.drop(columns=["year", "quarter"], errors="ignore")

    # Filter to tracked carriers
    if "carrier_code" in df.columns:
        df = df[df["carrier_code"].isin(_IATA_CARRIERS)]

    # Derived: fuel cost per gallon, gph
    df["fuel_cost_per_gal"] = (
        df["fuel_cost_usd"] / df["fuel_gallons"].replace(0, float("nan"))
    ).round(4)
    df["gph_block_hour"] = (
        df["fuel_gallons"] / df["block_hours"].replace(0, float("nan"))
    ).round(1)

    # Map BTS aircraft type to United variant
    if "aircraft_type_bts" in df.columns:
        df["aircraft_variant"] = df["aircraft_type_bts"].map(BTS_AIRCRAFT_TYPE_MAP)

    logger.info("Form 41: %d rows loaded.", len(df))
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# POH cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_against_poh(
    form41_df: pd.DataFrame,
    poh_path: Path | None = None,
    threshold: float = DIVERGENCE_THRESHOLD,
) -> pd.DataFrame:
    """
    Compare BTS Form 41 gal/block-hour against POH estimates.

    Parameters
    ----------
    form41_df : output of load_form41()
    poh_path  : path to fuel_burn_rates.json (auto-resolved from aircraft_data/)
    threshold : flag rows where |poh - bts| / bts > threshold

    Returns
    -------
    DataFrame with one row per (aircraft_variant, carrier), columns:
        aircraft_variant, bts_avg_gph, poh_cruise_gph, poh_block_gph,
        divergence_pct, flag
    """
    if poh_path is None:
        poh_path = (
            Path(__file__).resolve().parents[1]
            / "aircraft_data" / "fuel_burn_rates.json"
        )

    with poh_path.open() as f:
        poh_raw = json.load(f)

    # Flatten POH JSON into a lookup {variant: {poh_cruise_gph, bts_block_hour_gph}}
    poh_lookup: dict[str, dict] = {}
    for _section, aircraft in poh_raw.items():
        if _section.startswith("_"):
            continue
        for _, entry in aircraft.items():
            if not isinstance(entry, dict):
                continue
            variant = entry.get("aircraft_variant")
            if variant:
                poh_lookup[variant] = {
                    "poh_cruise_gph":     entry.get("poh_cruise_gph"),
                    "poh_cruise_gph_low": entry.get("poh_cruise_gph_low"),
                    "poh_cruise_gph_high":entry.get("poh_cruise_gph_high"),
                    "bts_block_hour_gph_poh": entry.get("bts_block_hour_gph"),
                    "engine":             entry.get("engine"),
                    "poh_source":         entry.get("poh_source"),
                }

    # Aggregate Form 41 to annual averages per aircraft_variant
    if "aircraft_variant" not in form41_df.columns or form41_df.empty:
        logger.warning("Cannot validate: Form 41 lacks aircraft_variant column.")
        return pd.DataFrame()

    agg = (
        form41_df.dropna(subset=["aircraft_variant", "gph_block_hour"])
        .groupby("aircraft_variant")
        .agg(
            bts_avg_gph     = ("gph_block_hour", "mean"),
            bts_obs_count   = ("gph_block_hour", "count"),
            bts_fuel_gallons= ("fuel_gallons",   "sum"),
            bts_block_hours = ("block_hours",    "sum"),
        )
        .reset_index()
    )

    # Recompute weighted-average gph from totals (more accurate)
    agg["bts_weighted_gph"] = (
        agg["bts_fuel_gallons"] / agg["bts_block_hours"].replace(0, float("nan"))
    ).round(1)

    # Join POH data
    poh_df = pd.DataFrame.from_dict(poh_lookup, orient="index").reset_index()
    poh_df = poh_df.rename(columns={"index": "aircraft_variant"})
    validation = agg.merge(poh_df, on="aircraft_variant", how="outer")

    # Divergence calculation (vs BTS weighted gph)
    validation["divergence_pct"] = (
        (validation["bts_block_hour_gph_poh"] - validation["bts_weighted_gph"])
        / validation["bts_weighted_gph"].replace(0, float("nan"))
    ).abs().round(4)

    validation["flag"] = validation["divergence_pct"] > threshold
    validation["flag_reason"] = validation.apply(
        lambda r: (
            f"POH={r['bts_block_hour_gph_poh']:.0f} vs BTS={r['bts_weighted_gph']:.0f} gph "
            f"({r['divergence_pct']*100:.1f}%)"
            if r["flag"] else ""
        ),
        axis=1,
    )

    flagged = validation[validation["flag"]]
    if flagged.empty:
        logger.info("POH validation: all variants within %.0f%% of BTS actuals.", threshold * 100)
    else:
        logger.warning(
            "POH validation: %d variant(s) diverge >%.0f%% from BTS:\n%s",
            len(flagged), threshold * 100,
            flagged[["aircraft_variant", "bts_weighted_gph",
                      "bts_block_hour_gph_poh", "flag_reason"]].to_string(index=False),
        )

    return validation.sort_values("divergence_pct", ascending=False, na_position="last")


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate: gal/block-hour by carrier × aircraft type (for DB load)
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_form41(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce the summary table for loading into the warehouse:
    (report_quarter, carrier_code, aircraft_variant) →
    fuel_gallons, block_hours, gph_block_hour, fuel_cost_per_gal,
    salary_pilots_usd, salary_fa_usd
    """
    group = ["report_quarter", "carrier_code"]
    if "aircraft_variant" in df.columns:
        group.append("aircraft_variant")

    agg_cols = {c: "sum" for c in
                ["fuel_gallons", "fuel_cost_usd", "block_hours", "departures",
                 "salary_pilots_usd", "salary_copilots_usd", "salary_fa_usd",
                 "op_revenues_usd", "op_expenses_usd"]
                if c in df.columns}

    out = df.groupby(group, dropna=False).agg(agg_cols).reset_index()

    # Recalculate derived fields after aggregation
    out["gph_block_hour"] = (
        out["fuel_gallons"] / out["block_hours"].replace(0, float("nan"))
    ).round(1)
    out["fuel_cost_per_gal"] = (
        out["fuel_cost_usd"] / out["fuel_gallons"].replace(0, float("nan"))
    ).round(4)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="BTS Form 41 extractor + POH validator")
    parser.add_argument("--download", action="store_true",
                        help="Download latest Form 41 ZIP from BTS")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Cross-validate against POH (default)")
    args = parser.parse_args()

    if args.download:
        zip_path = download_form41()
        df = load_form41(zip_path)
    else:
        df = load_form41()

    if df.empty:
        print("No data loaded. Use --download to fetch from BTS.")
    else:
        summary = aggregate_form41(df)
        print(f"\nAggregated Form 41: {len(summary)} rows")
        print(summary.head(10).to_string())

        if args.validate:
            val = validate_against_poh(df)
            print(f"\nPOH validation ({len(val)} aircraft variants):")
            print(val[["aircraft_variant", "bts_weighted_gph",
                        "bts_block_hour_gph_poh", "divergence_pct", "flag"]].to_string())

        out = DATA_DIR / "bts_form41_summary.csv"
        summary.to_csv(out, index=False)
        print(f"\nSaved summary → {out}")
