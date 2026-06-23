"""
extract_sec_edgar.py
Pulls United Airlines Holdings 10-K labor expense data from SEC EDGAR.

Uses two SEC APIs (both free, no key required):
  1. EDGAR Full-Text Search  — locate the latest 10-K filing accession numbers
  2. EDGAR Submissions API   — confirm filing metadata
  3. EDGAR Filing Index      — download the actual XBRL/HTML document

What we extract:
  - Total employee count (average full-year)
  - Total salaries & wages ($M)
  - Total benefits ($M)
  - Total labor cost = wages + benefits ($M)
  - Broken down by year (from the multi-year comparison tables in the 10-K)

United Airlines Holdings CIK: 0000100517

SEC EDGAR APIs:
  Submissions: https://data.sec.gov/submissions/CIK{cik}.json
  Company facts: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
  EDGAR search: https://efts.sec.gov/LATEST/search-index?...

Usage:
    python extract_sec_edgar.py                   # latest 10-K
    python extract_sec_edgar.py --years 5         # last 5 years
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import DATA_DIR

logger = logging.getLogger(__name__)

# ── EDGAR constants ──────────────────────────────────────────────────────────
UAL_CIK       = "0000100517"           # United Airlines Holdings
_CIK_PADDED   = UAL_CIK.lstrip("0").zfill(10)
_SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{_CIK_PADDED}.json"
_COMPANY_FACTS_URL = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{_CIK_PADDED}.json"
_EDGAR_HEADERS = {
    "User-Agent": "UAL Revenue Optimization Project research@example.com",
    "Accept": "application/json",
}

# US-GAAP XBRL concept names for labor line items
_LABOR_CONCEPTS = {
    "LaborAndRelatedExpense":               "labor_total_usd",
    "SalariesAndWages":                     "salaries_wages_usd",
    "EmployeeBenefitsAndShareBasedCompensation": "benefits_usd",
    "DefinedBenefitPlanNetPeriodicBenefitCost":  "pension_cost_usd",
    "EntityNumberOfEmployees":              "employees_count",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, retries: int = 3) -> dict:
    for i in range(retries):
        try:
            r = requests.get(url, headers=_EDGAR_HEADERS, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as exc:
            if exc.response and exc.response.status_code == 429:
                wait = 2 ** (i + 1)
                logger.warning("Rate limited; waiting %ds …", wait)
                time.sleep(wait)
            elif i == retries - 1:
                raise
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fetch structured XBRL financial data via company-facts API
#    This is the cleanest path — no HTML parsing needed.
# ─────────────────────────────────────────────────────────────────────────────

def fetch_labor_from_xbrl(years: int = 10) -> pd.DataFrame:
    """
    Pull labor-related XBRL facts from EDGAR company-facts API.

    Returns a DataFrame with columns:
        fiscal_year, concept, value_usd (or count for employees)
    """
    logger.info("Fetching EDGAR company facts for UAL (CIK %s) …", UAL_CIK)
    data = _get(_COMPANY_FACTS_URL)
    if not data:
        logger.error("EDGAR company-facts returned empty response.")
        return pd.DataFrame()

    us_gaap = data.get("facts", {}).get("us-gaap", {})
    rows: list[dict] = []

    for concept_name, col_name in _LABOR_CONCEPTS.items():
        concept = us_gaap.get(concept_name, {})
        units = concept.get("units", {})

        # Prefer USD; fall back to pure (for employee counts)
        unit_data = units.get("USD", units.get("pure", []))
        for entry in unit_data:
            form = entry.get("form", "")
            if form not in ("10-K", "10-K/A"):
                continue
            end_date = entry.get("end", "")
            fy = entry.get("fy")
            val = entry.get("val")
            if fy and val is not None:
                rows.append({
                    "fiscal_year":  int(fy),
                    "period_end":   end_date,
                    "concept":      concept_name,
                    "column":       col_name,
                    "value":        float(val),
                    "form":         form,
                    "accession":    entry.get("accn", ""),
                })

    if not rows:
        logger.warning("No labor XBRL facts found for UAL.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Keep most recent filing for each (fiscal_year, concept)
    df = (df.sort_values("accession", ascending=False)
            .drop_duplicates(subset=["fiscal_year", "concept"])
            .sort_values("fiscal_year", ascending=False))

    # Filter to requested years
    if years:
        cutoff = df["fiscal_year"].max() - years + 1
        df = df[df["fiscal_year"] >= cutoff]

    # Pivot to wide format: one row per fiscal year
    wide = (df.pivot_table(
                index="fiscal_year",
                columns="column",
                values="value",
                aggfunc="last",
            )
            .reset_index()
            .rename_axis(None, axis=1))

    # Convert USD values from raw units → millions
    for col in wide.columns:
        if col.endswith("_usd") and col in wide.columns:
            wide[col] = (wide[col] / 1e6).round(2)
            wide = wide.rename(columns={col: col.replace("_usd", "_usd_m")})

    # Derive: total labor = wages + benefits (if LaborAndRelatedExpense missing)
    if "labor_total_usd_m" not in wide.columns:
        wage_col   = "salaries_wages_usd_m"
        ben_col    = "benefits_usd_m"
        if wage_col in wide.columns and ben_col in wide.columns:
            wide["labor_total_usd_m"] = (
                wide[wage_col].fillna(0) + wide[ben_col].fillna(0)
            ).round(2)

    logger.info("XBRL labor data: %d fiscal years extracted.", len(wide))
    return wide.sort_values("fiscal_year", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fetch 10-K filing list (for reference / manual inspection)
# ─────────────────────────────────────────────────────────────────────────────

def get_10k_filings(max_filings: int = 10) -> pd.DataFrame:
    """
    Return a list of United's most recent 10-K filings with accession numbers,
    filing dates, and document URLs.
    """
    logger.info("Fetching UAL 10-K filing list …")
    data = _get(_SUBMISSIONS_URL)
    filings = data.get("filings", {}).get("recent", {})

    forms       = filings.get("form", [])
    dates       = filings.get("filingDate", [])
    accessions  = filings.get("accessionNumber", [])
    descriptions= filings.get("primaryDocument", [])

    rows = [
        {"form": f, "filing_date": d, "accession": a, "primary_doc": p}
        for f, d, a, p in zip(forms, dates, accessions, descriptions)
        if f in ("10-K", "10-K/A")
    ][:max_filings]

    df = pd.DataFrame(rows)
    df["document_url"] = df["accession"].apply(
        lambda a: (
            "https://www.sec.gov/Archives/edgar/data/"
            + UAL_CIK.lstrip("0") + "/"
            + a.replace("-", "") + "/" + a + "-index.htm"
        )
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Save and summarise
# ─────────────────────────────────────────────────────────────────────────────

def fetch_and_save(years: int = 10, save_csv: bool = True) -> pd.DataFrame:
    """
    Full fetch → clean → save pipeline. Returns the labor DataFrame.
    """
    labor_df = fetch_labor_from_xbrl(years=years)
    if labor_df.empty:
        logger.warning("No labor data retrieved.")
        return labor_df

    if save_csv:
        path = DATA_DIR / "ual_10k_labor.csv"
        labor_df.to_csv(path, index=False)
        logger.info("Saved → %s", path)

    return labor_df


def derive_labor_cost_per_block_hour(
    labor_df: pd.DataFrame,
    form41_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge 10-K annual labor totals with Form 41 block hours to compute
    labor cost per block hour by year.

    Parameters
    ----------
    labor_df   : from fetch_labor_from_xbrl()
    form41_df  : from extract_bts_form41.load_form41(); if None, skips BH merge
    """
    out = labor_df.copy()

    if form41_df is not None and not form41_df.empty:
        bh_by_year = (
            form41_df[form41_df["carrier_code"] == "UA"]
            .assign(fiscal_year=lambda d: pd.to_datetime(d["report_quarter"]).dt.year)
            .groupby("fiscal_year")["block_hours"]
            .sum()
            .rename("ua_block_hours")
            .reset_index()
        )
        out = out.merge(bh_by_year, on="fiscal_year", how="left")
        if "labor_total_usd_m" in out.columns:
            out["labor_cost_per_bh_usd"] = (
                out["labor_total_usd_m"] * 1e6
                / out["ua_block_hours"].replace(0, float("nan"))
            ).round(2)

    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="SEC EDGAR 10-K labor data extractor")
    parser.add_argument("--years", type=int, default=10, help="Fiscal years to pull")
    parser.add_argument("--filings", action="store_true", help="List 10-K filing URLs")
    args = parser.parse_args()

    if args.filings:
        filings = get_10k_filings()
        print(filings.to_string(index=False))
    else:
        df = fetch_and_save(years=args.years)
        if not df.empty:
            print("\n── United Airlines 10-K Labor Costs ($M) ──")
            print(df.to_string(index=False))
