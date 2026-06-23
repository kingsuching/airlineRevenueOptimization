"""
make_sample_data.py
Generate synthetic route_demand_summary.csv and fuel_prices.csv from
Project 1 FlightAware data so Project 3 can run without live BTS T-100 data.

Run once from the Project 3/analysis/ directory:
    python make_sample_data.py
"""

from __future__ import annotations

import ast
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

_HERE        = Path(__file__).resolve().parent
_P1_DATA     = _HERE.parents[1] / "Project 1" / "data"
_P1_AIRCRAFT = _HERE.parents[1] / "Project 1" / "aircraft_data"
_P2_OUTPUTS  = _HERE.parents[1] / "Project 2" / "outputs"
_P2_OUTPUTS.mkdir(exist_ok=True)
_P1_DATA.mkdir(exist_ok=True)

# ── Aircraft type → variant & seat count ─────────────────────────────────────
AC_MAP = {
    "B737": ("737-700",             126),
    "B738": ("737-800 Version 1",   166),
    "B739": ("737-900 Version 1",   179),
    "B38M": ("737 MAX 8 Version 1", 160),
    "B39M": ("737 MAX 9 Version 1", 179),
    "A319": ("A319",                126),
    "A320": ("A320",                150),
    "A21N": ("A321neo",             196),
    "B752": ("757-200",             167),
    "B763": ("767-300ER Version 1", 214),
    "B788": ("B787-8",              219),
    "B789": ("B787-9 Version 1",    252),
    "B77W": ("B777-300ER",          369),
    "B772": ("B777-200",            364),
    "E75L": ("Embraer E175 Version 1", 76),
    "CL65": ("CRJ550",              50),
}

SEASONAL_INDEX = {
    1: 0.88, 2: 0.86, 3: 0.95, 4: 0.96,
    5: 1.02, 6: 1.10, 7: 1.12, 8: 1.08,
    9: 0.97, 10: 0.96, 11: 0.93, 12: 1.01,
}


def _iata(cell: str) -> str | None:
    try:
        d = ast.literal_eval(cell)
        return d.get("code_iata") or d.get("code_lid")
    except Exception:
        return None


def _load_ual_routes() -> list[dict]:
    flights = pd.read_csv(_P1_DATA / "UAL_flights.csv")
    routes = []
    for _, row in flights.iterrows():
        orig = _iata(str(row["origin"]))
        dest = _iata(str(row["destination"]))
        dist = row.get("route_distance", 0)
        ac   = str(row.get("aircraft_type", "B738"))
        if not orig or not dest or float(dist) < 50:
            continue
        variant, seats = AC_MAP.get(ac, ("737-800 Version 1", 166))
        routes.append({
            "origin_iata":      orig,
            "destination_iata": dest,
            "aircraft_type_bts": ac,
            "aircraft_variant": variant,
            "seats_per_dep":    seats,
            "distance_mi":      int(dist),
        })

    # Deduplicate on (origin, dest, ac_type)
    seen = set()
    unique = []
    for r in routes:
        key = (r["origin_iata"], r["destination_iata"], r["aircraft_type_bts"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def _extra_routes() -> list[dict]:
    """Add representative UA routes not in the FlightAware snapshot."""
    return [
        {"origin_iata":"EWR","destination_iata":"LAX","aircraft_type_bts":"B77W",
         "aircraft_variant":"B777-300ER","seats_per_dep":369,"distance_mi":2453},
        {"origin_iata":"ORD","destination_iata":"SFO","aircraft_type_bts":"B789",
         "aircraft_variant":"B787-9 Version 1","seats_per_dep":252,"distance_mi":1843},
        {"origin_iata":"EWR","destination_iata":"SFO","aircraft_type_bts":"B789",
         "aircraft_variant":"B787-9 Version 1","seats_per_dep":252,"distance_mi":2565},
        {"origin_iata":"IAH","destination_iata":"LAX","aircraft_type_bts":"B738",
         "aircraft_variant":"737-800 Version 1","seats_per_dep":166,"distance_mi":1379},
        {"origin_iata":"ORD","destination_iata":"LAX","aircraft_type_bts":"B739",
         "aircraft_variant":"737-900 Version 1","seats_per_dep":179,"distance_mi":1744},
        {"origin_iata":"ORD","destination_iata":"EWR","aircraft_type_bts":"B738",
         "aircraft_variant":"737-800 Version 1","seats_per_dep":166,"distance_mi":719},
        {"origin_iata":"ORD","destination_iata":"IAH","aircraft_type_bts":"B738",
         "aircraft_variant":"737-800 Version 1","seats_per_dep":166,"distance_mi":925},
        {"origin_iata":"IAH","destination_iata":"ORD","aircraft_type_bts":"B38M",
         "aircraft_variant":"737 MAX 8 Version 1","seats_per_dep":160,"distance_mi":925},
        {"origin_iata":"EWR","destination_iata":"MIA","aircraft_type_bts":"B739",
         "aircraft_variant":"737-900 Version 1","seats_per_dep":179,"distance_mi":1094},
        {"origin_iata":"IAD","destination_iata":"ORD","aircraft_type_bts":"A319",
         "aircraft_variant":"A319","seats_per_dep":126,"distance_mi":589},
        {"origin_iata":"SFO","destination_iata":"ORD","aircraft_type_bts":"B789",
         "aircraft_variant":"B787-9 Version 1","seats_per_dep":252,"distance_mi":1843},
        {"origin_iata":"LAX","destination_iata":"EWR","aircraft_type_bts":"B77W",
         "aircraft_variant":"B777-300ER","seats_per_dep":369,"distance_mi":2453},
        # Regional jets
        {"origin_iata":"ORD","destination_iata":"MSN","aircraft_type_bts":"E75L",
         "aircraft_variant":"Embraer E175 Version 1","seats_per_dep":76,"distance_mi":115},
        {"origin_iata":"IAH","destination_iata":"AUS","aircraft_type_bts":"E75L",
         "aircraft_variant":"Embraer E175 Version 1","seats_per_dep":76,"distance_mi":148},
        {"origin_iata":"ORD","destination_iata":"DTW","aircraft_type_bts":"CL65",
         "aircraft_variant":"CRJ550","seats_per_dep":50,"distance_mi":238},
        # Wide-body domestic
        {"origin_iata":"EWR","destination_iata":"HNL","aircraft_type_bts":"B789",
         "aircraft_variant":"B787-9 Version 1","seats_per_dep":252,"distance_mi":5095},
        {"origin_iata":"SFO","destination_iata":"HNL","aircraft_type_bts":"B763",
         "aircraft_variant":"767-300ER Version 1","seats_per_dep":214,"distance_mi":2398},
        # Max routes
        {"origin_iata":"EWR","destination_iata":"DEN","aircraft_type_bts":"B39M",
         "aircraft_variant":"737 MAX 9 Version 1","seats_per_dep":179,"distance_mi":1605},
        {"origin_iata":"ORD","destination_iata":"DEN","aircraft_type_bts":"B39M",
         "aircraft_variant":"737 MAX 9 Version 1","seats_per_dep":179,"distance_mi":920},
    ]


# ── Departures per month heuristics ──────────────────────────────────────────

def _monthly_departures(dist_mi: int, seats: int) -> int:
    if seats >= 250:   base = 30   # widebody
    elif seats >= 150: base = 60   # mainline narrowbody
    elif seats >= 100: base = 75
    else:              base = 45   # regional

    if dist_mi < 300:  return int(base * 1.4)
    if dist_mi > 3000: return int(base * 0.6)
    return base


def _base_lf(dist_mi: int, seats: int) -> float:
    """Base annual-average load factor by route characteristics."""
    if seats >= 250:
        return 0.86
    if dist_mi < 300:
        return 0.74
    if dist_mi > 2000:
        return 0.88
    return 0.82


def build_demand(routes: list[dict], periods: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for r in routes:
        base_lf  = _base_lf(r["distance_mi"], r["seats_per_dep"])
        base_dep = _monthly_departures(r["distance_mi"], r["seats_per_dep"])

        # Slight year-over-year growth (1.5–3%)
        for period in periods:
            month  = period.month
            year   = period.year
            yr_idx = (year - periods[0].year)

            season_adj = SEASONAL_INDEX[month]
            # Florida routes (MCO, MIA, FLL, TPA): inverted (peak in Jan–Mar)
            if r["destination_iata"] in ("MCO","MIA","FLL","TPA","SRQ") or \
               r["origin_iata"]      in ("MCO","MIA","FLL","TPA","SRQ"):
                season_adj = {1:1.12,2:1.10,3:1.08,4:1.02,5:0.95,6:0.90,
                              7:0.88,8:0.88,9:0.92,10:0.96,11:1.00,12:1.05}[month]

            # HNL: peak summer + December
            if "HNL" in (r["origin_iata"], r["destination_iata"]):
                season_adj = {1:0.92,2:0.88,3:0.93,4:0.98,5:1.02,6:1.08,
                              7:1.15,8:1.12,9:0.96,10:0.92,11:0.93,12:1.10}[month]

            yoy_growth = 1 + yr_idx * (0.015 + 0.015 * np.random.random())
            noise      = 1 + np.random.normal(0, 0.015)

            lf         = float(np.clip(base_lf * season_adj * yoy_growth * noise, 0.55, 0.99))
            dep        = max(1, int(base_dep * season_adj * yoy_growth + np.random.normal(0, 3)))
            seats_avail = dep * r["seats_per_dep"]
            pax         = int(seats_avail * lf)
            dist        = r["distance_mi"]

            rows.append({
                "report_period":     period,
                "carrier_code":      "UA",
                "carrier_name":      "United Air Lines Inc.",
                "origin_iata":       r["origin_iata"],
                "destination_iata":  r["destination_iata"],
                "aircraft_type_bts": r["aircraft_type_bts"],
                "aircraft_variant":  r["aircraft_variant"],
                "departures":        dep,
                "seats_available":   seats_avail,
                "passengers":        pax,
                "load_factor":       round(lf, 4),
                "load_factor_imputed": round(lf, 4),
                "lf_source":         "synthetic",
                "lf_outlier_flag":   False,
                "distance_mi":       dist,
                "asm":               seats_avail * dist,
                "rpm":               pax * dist,
                "payload_lbs":       int(pax * 215),  # ~215 lb/pax incl. bags
                "data_source":       "synthetic_t100",
            })

    return pd.DataFrame(rows)


def build_fuel_prices(periods: list[pd.Timestamp]) -> pd.DataFrame:
    """Synthetic weekly Jet-A prices (USD/gal) with trend + volatility."""
    rows = []
    start_price = 2.10
    for i, p in enumerate(pd.date_range(periods[0] - pd.offsets.Day(7),
                                         periods[-1] + pd.offsets.Day(31), freq="W-MON")):
        # Trend up from 2.10 → ~2.80 over 2 years, with cycles
        t     = i / (len(periods) * 4.3)
        cycle = 0.15 * np.sin(2 * np.pi * t * 2)      # seasonal cycle
        trend = 0.60 * t                                 # mild upward trend
        noise = np.random.normal(0, 0.04)
        price = round(float(start_price + trend + cycle + noise), 4)
        rows.append({"price_date": p.date(), "jet_a_usd_per_gal": max(1.50, price),
                     "data_source": "synthetic"})
    return pd.DataFrame(rows)


def main():
    # 24 months: Jan 2023 – Dec 2024
    periods = pd.date_range("2023-01-01", periods=24, freq="MS").tolist()

    fa_routes    = _load_ual_routes()
    extra_routes = _extra_routes()

    # Merge, deduplicate on (orig, dest, ac_type)
    all_routes = fa_routes.copy()
    existing   = {(r["origin_iata"], r["destination_iata"], r["aircraft_type_bts"])
                  for r in fa_routes}
    for r in extra_routes:
        k = (r["origin_iata"], r["destination_iata"], r["aircraft_type_bts"])
        if k not in existing:
            all_routes.append(r)
            existing.add(k)

    print(f"Building synthetic demand for {len(all_routes)} routes × {len(periods)} periods …")
    demand_df    = build_demand(all_routes, periods)
    fuel_df      = build_fuel_prices(periods)

    demand_out = _P2_OUTPUTS / "route_demand_summary.csv"
    fuel_out   = _P1_DATA    / "fuel_prices.csv"

    demand_df.to_csv(demand_out, index=False)
    fuel_df.to_csv(fuel_out, index=False)

    print(f"Saved {len(demand_df)} demand rows → {demand_out}")
    print(f"Saved {len(fuel_df)} fuel price rows → {fuel_out}")
    print(f"\nSample demand rows:")
    print(demand_df.head(5)[["report_period","origin_iata","destination_iata",
                               "aircraft_variant","departures","load_factor","asm"]].to_string(index=False))
    print(f"\nFuel price range: ${fuel_df['jet_a_usd_per_gal'].min():.2f} – "
          f"${fuel_df['jet_a_usd_per_gal'].max():.2f}/gal")


if __name__ == "__main__":
    main()
