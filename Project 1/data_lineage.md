# Project 1 — ETL Data Lineage

## Pipeline overview

```
FlightAware AeroAPI          EIA OpenData API          BTS Bulk Downloads
       │                           │                          │
       ▼                           ▼                          ▼
extract_flightaware.py    extract_eia_fuel.py         extract_bts.py
       │                           │                          │
       └──────────┬────────────────┘            ┌────────────┘
                  │                             │
                  ▼                             ▼
            transform.py ←── UAL_aircraft_data.json
                  │
                  ▼
              load.py
                  │
       ┌──────────┼──────────┬──────────┬──────────┬──────────┐
       ▼          ▼          ▼          ▼          ▼          ▼
   dim_time   flights  flight_  fuel_   bts_t100_  bts_db1b_
              _capacity prices  segments   fares
```

---

## Source → Target field mappings

### FlightAware → flights

| FlightAware field | Warehouse column | Transformation |
|-------------------|-----------------|----------------|
| `fa_flight_id` | `flight_id` | Rename |
| `ident` | `ident` | Direct |
| `ident_iata` | `ident_iata` | Direct |
| `operator_icao` | `operator_icao` | Direct |
| `operator_iata` | `operator_iata` | Direct |
| `flight_number` | `flight_number` | Cast to INTEGER |
| `registration` | `registration` | Direct |
| `aircraft_type` | `aircraft_type` | Direct (raw ICAO type code) |
| `aircraft_type` | `aircraft_variant` | `AIRCRAFT_TYPE_MAP` lookup (config.py) |
| `origin.code_iata` | `origin_iata` | Extract from dict / stringified dict |
| `destination.code_iata` | `destination_iata` | Extract from dict / stringified dict |
| `route_distance` | `route_distance_mi` | Rename |
| `scheduled_out/off/on/in` | Same names | Parse ISO-8601 → UTC datetime |
| `actual_out/off/on/in` | Same names | Parse ISO-8601 → UTC datetime |
| — | `block_time_min` | PostgreSQL generated column: `(actual_on − actual_off) / 60` |
| — | `flight_date` | PostgreSQL generated column: `COALESCE(actual_off, scheduled_off)::DATE` |
| `departure_delay` | `departure_delay_min` | Rename |
| `arrival_delay` | `arrival_delay_min` | Rename |
| `status` | `status` | Direct |
| `cancelled` | `cancelled` | Cast bool |
| `diverted` | `diverted` | Cast bool |
| `seats_cabin_first` | `fa_seats_first` | Rename |
| `seats_cabin_business` | `fa_seats_business` | Rename |
| `seats_cabin_coach` | `fa_seats_coach` | Rename |

---

### UAL_aircraft_data.json → aircraft_configs

| JSON field | Warehouse column | Transformation |
|------------|-----------------|----------------|
| Top-level key | `aircraft_variant` | Direct |
| Second-level key | `cabin_class` | Direct |
| `Number of seats` | `seat_count` | Cast to INTEGER |
| `Seat configuration` | `seat_configuration` | Direct |
| `Standard seat pitch` | `seat_pitch_in` | Regex extract first number |
| `Seat width` | `seat_width_in` | Regex extract first number |
| `Wi-Fi` | `has_wifi` | Coerce to BOOLEAN |
| `Power outlets` | `has_power` | Coerce to BOOLEAN |
| — | `cabin_tier` | `CABIN_TIER_MAP` lookup (config.py) |

---

### EIA API → fuel_prices

| EIA field | Warehouse column | Transformation |
|-----------|-----------------|----------------|
| `period` | `price_date` | Parse to DATE |
| `value` | `jet_a_usd_per_gal` | Cast to NUMERIC |

**Match to flights:** `v_fuel_matched_flights` uses a LATERAL join that selects the most-recent `price_date ≤ flight_date` (backward ASOF join).

---

### BTS T-100 → bts_t100_segments

| T-100 column | Warehouse column | Transformation |
|-------------|-----------------|----------------|
| `UNIQUE_CARRIER` | `carrier_code` | Direct |
| `UNIQUE_CARRIER_NAME` | `carrier_name` | Direct |
| `AIRCRAFT_TYPE` | `aircraft_type_bts` | Direct (BTS codes differ from ICAO) |
| `ORIGIN` | `origin_iata` | Direct |
| `DEST` | `destination_iata` | Direct |
| `YEAR` + `MONTH` | `report_period` | Concat → `YYYY-MM-01` DATE |
| `DEPARTURES_PERFORMED` | `departures_performed` | Cast INTEGER |
| `SEATS` | `seats_available` | Cast INTEGER |
| `PASSENGERS` | `passengers` | Cast INTEGER |
| `PAYLOAD` | `payload_lbs` | Cast NUMERIC |
| `DISTANCE` | `distance_mi` | Cast INTEGER |
| — | `load_factor` | PostgreSQL generated: `passengers / seats_available` |
| — | `asm` | PostgreSQL generated: `seats_available × distance_mi` |
| — | `rpm` | PostgreSQL generated: `passengers × distance_mi` |

---

### BTS DB1B → bts_db1b_fares

| DB1B column | Warehouse column | Transformation |
|-------------|-----------------|----------------|
| `REPORTING_CARRIER` | `carrier_code` | Direct |
| `YEAR` + `QUARTER` | `report_quarter` | Map quarter → first month, DATE |
| `ORIGIN` | `origin_iata` | Direct |
| `DEST` | `destination_iata` | Direct |
| `CABIN_CLASS` | `cabin_class_bts` | Direct |
| `MARKET_FARE` | `avg_fare_usd` | Aggregated: mean over grouping |
| `PASSENGERS` | `passengers` | Aggregated: sum |
| `MARKET_MILES_FLOWN` | `miles` | Aggregated: mean |
| — | `yield_per_mile` | PostgreSQL generated: `avg_fare_usd / miles` |

---

## Aircraft type code crosswalk

FlightAware ICAO type codes (e.g. `B77W`) differ from both ICAO type certificates and
BTS equipment codes.  The canonical mapping lives in `config.AIRCRAFT_TYPE_MAP`:

| FA Code | United Variant | BTS Equiv. |
|---------|---------------|-----------|
| B77W | B777-300ER | 77W |
| B772 | B777-200 | 772 |
| B77L | B777-200 | 77L |
| B788 | B787-8 | 788 |
| B789 | B787-9 Version 1 | 789 |
| B78X | B787-10 | 78X |
| B763 | 767-300ER Version 1 | 763 |
| B764 | 767-400ER | 764 |
| B752 | 757-200 | 752 |
| B753 | 757-300 | 753 |
| B737 | 737-700 | 737 |
| B738 | 737-800 Version 1 | 738 |
| B739 | 737-900 Version 1 | 739 |
| B38M | 737 MAX 8 Version 1 | 7M8 |
| B39M | 737 MAX 9 Version 1 | 7M9 |
| A319 | A319 | 319 |
| A320 | A320 | 320 |
| A21N | A321neo | 32Q |
| CRJ2 | CRJ200 | CRJ |
| CL65 | CRJ550 | CR5 |
| CRJ7 | CRJ700 | CR7 |
| E170 | Embraer E170 | E70 |
| E75L | Embraer E175 Version 1 | E75 |

---

## Data refresh schedule (recommended)

| Source | Frequency | Script | Notes |
|--------|-----------|--------|-------|
| FlightAware flights | Daily | `pipeline.py --live` | Run after midnight UTC |
| EIA fuel prices | Weekly (Mondays) | `pipeline.py --fuel-only` | EIA publishes Mondays |
| BTS T-100 | Monthly | `pipeline.py --bts` | ~2 month lag; download manually |
| BTS DB1B | Quarterly | `pipeline.py --bts` | ~6 month lag; download manually |
| Aircraft configs | On-demand | `pipeline.py --init-db` | Re-run when fleet changes |
