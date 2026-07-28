# Source Data Audit

Audit timestamp: `2026-07-28T18:20:46.304979+00:00`<br>
Source: `https://github.com/DanieleGammelli/variational-poisson-rnn.git`<br>
Pinned source commit: `abf77f79fc64be75ae9102ec8d537f77ed9c5f8f`<br>
Download date: `2026-07-28T18:18:08.083629+00:00`

## Source and schema

The audit inspected the official VP-RNN repository before implementing mappings. It found 30 60-minute station demand-rate files and their matching raw event files. The field mapping is:

- `station_id`: demand-rate filename prefix
- `pickups`: `count_pickup`, verified as raw `inventory_change == -1`
- `returns`: `count_return`, verified as raw `inventory_change == +1`
- `net_flow`: `returns - pickups`
- `station_capacity`: `station_information_citibike.capacity`
- `data_complete`: a present source grid row with both target values present

Reference station IDs (30): `128, 151, 168, 229, 285, 293, 327, 358, 359, 368, 387, 402, 405, 426, 435, 445, 446, 453, 462, 482, 491, 497, 499, 504, 514, 519, 3263, 3435, 3641, 3711`.

Demand schema: `date, hour, count_pickup, count_return, day_of_week, historical_average_pickup, historical_average_return, monthly_historical_average_pickup, monthly_historical_average_return`.<br>
Raw booking schema: `event_time, inventory_change, date, hour, minute, event_order`.<br>
Weather 60-minute schema is present but excluded from the main comparison.

## Quality checks

| Check | Result |
| --- | ---: |
| Demand rows | 262,800 |
| Demand duplicate station-hours | 0 |
| Missing source date/hour grid rows | 0 |
| Missing pickup/return values | 0 |
| Negative pickup/return values | 0 |
| Raw events outside the 2018 demand range | 3 |
| Raw-hourly versus demand count mismatch rows | 0 |
| Reference station capacities missing | 0 |
| 60-minute weather duplicate date-hours | 0 |
| 60-minute weather missing values | 0 |

All 30 demand files cover `2018-01-01` through `2018-12-31` with 8,760 rows each. Raw-event aggregation matched every hourly pickup and return count for every reference station **within the 2018 demand range**. Three raw positive events are timestamped in 2019 (one each in the 285, 402, and 497 files); they are outside the named 2018 source scope and are excluded from the canonical 2018 panel and any exact-event decision replay.

## DST handling limitation

The source supplies exactly 24 unzoned civil-hour labels for every 2018 date, including the DST spring-forward and fall-back dates. It cannot identify the repeated fall 01:00 hour or represent the nonexistent spring 02:00 as an actual America/New_York instant. The panel builder uses a documented explicit conversion and marks affected rows incomplete rather than silently treating the labels as unambiguous.

## License notes

The upstream repository LICENSE is MIT (Copyright 2021 Daniele Gammelli). No distinct data-license statement was found in the root documentation; retain upstream terms and validate suitability before redistribution.

The machine-readable audit and hashes are in `data/manifests/source_manifest.json` (intentionally gitignored because it is a generated, hash-heavy run artifact).
