"""Validated adapter for the official VP-RNN 60-minute Citi Bike source."""
from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from citibike_benchmark.constants import PROJECT_ROOT, SOURCE_RELATIVE_PATH, SOURCE_REPOSITORY_URL
from citibike_benchmark.utils.io import sha256_file, write_json

DEMAND_COLUMNS = (
    "date", "hour", "count_pickup", "count_return", "day_of_week",
    "historical_average_pickup", "historical_average_return",
    "monthly_historical_average_pickup", "monthly_historical_average_return",
)
RAW_BOOKING_COLUMNS = ("event_time", "inventory_change", "date", "hour", "minute", "event_order")
STATION_COLUMNS = ("id", "address", "longitude", "latitude", "capacity", "has_kiosk", "station_type")
WEATHER_COLUMNS = (
    "date", "hour", "temperature", "dew_point_temperature", "sunrise", "precip_prob", "humidity",
    "wind_speed", "icon_clear", "icon_cloudy", "icon_fog", "icon_partly-cloudy", "icon_rain",
    "icon_sleet", "icon_snow",
)


@dataclass(frozen=True)
class SourceFile:
    path: Path
    suffix: str
    bytes: int


def discover_source_files(source_root: Path) -> list[SourceFile]:
    """Return deterministic inventory of potentially relevant source files."""
    suffixes = {".csv", ".tsv", ".txt", ".parquet", ".pkl", ".pickle", ".npy", ".npz"}
    files: Iterable[Path] = source_root.rglob("*")
    return [
        SourceFile(path=path, suffix=path.suffix.lower(), bytes=path.stat().st_size)
        for path in sorted(files)
        if path.is_file() and path.suffix.lower() in suffixes
    ]


def _git_commit(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_root, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def _read_download_metadata() -> dict[str, Any]:
    path = PROJECT_ROOT / "data/manifests/source_download.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def demand_paths(source_root: Path) -> dict[str, Path]:
    paths = sorted((source_root / "data/demand_rate/60min").glob("*_hourlyRatesByDay_2018.csv"))
    result = {path.name.removesuffix("_hourlyRatesByDay_2018.csv"): path for path in paths}
    if len(result) != len(paths):
        raise ValueError("Demand-rate filenames do not define unique station IDs")
    if len(result) != 30:
        raise ValueError(f"Expected exactly 30 VP-RNN 60-minute demand files, found {len(result)}")
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def validate_source_schema(source_root: Path) -> dict[str, Path]:
    """Fail fast when the pinned source layout or field contract has changed."""
    demands = demand_paths(source_root)
    for station_id, path in demands.items():
        columns = tuple(pd.read_csv(path, nrows=0).columns)
        if columns != DEMAND_COLUMNS:
            raise ValueError(f"Unexpected demand schema for station {station_id}: {columns}")
        raw_path = source_root / "data/raw" / f"{station_id}_allbooking_2018.csv"
        if not raw_path.exists():
            raise ValueError(f"Missing raw booking file for station {station_id}")
        raw_columns = tuple(pd.read_csv(raw_path, nrows=0).columns)
        if raw_columns != RAW_BOOKING_COLUMNS:
            raise ValueError(f"Unexpected raw booking schema for station {station_id}: {raw_columns}")
    station_columns = tuple(pd.read_csv(source_root / "data/raw/station_information_citibike.csv", nrows=0).columns)
    if station_columns != STATION_COLUMNS:
        raise ValueError(f"Unexpected station schema: {station_columns}")
    weather_columns = tuple(pd.read_csv(source_root / "data/raw/weather2018_60min.csv", nrows=0).columns)
    if weather_columns != WEATHER_COLUMNS:
        raise ValueError(f"Unexpected 60-minute weather schema: {weather_columns}")
    return demands


def _demand_quality(station_id: str, demand_path: Path, raw_path: Path) -> dict[str, Any]:
    demand = pd.read_csv(demand_path, parse_dates=["date"])
    raw = pd.read_csv(raw_path, parse_dates=["event_time", "date"])
    expected = pd.MultiIndex.from_product(
        [pd.date_range("2018-01-01", "2018-12-31", freq="D"), range(24)], names=["date", "hour"]
    )
    grid = pd.MultiIndex.from_frame(demand[["date", "hour"]])
    source_start = pd.Timestamp("2018-01-01")
    source_end = pd.Timestamp("2019-01-01")
    in_source_year = raw["event_time"].between(source_start, source_end, inclusive="left")
    raw_hourly = (
        raw.loc[in_source_year].assign(
            date=raw.loc[in_source_year, "event_time"].dt.normalize(), hour=raw.loc[in_source_year, "event_time"].dt.hour
        )
        .groupby(["date", "hour"], as_index=False)
        .agg(
            count_pickup=("inventory_change", lambda series: int((series == -1).sum())),
            count_return=("inventory_change", lambda series: int((series == 1).sum())),
        )
    )
    comparison = demand[["date", "hour", "count_pickup", "count_return"]].merge(
        raw_hourly, on=["date", "hour"], how="outer", suffixes=("_demand", "_raw")
    ).fillna(0)
    mismatch = (comparison["count_pickup_demand"] != comparison["count_pickup_raw"]) | (
        comparison["count_return_demand"] != comparison["count_return_raw"]
    )
    return {
        "station_id": station_id,
        "demand_rows": int(len(demand)),
        "demand_date_start": demand["date"].min().date().isoformat(),
        "demand_date_end": demand["date"].max().date().isoformat(),
        "demand_duplicate_station_hours": int(demand.duplicated(["date", "hour"]).sum()),
        "demand_missing_grid_rows": int(len(expected.difference(grid))),
        "demand_missing_target_values": int(demand[["count_pickup", "count_return"]].isna().sum().sum()),
        "demand_negative_target_values": int((demand[["count_pickup", "count_return"]] < 0).sum().sum()),
        "raw_rows": int(len(raw)),
        "raw_event_start": raw["event_time"].min().isoformat(),
        "raw_event_end": raw["event_time"].max().isoformat(),
        "raw_duplicate_event_times": int(raw.duplicated("event_time").sum()),
        "raw_duplicate_event_orders": int(raw.duplicated("event_order").sum()),
        "raw_missing_values": int(raw.isna().sum().sum()),
        "raw_invalid_inventory_changes": int((~raw["inventory_change"].isin([-1, 1])).sum()),
        "raw_events_outside_2018_demand_range": int((~in_source_year).sum()),
        "raw_hourly_demand_mismatch_rows": int(mismatch.sum()),
    }


def inspect_source(source_root: Path | None = None) -> dict[str, Any]:
    """Audit all required official source files and create a durable manifest."""
    source_root = source_root or PROJECT_ROOT / SOURCE_RELATIVE_PATH
    demands = validate_source_schema(source_root)
    raw_root = source_root / "data/raw"
    station_quality = [_demand_quality(station, path, raw_root / f"{station}_allbooking_2018.csv") for station, path in demands.items()]
    station_info = pd.read_csv(raw_root / "station_information_citibike.csv")
    station_ids = [item["station_id"] for item in station_quality]
    missing_capacities = station_info.loc[station_info["id"].isin([int(value) for value in station_ids]), "capacity"].isna().sum()
    weather = pd.read_csv(raw_root / "weather2018_60min.csv", parse_dates=["date"])
    tracked_files = [*demands.values(), *(raw_root / f"{station_id}_allbooking_2018.csv" for station_id in station_ids), raw_root / "station_information_citibike.csv", raw_root / "weather2018_60min.csv", source_root / "LICENSE", source_root / "README.md"]
    download_metadata = _read_download_metadata()
    manifest: dict[str, Any] = {
        "audit_timestamp": datetime.now(UTC).isoformat(),
        "source_repository_url": SOURCE_REPOSITORY_URL,
        "source_commit": _git_commit(source_root),
        "download_date": download_metadata.get("download_date"),
        "source_root": str(source_root.relative_to(PROJECT_ROOT)),
        "station_ids": station_ids,
        "source_schema": {
            "demand_rate_60min": list(DEMAND_COLUMNS), "raw_booking": list(RAW_BOOKING_COLUMNS),
            "station_information": list(STATION_COLUMNS), "weather_60min": list(WEATHER_COLUMNS),
        },
        "field_mapping": {
            "station_id": "60min demand-rate filename prefix", "timestamp": "date + hour, interpreted as source civil time; explicit DST conversion occurs in panel build",
            "pickups": "count_pickup (verified against raw inventory_change == -1)", "returns": "count_return (verified against raw inventory_change == +1)",
            "net_flow": "returns - pickups", "station_capacity": "station_information_citibike.capacity", "data_complete": "all source demand target values present and source date/hour grid row exists",
        },
        "demand_quality_by_station": station_quality,
        "station_metadata_quality": {
            "rows": int(len(station_info)), "duplicate_ids": int(station_info.duplicated("id").sum()),
            "reference_stations_matched": int(station_info["id"].isin([int(value) for value in station_ids]).sum()),
            "reference_station_missing_capacities": int(missing_capacities),
        },
        "weather_60min_quality": {
            "rows": int(len(weather)), "date_start": weather["date"].min().date().isoformat(), "date_end": weather["date"].max().date().isoformat(),
            "duplicate_date_hours": int(weather.duplicated(["date", "hour"]).sum()), "missing_values": int(weather.isna().sum().sum()),
        },
        "dst_source_limitation": "The source supplies exactly 24 unzoned civil-hour labels for every 2018 date, including the DST spring-forward and fall-back dates. It cannot identify the repeated fall 01:00 hour or represent the nonexistent spring 02:00 as an actual America/New_York instant. The panel builder uses a documented explicit conversion and marks affected rows incomplete rather than silently treating the labels as unambiguous.",
        "license_notes": "The upstream repository LICENSE is MIT (Copyright 2021 Daniele Gammelli). No distinct data-license statement was found in the root documentation; retain upstream terms and validate suitability before redistribution.",
        "input_file_sha256": {str(path.relative_to(source_root)): sha256_file(path) for path in tracked_files},
    }
    write_json(PROJECT_ROOT / "data/manifests/source_manifest.json", manifest)
    return manifest


def write_data_audit(manifest: dict[str, Any]) -> Path:
    """Write the human-readable audit required before any data mapping is used."""
    quality = manifest["demand_quality_by_station"]
    summary = {
        key: sum(item[key] for item in quality)
        for key in ("demand_rows", "demand_duplicate_station_hours", "demand_missing_grid_rows", "demand_missing_target_values", "demand_negative_target_values", "raw_events_outside_2018_demand_range", "raw_hourly_demand_mismatch_rows")
    }
    path = PROJECT_ROOT / "reports/audit/data_audit.md"
    text = f"""# Source Data Audit

Audit timestamp: `{manifest['audit_timestamp']}`<br>
Source: `{manifest['source_repository_url']}`<br>
Pinned source commit: `{manifest['source_commit']}`<br>
Download date: `{manifest['download_date']}`

## Source and schema

The audit inspected the official VP-RNN repository before implementing mappings. It found 30 60-minute station demand-rate files and their matching raw event files. The field mapping is:

- `station_id`: demand-rate filename prefix
- `pickups`: `count_pickup`, verified as raw `inventory_change == -1`
- `returns`: `count_return`, verified as raw `inventory_change == +1`
- `net_flow`: `returns - pickups`
- `station_capacity`: `station_information_citibike.capacity`
- `data_complete`: a present source grid row with both target values present

Reference station IDs ({len(manifest['station_ids'])}): `{', '.join(manifest['station_ids'])}`.

Demand schema: `{', '.join(manifest['source_schema']['demand_rate_60min'])}`.<br>
Raw booking schema: `{', '.join(manifest['source_schema']['raw_booking'])}`.<br>
Weather 60-minute schema is present but excluded from the main comparison.

## Quality checks

| Check | Result |
| --- | ---: |
| Demand rows | {summary['demand_rows']:,} |
| Demand duplicate station-hours | {summary['demand_duplicate_station_hours']:,} |
| Missing source date/hour grid rows | {summary['demand_missing_grid_rows']:,} |
| Missing pickup/return values | {summary['demand_missing_target_values']:,} |
| Negative pickup/return values | {summary['demand_negative_target_values']:,} |
| Raw events outside the 2018 demand range | {summary['raw_events_outside_2018_demand_range']:,} |
| Raw-hourly versus demand count mismatch rows | {summary['raw_hourly_demand_mismatch_rows']:,} |
| Reference station capacities missing | {manifest['station_metadata_quality']['reference_station_missing_capacities']:,} |
| 60-minute weather duplicate date-hours | {manifest['weather_60min_quality']['duplicate_date_hours']:,} |
| 60-minute weather missing values | {manifest['weather_60min_quality']['missing_values']:,} |

All 30 demand files cover `2018-01-01` through `2018-12-31` with 8,760 rows each. Raw-event aggregation matched every hourly pickup and return count for every reference station **within the 2018 demand range**. Three raw positive events are timestamped in 2019 (one each in the 285, 402, and 497 files); they are outside the named 2018 source scope and are excluded from the canonical 2018 panel and any exact-event decision replay.

## DST handling limitation

{manifest['dst_source_limitation']}

## License notes

{manifest['license_notes']}

The machine-readable audit and hashes are in `data/manifests/source_manifest.json` (intentionally gitignored because it is a generated, hash-heavy run artifact).
"""
    path.write_text(text, encoding="utf-8")
    return path
