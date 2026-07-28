"""Build the canonical, complete America/New_York hourly station panel."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from citibike_benchmark.config import load_config
from citibike_benchmark.constants import CANONICAL_COLUMNS, PROJECT_ROOT, SOURCE_RELATIVE_PATH, TIMEZONE
from citibike_benchmark.data.source_adapter import demand_paths, validate_source_schema
from citibike_benchmark.data.validate import assert_complete_hourly_grid, coverage_by_station
from citibike_benchmark.utils.io import sha256_file, write_json

SPRING_FORWARD_DATE = pd.Timestamp("2018-03-11").date()
FALL_BACK_DATE = pd.Timestamp("2018-11-04").date()


def selected_station_ids(source_root: Path, station_count: int) -> list[str]:
    """Choose a deterministic source-defined subset without observing demand values."""
    available = list(demand_paths(source_root))
    if station_count > len(available):
        raise ValueError(f"Requested {station_count} stations but source provides only {len(available)}")
    return available[:station_count]


def normalize_source_civil_hours(source: pd.DataFrame, timezone: str = TIMEZONE) -> pd.DataFrame:
    """Convert source civil labels to an actual hourly NY grid with explicit DST flags.

    The source gives 24 unzoned labels for both 2018 DST transition dates. The
    spring 02:00 label is a genuine zero-demand but nonexistent local hour and
    is removed. The fall 01:00 source row cannot be assigned to either actual
    occurrence, so it is retained as the first occurrence and a synthetic
    second occurrence is added with zero counts; both are marked incomplete.
    Downstream model/evaluation code must exclude `data_complete == false`.
    """
    required = {"date", "hour", "pickups", "returns", "station_id", "station_capacity"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Cannot normalize source civil hours; missing {sorted(missing)}")
    result = source.copy()
    source_dates = pd.to_datetime(result["date"]).dt.date
    spring = (source_dates == SPRING_FORWARD_DATE) & result["hour"].eq(2)
    fall = (source_dates == FALL_BACK_DATE) & result["hour"].eq(1)
    if ((result.loc[spring, ["pickups", "returns"]] != 0).any(axis=None)):
        raise ValueError("Source has nonzero counts in nonexistent 2018-03-11 02:00; explicit reconciliation is required")
    regular = result.loc[~(spring | fall)].copy()
    naive = pd.to_datetime(regular["date"]) + pd.to_timedelta(regular["hour"], unit="h")
    regular["timestamp"] = naive.dt.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    regular["data_complete"] = True

    fall_observed = result.loc[fall].copy()
    fall_naive = pd.to_datetime(fall_observed["date"]) + pd.to_timedelta(fall_observed["hour"], unit="h")
    fall_observed["timestamp"] = fall_naive.dt.tz_localize(timezone, ambiguous=True)
    fall_observed["data_complete"] = False
    fall_synthetic = fall_observed.copy()
    fall_synthetic["timestamp"] = fall_naive.dt.tz_localize(timezone, ambiguous=False)
    fall_synthetic[["pickups", "returns"]] = 0
    fall_synthetic["data_complete"] = False

    combined = pd.concat([regular, fall_observed, fall_synthetic], ignore_index=True)
    combined["date"] = combined["timestamp"].dt.date
    combined["hour"] = combined["timestamp"].dt.hour.astype("int8")
    combined["day_of_week"] = combined["timestamp"].dt.dayofweek.astype("int8")
    combined["is_weekend"] = combined["day_of_week"].ge(5)
    combined["pickups"] = combined["pickups"].astype("int32")
    combined["returns"] = combined["returns"].astype("int32")
    combined["net_flow"] = (combined["returns"] - combined["pickups"]).astype("int32")
    combined["station_capacity"] = combined["station_capacity"].astype("int32")
    combined["data_complete"] = combined["data_complete"].astype(bool)
    return combined.loc[:, CANONICAL_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def _source_station_frame(source_root: Path, station_id: str, station_capacity: int) -> pd.DataFrame:
    path = demand_paths(source_root)[station_id]
    demand = pd.read_csv(path, parse_dates=["date"])
    expected_weekday = demand["date"].dt.dayofweek
    if not (demand["day_of_week"].to_numpy() == expected_weekday.to_numpy()).all():
        raise ValueError(f"Source weekday field conflicts with calendar for station {station_id}")
    frame = demand.loc[:, ["date", "hour", "count_pickup", "count_return"]].rename(
        columns={"count_pickup": "pickups", "count_return": "returns"}
    )
    frame["station_id"] = station_id
    frame["station_capacity"] = station_capacity
    return normalize_source_civil_hours(frame)


def _cache_is_valid(panel_path: Path, metadata_path: Path, station_ids: list[str], source_commit: str) -> bool:
    if not panel_path.exists() or not metadata_path.exists() or not list(panel_path.rglob("*.parquet")):
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return (
        metadata.get("station_ids") == station_ids
        and metadata.get("source_commit") == source_commit
    )


def _write_quality_tables(panel: pd.DataFrame, config_path: Path, run_name: str) -> None:
    """Keep profile-local quality tables from overwriting the core table."""
    quality = coverage_by_station(panel).assign(profile=run_name)
    run_id = f"{run_name}_{sha256_file(config_path)[:12]}"
    profile_path = PROJECT_ROOT / "reports/runs" / run_id / "data_quality.csv"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(profile_path, index=False)
    if run_name == "core_no_weather":
        canonical = PROJECT_ROOT / "reports/tables/data_quality.csv"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(profile_path, canonical)


def build_hourly_panel(config_path: str | Path) -> dict[str, Any]:
    """Build or safely reuse a partitioned parquet canonical panel."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    source_root = PROJECT_ROOT / SOURCE_RELATIVE_PATH
    validate_source_schema(source_root)
    source_manifest_path = PROJECT_ROOT / "data/manifests/source_manifest.json"
    if not source_manifest_path.exists():
        raise RuntimeError("Source audit manifest is missing; run `make inspect` before building the panel")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    station_ids = selected_station_ids(source_root, int(config["data"]["station_count"]))
    panel_path = PROJECT_ROOT / config["data"]["panel_path"]
    run_name = config["run"]["name"]
    metadata_path = PROJECT_ROOT / "data/manifests" / f"panel_{run_name}.json"
    if _cache_is_valid(panel_path, metadata_path, station_ids, source_manifest["source_commit"]):
        _write_quality_tables(pd.read_parquet(panel_path), config_path, run_name)
        return json.loads(metadata_path.read_text(encoding="utf-8")) | {"cached": True}
    if panel_path.exists() and list(panel_path.rglob("*.parquet")):
        raise RuntimeError(
            f"Panel path contains an incompatible cache: {panel_path}. Preserve it and configure a new panel_path instead of overwriting it."
        )

    station_info = pd.read_csv(source_root / "data/raw/station_information_citibike.csv").set_index("id")
    frames = []
    for station_id in station_ids:
        capacity = station_info.loc[int(station_id), "capacity"]
        if pd.isna(capacity) or int(capacity) <= 0:
            raise ValueError(f"Invalid capacity for station {station_id}: {capacity}")
        frames.append(_source_station_frame(source_root, station_id, int(capacity)))
    panel = pd.concat(frames, ignore_index=True)
    assert_complete_hourly_grid(panel, TIMEZONE)
    panel = panel.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    panel["year"] = panel["timestamp"].dt.year
    panel_path.mkdir(parents=True, exist_ok=True)
    # A profile-specific path and manifest make repeated calls cache-safe; no
    # existing data are removed or overwritten in place.
    panel.to_parquet(panel_path, index=False, partition_cols=["station_id", "year"])
    _write_quality_tables(panel, config_path, run_name)
    metadata = {
        "profile": run_name,
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": sha256_file(config_path),
        "source_commit": source_manifest["source_commit"],
        "station_ids": station_ids,
        "panel_path": str(panel_path.relative_to(PROJECT_ROOT)),
        "rows": int(len(panel)),
        "date_start": panel["timestamp"].min().isoformat(),
        "date_end": panel["timestamp"].max().isoformat(),
        "data_complete_rows": int(panel["data_complete"].sum()),
        "dst_incomplete_rows": int((~panel["data_complete"]).sum()),
        "dst_policy": "Dropped source spring-forward 02:00 zero rows; retained source fall 01:00 on first occurrence and synthesized zero second occurrence; both fall rows are data_complete=false.",
        "cached": False,
    }
    write_json(metadata_path, metadata)
    return metadata
