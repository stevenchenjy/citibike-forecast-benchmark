from pathlib import Path

import pandas as pd
import pytest

from citibike_benchmark.constants import TIMEZONE
from citibike_benchmark.data.panel import normalize_source_civil_hours
from citibike_benchmark.data.validate import duplicate_station_timestamps


def test_duplicate_station_timestamp_is_detected() -> None:
    timestamp = pd.Timestamp("2018-01-01", tz="America/New_York")
    frame = pd.DataFrame({"station_id": ["a", "a"], "timestamp": [timestamp, timestamp]})
    assert duplicate_station_timestamps(frame) == 1


def test_dst_normalization_drops_nonexistent_and_flags_ambiguous_hours() -> None:
    source = pd.DataFrame({
        "station_id": ["x"] * 6,
        "date": pd.to_datetime(["2018-03-11"] * 3 + ["2018-11-04"] * 3),
        "hour": [1, 2, 3, 0, 1, 2],
        "pickups": [2, 0, 4, 1, 6, 3],
        "returns": [1, 0, 5, 2, 7, 4],
        "station_capacity": [20] * 6,
    })
    result = normalize_source_civil_hours(source)
    assert len(result) == 6
    assert not result.duplicated(["station_id", "timestamp"]).any()
    assert not ((result["date"].astype(str) == "2018-03-11") & (result["hour"] == 2)).any()
    fall_hour = result[(result["date"].astype(str) == "2018-11-04") & (result["hour"] == 1)]
    assert len(fall_hour) == 2
    assert not fall_hour["data_complete"].any()
    assert sorted(fall_hour["pickups"].tolist()) == [0, 6]


def test_official_station_maps_to_complete_actual_hour_grid() -> None:
    source = Path("data/external/variational-poisson-rnn/data/demand_rate/60min/128_hourlyRatesByDay_2018.csv")
    if not source.exists():
        pytest.skip("Official source is acquired by `make data`, not versioned in the benchmark repository")
    demand = pd.read_csv(source, parse_dates=["date"])
    frame = demand.rename(columns={"count_pickup": "pickups", "count_return": "returns"})
    frame["station_id"] = "128"
    frame["station_capacity"] = 50
    result = normalize_source_civil_hours(frame)
    expected = pd.date_range("2018-01-01", "2019-01-01", freq="h", tz=TIMEZONE, inclusive="left")
    assert len(result) == 8760
    assert pd.DatetimeIndex(result["timestamp"]).equals(expected)
    assert (~result["data_complete"]).sum() == 2
