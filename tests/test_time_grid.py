import pandas as pd

from citibike_benchmark.data.validate import duplicate_station_timestamps


def test_duplicate_station_timestamp_is_detected() -> None:
    timestamp = pd.Timestamp("2018-01-01", tz="America/New_York")
    frame = pd.DataFrame({"station_id": ["a", "a"], "timestamp": [timestamp, timestamp]})
    assert duplicate_station_timestamps(frame) == 1
