"""Data quality checks shared by inspection and panel building."""
from __future__ import annotations

import pandas as pd


def duplicate_station_timestamps(frame: pd.DataFrame) -> int:
    return int(frame.duplicated(["station_id", "timestamp"]).sum())


def coverage_by_station(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("station_id", dropna=False)
    return grouped.agg(
        rows=("timestamp", "size"),
        missing_pickups=("pickups", lambda s: int(s.isna().sum())),
        missing_returns=("returns", lambda s: int(s.isna().sum())),
        complete_rows=("data_complete", lambda s: int(s.fillna(False).sum())),
    ).reset_index()


def assert_complete_hourly_grid(frame: pd.DataFrame, timezone: str) -> None:
    """Require every retained station to share an exact actual-hour time grid."""
    if frame.empty:
        raise ValueError("Canonical panel is empty")
    if duplicate_station_timestamps(frame):
        raise ValueError("Canonical panel has duplicate station timestamps")
    expected = pd.date_range(
        frame["timestamp"].min(), frame["timestamp"].max(), freq="h", tz=timezone
    )
    for station_id, station_frame in frame.groupby("station_id", sort=False):
        actual = pd.DatetimeIndex(station_frame.sort_values("timestamp")["timestamp"])
        if not actual.equals(expected):
            missing = len(expected.difference(actual))
            extra = len(actual.difference(expected))
            raise ValueError(f"Station {station_id} does not have the complete hourly grid (missing={missing}, extra={extra})")
