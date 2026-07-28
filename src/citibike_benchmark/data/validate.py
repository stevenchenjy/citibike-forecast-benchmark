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
