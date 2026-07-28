"""Station metadata feature helpers."""
from __future__ import annotations

import pandas as pd


def add_station_capacity_feature(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["station_capacity_valid"] = result["station_capacity"].notna().astype(int)
    return result
