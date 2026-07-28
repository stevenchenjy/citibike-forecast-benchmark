"""Calendar features available at every forecast origin."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    timestamp = result["timestamp"]
    result["hour_sin"] = np.sin(2 * np.pi * timestamp.dt.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * timestamp.dt.hour / 24)
    result["day_of_year_sin"] = np.sin(2 * np.pi * timestamp.dt.dayofyear / 366)
    result["day_of_year_cos"] = np.cos(2 * np.pi * timestamp.dt.dayofyear / 366)
    return result
