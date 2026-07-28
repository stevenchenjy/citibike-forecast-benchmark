"""Minimal common forecast-model protocol."""
from __future__ import annotations

from typing import Protocol

import pandas as pd


class ForecastModel(Protocol):
    name: str

    def fit(self, frame: pd.DataFrame) -> "ForecastModel": ...

    def predict(self, frame: pd.DataFrame) -> pd.Series: ...
