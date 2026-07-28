"""Canonical-panel schema contract."""
from __future__ import annotations

import pandas as pd

from citibike_benchmark.constants import CANONICAL_COLUMNS


def require_canonical_columns(frame: pd.DataFrame) -> None:
    missing = set(CANONICAL_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Canonical panel is missing required columns: {sorted(missing)}")


def validate_canonical_dtypes(frame: pd.DataFrame) -> None:
    require_canonical_columns(frame)
    if not isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype):
        raise ValueError("timestamp must be timezone-aware")
    for target in ("pickups", "returns"):
        if (frame[target] < 0).any():
            raise ValueError(f"{target} must be nonnegative")
