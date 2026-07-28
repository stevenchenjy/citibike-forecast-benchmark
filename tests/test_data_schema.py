from pathlib import Path

import pandas as pd
import pytest

from citibike_benchmark.data.schema import require_canonical_columns, validate_canonical_dtypes
from citibike_benchmark.data.source_adapter import validate_source_schema


def test_canonical_schema_requires_all_columns() -> None:
    with pytest.raises(ValueError, match="missing required"):
        require_canonical_columns(pd.DataFrame({"station_id": ["x"]}))


def test_canonical_schema_accepts_timezone_aware_nonnegative_counts() -> None:
    frame = pd.DataFrame({
        "station_id": ["x"], "timestamp": pd.to_datetime(["2018-01-01T00:00:00Z"]),
        "date": ["2017-12-31"], "hour": [19], "day_of_week": [6], "is_weekend": [True],
        "pickups": [0], "returns": [1], "net_flow": [1], "station_capacity": [20], "data_complete": [True],
    })
    validate_canonical_dtypes(frame)


def test_official_source_schema_parses_after_acquisition() -> None:
    source = Path("data/external/variational-poisson-rnn")
    if not source.exists():
        pytest.skip("Official source is acquired by `make data`, not versioned in the benchmark repository")
    demand_paths = validate_source_schema(source)
    assert len(demand_paths) == 30
    assert "128" in demand_paths
