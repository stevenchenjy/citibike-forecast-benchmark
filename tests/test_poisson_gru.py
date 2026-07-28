import numpy as np
import pandas as pd

from citibike_benchmark.config import load_config
from citibike_benchmark.models.poisson_gru import _sequence_arrays


def test_poisson_gru_config_is_capped_and_no_weather() -> None:
    config = load_config("configs/poisson_gru.yaml")
    settings = config["models"]["poisson_gru"]
    assert config["weather"]["enabled"] is False
    assert config["models"]["enabled"] == ["historical_average", "poisson_gru"]
    assert settings["max_epochs"] == 5
    assert settings["patience"] == 2
    assert settings["sequence_length"] == 24


def test_poisson_gru_history_ends_at_origin_not_target() -> None:
    timestamps = pd.date_range("2018-01-01", periods=5, freq="h", tz="America/New_York")
    panel = pd.DataFrame({
        "station_id": ["a"] * 5, "timestamp": timestamps,
        "pickups": [0, 1, 2, 3, 99], "returns": [0, 10, 20, 30, 99],
        "data_complete": [True, True, True, True, True],
    })
    examples = pd.DataFrame({
        "station_id": ["a"], "origin_sequence_index": [3],
        "target_hour": [4], "target_day_of_week": [0], "horizon_step": [1],
    })
    sequences, stations, context, ids = _sequence_arrays(examples, panel, sequence_length=4)
    np.testing.assert_allclose(sequences[0, :, 0], np.log1p([0, 1, 2, 3]))
    np.testing.assert_allclose(sequences[0, :, 1], np.log1p([0, 10, 20, 30]))
    assert sequences[0, :, 0].max() < np.log1p(99)
    assert stations.tolist() == [0]
    assert ids.tolist() == ["a"]
    assert context.shape == (1, 5)
