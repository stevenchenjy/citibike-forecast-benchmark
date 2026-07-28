import numpy as np

from citibike_benchmark.evaluation.metrics import forecast_metrics


def test_metrics_on_hand_worked_example() -> None:
    metrics = forecast_metrics(np.array([1, 3]), np.array([2, 1]))
    assert metrics["mae"] == 1.5
    assert metrics["rmse"] == np.sqrt(2.5)
    assert metrics["wape"] == 0.75
    assert metrics["mean_signed_error"] == -0.5
