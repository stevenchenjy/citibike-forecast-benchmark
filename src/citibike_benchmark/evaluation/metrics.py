"""Forecast metric calculations."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_poisson_deviance, r2_score


def forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0.0)
    error = predicted - actual
    absolute = np.abs(error)
    safe_positive = np.maximum(predicted, 1e-12)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(absolute.sum() / actual.sum()) if actual.sum() else float("nan"),
        "mean_poisson_deviance": float(mean_poisson_deviance(actual, safe_positive)),
        "r_squared": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
        "mean_signed_error": float(error.mean()),
        "negative_predictions_clipped": int((np.asarray(predicted) < 0).sum()),
    }
