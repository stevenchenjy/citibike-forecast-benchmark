"""Paired day-level bootstrap support."""
from __future__ import annotations

import numpy as np


def paired_mean_difference_ci(differences: np.ndarray, seed: int, replicates: int = 300) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if not len(values):
        return (float("nan"), float("nan"))
    generator = np.random.default_rng(seed)
    draws = generator.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return tuple(float(v) for v in np.quantile(draws, [0.025, 0.975]))
