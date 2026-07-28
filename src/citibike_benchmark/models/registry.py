"""Explicit model registry; optional GRU is never included by default."""
from citibike_benchmark.constants import REQUIRED_MODELS


def required_model_names() -> tuple[str, ...]:
    return REQUIRED_MODELS
