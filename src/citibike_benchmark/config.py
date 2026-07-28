"""Configuration loading with small, explicit YAML inheritance."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from citibike_benchmark.constants import PROJECT_ROOT


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("extends", None)
    if parent:
        return _merge(load_config(config_path.parent / parent), config)
    return config
