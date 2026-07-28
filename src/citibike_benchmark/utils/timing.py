"""Simple timing context manager."""
from __future__ import annotations

import time
from contextlib import contextmanager
from collections.abc import Iterator


@contextmanager
def elapsed_seconds() -> Iterator[dict[str, float]]:
    result: dict[str, float] = {}
    start = time.perf_counter()
    yield result
    result["seconds"] = time.perf_counter() - start
