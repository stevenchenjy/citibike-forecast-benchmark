"""Starting-inventory oracle for station-day evaluation."""
from __future__ import annotations

from typing import Iterable

from citibike_benchmark.evaluation.inventory import InventoryOutcome, simulate_inventory


def select_starting_inventory(
    capacity: int, pickups: Iterable[int], returns: Iterable[int], ordering: str = "pickups_then_returns"
) -> tuple[int, InventoryOutcome]:
    pickups, returns = list(pickups), list(returns)
    candidates = [
        (initial, simulate_inventory(initial, capacity, pickups, returns, ordering))
        for initial in range(capacity + 1)
    ]
    return min(candidates, key=lambda item: (item[1].total_failures, abs(item[0] - capacity / 2), item[0]))
