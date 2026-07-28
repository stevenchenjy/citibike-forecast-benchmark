"""Station-level hourly inventory simulation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class InventoryOutcome:
    failed_pickups: int
    failed_returns: int
    fulfilled_demand: int
    total_demand: int
    ending_inventory: int

    @property
    def total_failures(self) -> int:
        return self.failed_pickups + self.failed_returns

    @property
    def service_level(self) -> float:
        return self.fulfilled_demand / self.total_demand if self.total_demand else 1.0


def simulate_inventory(
    starting_inventory: int,
    capacity: int,
    pickups: Iterable[int],
    returns: Iterable[int],
    ordering: str = "pickups_then_returns",
) -> InventoryOutcome:
    if not 0 <= starting_inventory <= capacity:
        raise ValueError("starting_inventory must be within station capacity")
    inventory = starting_inventory
    failed_pickups = failed_returns = fulfilled = total = 0
    for pickup_count, return_count in zip(pickups, returns, strict=True):
        pickup_count, return_count = max(0, int(pickup_count)), max(0, int(return_count))
        total += pickup_count + return_count
        if ordering == "pickups_then_returns":
            served_pickups = min(inventory, pickup_count)
            failed_pickups += pickup_count - served_pickups
            inventory -= served_pickups
            served_returns = min(capacity - inventory, return_count)
            failed_returns += return_count - served_returns
            inventory += served_returns
        elif ordering == "returns_then_pickups":
            served_returns = min(capacity - inventory, return_count)
            failed_returns += return_count - served_returns
            inventory += served_returns
            served_pickups = min(inventory, pickup_count)
            failed_pickups += pickup_count - served_pickups
            inventory -= served_pickups
        else:
            raise ValueError(f"Unsupported ordering: {ordering}")
        fulfilled += served_pickups + served_returns
    return InventoryOutcome(failed_pickups, failed_returns, fulfilled, total, inventory)
