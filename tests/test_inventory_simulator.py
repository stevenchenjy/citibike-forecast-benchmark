from citibike_benchmark.evaluation.inventory import simulate_inventory
from citibike_benchmark.evaluation.oracle import select_starting_inventory


def test_inventory_simulator_counts_pickup_and_return_failures() -> None:
    result = simulate_inventory(1, 2, [2], [3])
    assert result.failed_pickups == 1
    assert result.failed_returns == 1
    assert result.total_failures == 2
    assert result.service_level == 0.6


def test_oracle_search_is_feasible() -> None:
    initial, outcome = select_starting_inventory(3, [2], [1])
    assert 0 <= initial <= 3
    assert outcome.total_failures == 0
