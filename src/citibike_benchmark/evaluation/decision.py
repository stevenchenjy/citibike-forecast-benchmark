"""Transparent day-ahead station inventory decisions and oracle comparison."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from citibike_benchmark.config import load_config
from citibike_benchmark.constants import PROJECT_ROOT
from citibike_benchmark.evaluation.backtest import load_panel
from citibike_benchmark.evaluation.inventory import simulate_inventory
from citibike_benchmark.evaluation.oracle import select_starting_inventory
from citibike_benchmark.utils.io import sha256_file


def run_decision_evaluation(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    run_id = f"{config['run']['name']}_{sha256_file(config_path)[:12]}"
    prediction_path = PROJECT_ROOT / "artifacts/predictions" / f"{run_id}.parquet"
    if not prediction_path.exists():
        raise RuntimeError("Forecast predictions are missing; run backtest first")
    predictions = pd.read_parquet(prediction_path)
    predictions = predictions[predictions["track"].eq("day_ahead")].copy()
    predictions["date"] = pd.to_datetime(predictions["timestamp"]).dt.date
    panel = load_panel(config)
    capacities = panel.groupby("station_id")["station_capacity"].first().to_dict()
    rows: list[dict[str, Any]] = []
    for ordering in (config["evaluation"]["inventory_ordering"], config["evaluation"]["inventory_ordering_sensitivity"]):
        for keys, group in predictions.groupby(["model", "fold", "station_id", "date"], sort=True):
            model, fold, station_id, date = keys
            path = group.pivot(index="horizon_step", columns="target_type", values=["actual", "prediction"]).sort_index()
            if len(path) != 24 or not {("actual", "pickups"), ("actual", "returns"), ("prediction", "pickups"), ("prediction", "returns")}.issubset(path.columns):
                continue
            actual_pickups = path[("actual", "pickups")].round().astype(int).tolist()
            actual_returns = path[("actual", "returns")].round().astype(int).tolist()
            predicted_pickups = path[("prediction", "pickups")].clip(lower=0).round().astype(int).tolist()
            predicted_returns = path[("prediction", "returns")].clip(lower=0).round().astype(int).tolist()
            capacity = int(capacities[str(station_id)])
            selected, _ = select_starting_inventory(capacity, predicted_pickups, predicted_returns, ordering)
            realized = simulate_inventory(selected, capacity, actual_pickups, actual_returns, ordering)
            oracle_selected, oracle = select_starting_inventory(capacity, actual_pickups, actual_returns, ordering)
            rows.append({
                "model": model, "fold": fold, "station_id": station_id, "date": date, "ordering": ordering,
                "capacity": capacity, "selected_starting_inventory": selected, "oracle_starting_inventory": oracle_selected,
                "selected_starting_inventory_error": selected - oracle_selected,
                "failed_pickups": realized.failed_pickups, "failed_returns": realized.failed_returns,
                "total_failures": realized.total_failures, "fulfilled_demand_rate": realized.service_level,
                "oracle_total_failures": oracle.total_failures, "regret_vs_oracle": realized.total_failures - oracle.total_failures,
                "within_one_failure_of_oracle": int(realized.total_failures <= oracle.total_failures + 1),
            })
    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("No complete 24-hour day-ahead prediction paths available for decisions")
    metrics = detail.groupby(["model", "fold", "ordering"], as_index=False).agg(
        failed_pickups=("failed_pickups", "sum"), failed_returns=("failed_returns", "sum"), total_failures=("total_failures", "sum"),
        fulfilled_demand_rate=("fulfilled_demand_rate", "mean"), average_regret_vs_oracle=("regret_vs_oracle", "mean"),
        percentage_within_one_failure_of_oracle=("within_one_failure_of_oracle", "mean"),
        mean_absolute_starting_inventory_error=("selected_starting_inventory_error", lambda x: x.abs().mean()), station_days=("date", "size"),
    )
    metrics["percentage_within_one_failure_of_oracle"] *= 100
    tables = PROJECT_ROOT / "reports/tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "decision_metrics.csv", index=False)
    detail.to_csv(PROJECT_ROOT / "reports/runs" / f"{run_id}_decision_detail.csv", index=False)
    return {"run_id": run_id, "station_days": len(detail), "decision_metrics": str((tables / "decision_metrics.csv").relative_to(PROJECT_ROOT))}
