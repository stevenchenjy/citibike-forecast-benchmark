"""Compact reproducible smoke/core report generation from saved tables."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from citibike_benchmark.config import load_config
from citibike_benchmark.constants import PROJECT_ROOT
from citibike_benchmark.utils.io import sha256_file


def build_report(config_path: str | Path) -> dict[str, str]:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    run_id = f"{config['run']['name']}_{sha256_file(config_path)[:12]}"
    tables = PROJECT_ROOT / "reports/tables"
    metrics_path, decisions_path = tables / "forecast_metrics.csv", tables / "decision_metrics.csv"
    if not metrics_path.exists() or not decisions_path.exists():
        raise RuntimeError("Forecast and decision tables are required before report generation")
    metrics, decisions = pd.read_csv(metrics_path), pd.read_csv(decisions_path)
    figures = PROJECT_ROOT / "reports/figures"
    figures.mkdir(parents=True, exist_ok=True)
    mae = metrics.groupby(["model", "track"], as_index=False).agg(mae=("mae", "mean"))
    pivot = mae.pivot(index="model", columns="track", values="mae").sort_index()
    ax = pivot.plot.bar(figsize=(9, 4), rot=30, ylabel="Mean MAE")
    ax.set_title("Smoke forecast MAE by model and horizon track")
    plt.tight_layout()
    mae_figure = figures / f"{run_id}_mae.png"
    plt.savefig(mae_figure, dpi=150)
    plt.close()
    decision = decisions[decisions["ordering"].eq("pickups_then_returns")].set_index("model").sort_index()
    axes = decision[["total_failures", "average_regret_vs_oracle"]].plot.bar(subplots=True, figsize=(9, 6), legend=False, rot=30)
    axes[0].set_title("Smoke inventory failures")
    axes[1].set_title("Average regret versus oracle")
    plt.tight_layout()
    decision_figure = figures / f"{run_id}_decisions.png"
    plt.savefig(decision_figure, dpi=150)
    plt.close()
    report = PROJECT_ROOT / "reports/final_report.md"
    overall = mae.groupby("model", as_index=False)["mae"].mean().sort_values("mae")
    report.write_text(
        "# Citi Bike Forecast Benchmark Report\n\n"
        f"## {config['run']['name']} run\n\n"
        "This is a smoke-scale, no-weather verification run, not a core finding. It uses five source-defined stations, 60 consecutive 2018 days, one strict chronological split (46 training, 7 validation, 7 test days), separate pickup and return targets, and both required horizon tracks. Weather is disabled.\n\n"
        f"The lowest average smoke MAE is **{overall.iloc[0]['model']}** ({overall.iloc[0]['mae']:.3f}); this result is only an integration check and must not be generalized to the 30-station core experiment.\n\n"
        "The inventory table uses day-ahead prediction paths, searches every feasible starting inventory, replays realized aggregate hourly demand, and compares failures with an oracle. The reversed within-hour ordering is reported as a sensitivity result, not as an operational claim.\n\n"
        "## Outputs\n\n"
        "- [Forecast metrics](tables/forecast_metrics.csv)\n- [Station metrics](tables/station_metrics.csv)\n- [Runtime metrics](tables/runtime_metrics.csv)\n- [Decision metrics](tables/decision_metrics.csv)\n"
        f"- [MAE figure](figures/{mae_figure.name})\n- [Decision figure](figures/{decision_figure.name})\n\n"
        "The future weather sensitivity must be named `observed_weather_hindsight_upper_bound` and is not run or interpreted here. Observed future weather would not represent forecast-vintage live-operational input.\n",
        encoding="utf-8",
    )
    return {"report": str(report.relative_to(PROJECT_ROOT)), "mae_figure": str(mae_figure.relative_to(PROJECT_ROOT)), "decision_figure": str(decision_figure.relative_to(PROJECT_ROOT))}
