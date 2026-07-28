"""Core report, tables, and required no-weather figures from saved artifacts."""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from citibike_benchmark.config import load_config
from citibike_benchmark.constants import PROJECT_ROOT
from citibike_benchmark.evaluation.metrics import forecast_metrics
from citibike_benchmark.utils.io import sha256_file


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _core_figures(run_id: str, predictions: pd.DataFrame, forecast: pd.DataFrame, station: pd.DataFrame, decision: pd.DataFrame, runtime: pd.DataFrame) -> list[Path]:
    figures = PROJECT_ROOT / "reports/figures"
    figures.mkdir(parents=True, exist_ok=True)
    output: list[Path] = []
    overall = forecast.query("slice_type == 'overall' and slice_value == 'all'").groupby(["model", "track"], as_index=False).mae.mean()
    pivot = overall.pivot(index="model", columns="track", values="mae").sort_index()
    output.append(_save(pivot.plot.bar(figsize=(9, 4), rot=25, ylabel="MAE", title="Overall MAE by model and horizon track").get_figure(), figures / f"{run_id}_overall_mae.png"))
    target = forecast.query("slice_type == 'overall' and slice_value == 'all'").groupby(["model", "target_type"], as_index=False).mae.mean()
    output.append(_save(target.pivot(index="model", columns="target_type", values="mae").sort_index().plot.bar(figsize=(9, 4), rot=25, ylabel="MAE", title="Pickup, return, and combined demand MAE").get_figure(), figures / f"{run_id}_targets.png"))
    peak = forecast.query("slice_type == 'period' and slice_value == 'peak'").groupby(["model", "track"], as_index=False).mae.mean()
    output.append(_save(peak.pivot(index="model", columns="track", values="mae").sort_index().plot.bar(figsize=(9, 4), rot=25, ylabel="Peak-period MAE").get_figure(), figures / f"{run_id}_peak_mae.png"))
    heat = station.query("target_type == 'combined'").groupby(["station_id", "model"], as_index=False).mae.mean().pivot(index="station_id", columns="model", values="mae").sort_index()
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(heat.columns)), heat.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set_title("Station-level combined-demand MAE")
    fig.colorbar(image, ax=ax, label="MAE")
    output.append(_save(fig, figures / f"{run_id}_station_heatmap.png"))
    operational = decision.query("ordering == 'pickups_then_returns'").groupby("model", as_index=True)[["total_failures", "average_regret_vs_oracle"]].sum().sort_index()
    decision_axes = operational.plot.bar(subplots=True, figsize=(9, 6), legend=False, rot=25, title=["Decision failures", "Oracle regret (sum of fold means)"])
    output.append(_save(decision_axes[0].get_figure(), figures / f"{run_id}_decisions.png"))
    runtime_plot = runtime.groupby("model", as_index=True).fit_seconds.sum().sort_values()
    output.append(_save(runtime_plot.plot.bar(figsize=(9, 4), ylabel="Fit seconds", title="Model fit time").get_figure(), figures / f"{run_id}_runtime.png"))
    day_ahead = predictions.query("track == 'day_ahead' and fold == 2 and model in ['lightgbm_poisson', 'historical_average']").copy()
    day_ahead["date"] = pd.to_datetime(day_ahead["timestamp"]).dt.date
    station_ids = sorted(day_ahead.station_id.astype(str).unique())
    representative = [station_ids[0], station_ids[len(station_ids) // 2], station_ids[-1]]
    for station_id in representative:
        subset = day_ahead[day_ahead.station_id.astype(str).eq(station_id)]
        date = sorted(subset.date.unique())[-1]
        rows = subset[subset.date.eq(date)]
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
        for axis, target_type in zip(axes, ("pickups", "returns"), strict=True):
            data = rows[rows.target_type.eq(target_type)]
            actual = data[data.model.eq("historical_average")]
            axis.plot(actual.horizon_step, actual.actual, color="black", label="actual")
            for model, color in (("historical_average", "tab:blue"), ("lightgbm_poisson", "tab:orange")):
                model_rows = data[data.model.eq(model)]
                axis.plot(model_rows.horizon_step, model_rows.prediction, label=model, color=color)
            axis.set_ylabel(target_type)
        axes[0].legend(ncol=3, fontsize=8)
        axes[1].set_xlabel("Day-ahead hour step")
        output.append(_save(fig, figures / f"{run_id}_station_{station_id}_forecast.png"))
    lgbm_path = PROJECT_ROOT / "artifacts/models" / run_id / "fold2_day_ahead_lightgbm_poisson_pickups.joblib"
    if lgbm_path.exists():
        model = joblib.load(lgbm_path)
        importance = pd.Series(model.feature_importances_, index=model.feature_name_).sort_values(ascending=False).head(20).sort_values()
        output.append(_save(importance.plot.barh(figsize=(8, 6), xlabel="Split importance", title="LightGBM pickup feature importance").get_figure(), figures / f"{run_id}_lightgbm_importance.png"))
    return output


def _core_report(run_id: str, figures: list[Path], predictions: pd.DataFrame, forecast: pd.DataFrame, bootstrap: pd.DataFrame, decision: pd.DataFrame) -> Path:
    report = PROJECT_ROOT / "reports/final_report.md"
    summary_rows = []
    for (model, track), group in predictions.groupby(["model", "track"], sort=True):
        metric = forecast_metrics(group.actual.to_numpy(), group.prediction.to_numpy())
        summary_rows.append({"model": model, "track": track, "mae": metric["mae"], "rmse": metric["rmse"], "wape": metric["wape"]})
    summary = pd.DataFrame(summary_rows).sort_values(["track", "mae"])
    baseline = summary[summary.model.eq("historical_average")].set_index("track").mae
    summary["mae_improvement_vs_historical"] = summary.apply(lambda row: baseline[row.track] - row.mae, axis=1)
    decision_operational = decision.query("ordering == 'pickups_then_returns'").groupby("model", as_index=False).agg(total_failures=("total_failures", "sum"), average_regret=("average_regret_vs_oracle", "mean"), service_level=("fulfilled_demand_rate", "mean"))
    bootstrap_summary = bootstrap.groupby("model", as_index=False).agg(mean_mae_difference=("mae_difference", "mean"), min_ci_low=("ci_95_low", "min"), max_ci_high=("ci_95_high", "max"))
    links = "\n".join(f"- [Figure: {path.stem}](figures/{path.name})" for path in figures)
    report.write_text(
        "# Citi Bike Forecast Benchmark — Core No-Weather Report\n\n"
        "## Design\n\n"
        "This primary comparison excludes weather. It uses the official VP-RNN 30-station source at 60-minute resolution, separate pickup and return counts, and three strict expanding chronological folds. The test windows are 2018-10-08–11-05, 2018-11-06–12-03, and 2018-12-04–12-31. The 60 source-DST-ambiguous rows are retained in the panel but excluded whenever their use would make a model support unequal.\n\n"
        "## Forecast accuracy\n\n" + summary.to_markdown(index=False, floatfmt=".4f") + "\n\n"
        "Negative MAE improvement means worse than historical average; positive means better. Recent average is the direct test of whether near-term history captures most obtainable improvement. Peak-period and station-level results are in the linked figures/tables; no causal interpretation is made.\n\n"
        "## Paired uncertainty\n\n" + bootstrap_summary.to_markdown(index=False, floatfmt=".4f") + "\n\n"
        "Intervals are paired day-block bootstrap comparisons against historical average; inspect `bootstrap_comparisons.csv` for each target, fold, horizon, and track.\n\n"
        "## Inventory decisions\n\n" + decision_operational.to_markdown(index=False, floatfmt=".4f") + "\n\n"
        "Each station-day searches every feasible starting inventory using the model’s day-ahead path, then replays realized aggregate hourly demand. Regret is relative to a realized-path oracle. Pickup-then-return is the main aggregate convention; reversed ordering is a reported sensitivity, not an operational assertion. Truck routing is outside scope.\n\n"
        "## Limits and next work\n\n"
        "Observed trips are treated as realized demand, so stockouts/capacity constraints can censor latent demand. The source’s repeated/nonexistent DST labels and three raw events outside the 2018 scope are documented in the data audit. The required observed-weather experiment has **not** run yet; when run it will be labeled `observed_weather_hindsight_upper_bound` and cannot be interpreted as forecast-vintage operational weather. Archived GBFS availability and forecast-vintage weather are future work. The optional Poisson GRU remains gated on earlier audits.\n\n"
        "## Machine-readable outputs\n\n"
        "- [Forecast metrics](tables/forecast_metrics.csv)\n- [Station metrics](tables/station_metrics.csv)\n- [Bootstrap comparisons](tables/bootstrap_comparisons.csv)\n- [Decision metrics](tables/decision_metrics.csv)\n- [Runtime metrics](tables/runtime_metrics.csv)\n- [Data quality](tables/data_quality.csv)\n\n"
        "## Figures\n\n" + links + "\n",
        encoding="utf-8",
    )
    return report


def build_report(config_path: str | Path) -> dict[str, str]:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    run_id = f"{config['run']['name']}_{sha256_file(config_path)[:12]}"
    tables = PROJECT_ROOT / "reports/tables"
    required = {name: tables / name for name in ("forecast_metrics.csv", "station_metrics.csv", "bootstrap_comparisons.csv", "decision_metrics.csv", "runtime_metrics.csv")}
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Report requires generated tables: {', '.join(missing)}")
    predictions = pd.read_parquet(PROJECT_ROOT / "artifacts/predictions" / f"{run_id}.parquet")
    figures = _core_figures(run_id, predictions, pd.read_csv(required["forecast_metrics.csv"]), pd.read_csv(required["station_metrics.csv"]), pd.read_csv(required["decision_metrics.csv"]), pd.read_csv(required["runtime_metrics.csv"]))
    report = _core_report(run_id, figures, predictions, pd.read_csv(required["forecast_metrics.csv"]), pd.read_csv(required["bootstrap_comparisons.csv"]), pd.read_csv(required["decision_metrics.csv"]))
    return {"report": str(report.relative_to(PROJECT_ROOT)), "figure_count": str(len(figures))}
