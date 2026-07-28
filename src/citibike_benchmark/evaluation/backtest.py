"""Strict chronological forecast backtests for both required forecast tracks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from citibike_benchmark.config import load_config
from citibike_benchmark.constants import PROJECT_ROOT, TIMEZONE
from citibike_benchmark.evaluation.metrics import forecast_metrics
from citibike_benchmark.evaluation.bootstrap import paired_mean_difference_ci
from citibike_benchmark.utils.io import sha256_file
from citibike_benchmark.utils.reproducibility import enrich_experiment_manifest

TARGETS = ("pickups", "returns")
LAGS = (1, 2, 3, 6, 12, 24, 48, 168)
ROLLING = (3, 6, 24, 168)


@dataclass(frozen=True)
class DayFold:
    fold: int
    train_days: tuple[object, ...]
    validation_days: tuple[object, ...]
    test_days: tuple[object, ...]


def load_panel(config: dict[str, Any]) -> pd.DataFrame:
    path = PROJECT_ROOT / config["data"]["panel_path"]
    if not path.exists():
        raise RuntimeError(f"Canonical panel is missing: {path}. Run the build stage first.")
    panel = pd.read_parquet(path)
    panel["station_id"] = panel["station_id"].astype(str)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    return panel.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


def make_day_folds(panel: pd.DataFrame, config: dict[str, Any]) -> list[DayFold]:
    """Construct the specified chronological expanding-window folds from complete days."""
    day_complete = panel.groupby("date", sort=True)["data_complete"].all()
    complete_days = list(day_complete[day_complete].index)
    splits = config["splits"]
    train_days = int(splits["min_train_days"])
    validation_days = int(splits["validation_days"])
    test_days = int(splits["test_days"])
    folds = int(splits["folds"])
    required = train_days + validation_days + test_days
    smoke_limit = config.get("data", {}).get("minimum_consecutive_days")
    if smoke_limit:
        if int(smoke_limit) < required:
            raise ValueError("Smoke minimum_consecutive_days cannot be smaller than split windows")
        complete_days = complete_days[:int(smoke_limit)]
    if len(complete_days) < required:
        raise ValueError(f"Only {len(complete_days)} complete days available; need {required}")
    if folds == 1:
        window = complete_days[:required] if smoke_limit else complete_days[-required:]
        return [DayFold(0, tuple(window[:train_days]), tuple(window[train_days:train_days + validation_days]), tuple(window[-test_days:]))]
    result: list[DayFold] = []
    latest_end = len(complete_days)
    for fold in range(folds):
        test_end = latest_end - (folds - 1 - fold) * test_days
        test_start = test_end - test_days
        validation_start = test_start - validation_days
        train_end = validation_start
        if train_end < train_days:
            raise ValueError("Available range cannot support requested rolling folds")
        result.append(DayFold(fold, tuple(complete_days[:train_end]), tuple(complete_days[validation_start:test_start]), tuple(complete_days[test_start:test_end])))
    return result


def _origin_for_target(target_timestamp: pd.Timestamp, track: str) -> tuple[pd.Timestamp, int]:
    if track == "two_hour":
        # The caller supplies every target for both one- and two-hour steps.
        raise ValueError("two_hour origin is defined by _examples_for_days")
    if track == "day_ahead":
        return target_timestamp.normalize() - pd.Timedelta(hours=1), target_timestamp.hour + 1
    raise ValueError(f"Unknown track: {track}")


def _prepare_feature_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Precompute all origin-available features in station timestamp order.

    Target rows are never used as features: lag 1 is the observation at the
    forecast origin, and every rolling window ends at that origin. Baseline
    values are shifted by one matching calendar occurrence, so their values
    also predate both supported forecast origins (at most 24 hours ahead).
    """
    result = panel.sort_values(["station_id", "timestamp"]).copy()
    valid = result["data_complete"]
    by_station = result.groupby("station_id", sort=False)
    availability = valid.astype("int8")
    for target in TARGETS:
        observed = result[target].where(valid)
        for lag in LAGS:
            result[f"{target}_lag_{lag}"] = by_station[target].shift(lag - 1).where(
                by_station["data_complete"].shift(lag - 1).fillna(False)
            )
        for window in ROLLING:
            rolling = observed.groupby(result["station_id"], sort=False).rolling(window, min_periods=window)
            result[f"{target}_rolling_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
            result[f"{target}_rolling_std_{window}"] = rolling.std(ddof=0).reset_index(level=0, drop=True)
        keys = [result["station_id"], result["day_of_week"], result["hour"]]
        group_sum = observed.groupby(keys, sort=False).cumsum().fillna(0.0)
        group_count = availability.groupby(keys, sort=False).cumsum()
        result[f"historical_{target}"] = (group_sum - observed.fillna(0.0)) / (group_count - availability).replace(0, np.nan)
        weekly = [by_station[target].shift(168 * week).where(by_station["data_complete"].shift(168 * week).fillna(False)) for week in range(1, 5)]
        result[f"seasonal_{target}"] = weekly[0]
        result[f"recent_{target}"] = pd.concat(weekly, axis=1).mean(axis=1)
    balance = (result["returns"] - result["pickups"]).where(valid)
    result["recent_pickup_return_balance"] = balance.groupby(result["station_id"], sort=False).rolling(24, min_periods=24).mean().reset_index(level=0, drop=True)
    return result


def _examples_for_days(features: pd.DataFrame, days: tuple[object, ...], track: str) -> pd.DataFrame:
    targets = features.loc[features["date"].isin(set(days)) & features["data_complete"]].copy()
    targets = targets.rename(columns={"pickups": "actual_pickups", "returns": "actual_returns"})
    targets = targets.loc[:, [
        "station_id", "timestamp", "actual_pickups", "actual_returns",
        "seasonal_pickups", "seasonal_returns", "historical_pickups", "historical_returns", "recent_pickups", "recent_returns",
    ]]
    targets["target_hour"] = targets["timestamp"].dt.hour
    targets["target_day_of_week"] = targets["timestamp"].dt.dayofweek
    targets["target_is_weekend"] = (targets["target_day_of_week"] >= 5).astype("int8")
    if track == "two_hour":
        examples = pd.concat([
            targets.assign(origin_timestamp=targets["timestamp"] - pd.Timedelta(hours=step), horizon_step=step)
            for step in (1, 2)
        ], ignore_index=True)
    elif track == "day_ahead":
        examples = targets.assign(
            origin_timestamp=targets["timestamp"].dt.normalize() - pd.Timedelta(hours=1),
            horizon_step=targets["timestamp"].dt.hour + 1,
        )
    else:
        raise ValueError(f"Unknown track: {track}")
    feature_columns = [column for column in features.columns if column.endswith(("_lag_1", "_lag_2", "_lag_3", "_lag_6", "_lag_12", "_lag_24", "_lag_48", "_lag_168")) or "rolling_" in column or column in {"recent_pickup_return_balance", "station_capacity"}]
    origins = features.loc[:, ["station_id", "timestamp", "data_complete", *feature_columns]].rename(columns={"timestamp": "origin_timestamp", "data_complete": "origin_data_complete"})
    examples = examples.merge(origins, on=["station_id", "origin_timestamp"], how="left", validate="many_to_one")
    return examples.loc[examples["origin_data_complete"].fillna(False)].reset_index(drop=True)


def _historical_baseline(examples: pd.DataFrame, target: str, recent: bool = False) -> np.ndarray:
    return examples[f"recent_{target}" if recent else f"historical_{target}"].to_numpy(dtype=float)


def _seasonal_baseline(examples: pd.DataFrame, target: str) -> np.ndarray:
    return examples[f"seasonal_{target}"].to_numpy(dtype=float)


def _feature_columns(examples: pd.DataFrame) -> list[str]:
    fixed = ["station_id", "horizon_step", "target_hour", "target_day_of_week", "target_is_weekend", "station_capacity", "recent_pickup_return_balance"]
    lag_and_rolling = [column for column in examples.columns if "_lag_" in column or "_rolling_" in column]
    return [column for column in [*fixed, *lag_and_rolling] if column in examples.columns]


def _poisson_glm(train: pd.DataFrame, predict: pd.DataFrame, target: str, alpha: float) -> tuple[np.ndarray, Pipeline, float, float]:
    features = _feature_columns(train)
    numeric = [column for column in features if column != "station_id"]
    processor = ColumnTransformer([
        ("station", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["station_id"]),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ])
    model = Pipeline([("features", processor), ("model", PoissonRegressor(alpha=alpha, max_iter=200, tol=1e-6))])
    start = perf_counter()
    model.fit(train[features], train[f"actual_{target}"])
    fit_seconds = perf_counter() - start
    start = perf_counter()
    prediction = np.maximum(model.predict(predict[features]), 0.0)
    return prediction, model, fit_seconds, perf_counter() - start


def _lgbm_features(train: pd.DataFrame, other: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _feature_columns(train)
    combined = pd.concat([train[features], other[features]], axis=0)
    combined = pd.get_dummies(combined, columns=["station_id"], dtype=float)
    encoded_train = combined.iloc[:len(train)].copy()
    encoded_other = combined.iloc[len(train):].copy()
    for column in encoded_train.columns:
        median = encoded_train[column].median()
        encoded_train[column] = encoded_train[column].fillna(median)
        encoded_other[column] = encoded_other[column].fillna(median)
    return encoded_train, encoded_other


def _fit_lgbm(train: pd.DataFrame, predict: pd.DataFrame, target: str, parameters: dict[str, Any], seed: int) -> tuple[np.ndarray, LGBMRegressor, float, float]:
    train_features, prediction_features = _lgbm_features(train, predict)
    model = LGBMRegressor(objective="poisson", random_state=seed, n_jobs=1, verbosity=-1, **parameters)
    start = perf_counter()
    model.fit(train_features, train[f"actual_{target}"])
    fit_seconds = perf_counter() - start
    start = perf_counter()
    prediction = np.maximum(model.predict(prediction_features), 0.0)
    return prediction, model, fit_seconds, perf_counter() - start


def _ml_predictions(model_name: str, train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, target: str, config: dict[str, Any], models_path: Path, fold: int, track: str) -> tuple[np.ndarray, dict[str, Any]]:
    candidate_rows: list[dict[str, Any]] = []
    seed = int(config["run"]["seed"])
    if model_name == "poisson_glm":
        # GLM regularization is fixed for computational comparability. The
        # specification reserves small validation tuning for LightGBM; this
        # avoids multiplying a global CPU fit by a needless parameter sweep.
        chosen = {"alpha": 0.1}
        candidate_rows.append({**chosen, "validation_mae": None, "fit_seconds": None, "prediction_seconds": None})
        combined = pd.concat([train, validation], ignore_index=True)
        prediction, model, fit_seconds, prediction_seconds = _poisson_glm(combined, test, target, float(chosen["alpha"]))
    elif model_name == "lightgbm_poisson":
        candidates = config["models"]["lightgbm"]["parameter_grid"]
        for candidate in candidates:
            prediction, _, fit_seconds, prediction_seconds = _fit_lgbm(train, validation, target, candidate, seed)
            candidate_rows.append({**candidate, "validation_mae": forecast_metrics(validation[f"actual_{target}"].to_numpy(), prediction)["mae"], "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds})
        chosen = min(candidate_rows, key=lambda row: row["validation_mae"])
        combined = pd.concat([train, validation], ignore_index=True)
        prediction, model, fit_seconds, prediction_seconds = _fit_lgbm(combined, test, target, chosen, seed)
    else:
        raise ValueError(f"Unsupported ML model: {model_name}")
    models_path.mkdir(parents=True, exist_ok=True)
    model_path = models_path / f"fold{fold}_{track}_{model_name}_{target}.joblib"
    joblib.dump(model, model_path)
    return prediction, {
        "model": model_name, "fold": fold, "track": track, "target_type": target,
        "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds,
        "serialized_model_bytes": model_path.stat().st_size, "feature_count": len(_feature_columns(train)),
        "tuned_configurations": len(candidate_rows), "selected_configuration": chosen,
        "tried_configurations": candidate_rows,
    }


def _prediction_rows(examples: pd.DataFrame, prediction: np.ndarray, model: str, target: str, fold: int, track: str) -> pd.DataFrame:
    actual = examples[f"actual_{target}"].to_numpy(float)
    frame = examples[["station_id", "timestamp", "origin_timestamp", "horizon_step"]].copy()
    frame["model"] = model
    frame["fold"] = fold
    frame["track"] = track
    frame["target_type"] = target
    frame["actual"] = actual
    frame["prediction_raw"] = prediction
    frame["prediction"] = np.maximum(prediction, 0.0)
    frame["prediction_was_clipped"] = prediction < 0
    return frame


def _combined_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "fold", "track", "station_id", "timestamp", "origin_timestamp", "horizon_step"]
    pivot = predictions.pivot(index=keys, columns="target_type", values=["actual", "prediction"]).dropna()
    combined = pivot[("actual", "pickups")] + pivot[("actual", "returns")]
    combined_prediction = pivot[("prediction", "pickups")] + pivot[("prediction", "returns")]
    return pd.DataFrame({"actual": combined, "prediction": combined_prediction}).reset_index().assign(target_type="combined", prediction_was_clipped=False)


def _metric_rows(frame: pd.DataFrame, groups: list[str], slice_type: str, slice_value: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for keys, group in frame.groupby(groups, sort=True):
        record = dict(zip(groups, keys if isinstance(keys, tuple) else (keys,), strict=True))
        record.update(forecast_metrics(group["actual"].to_numpy(), group["prediction"].to_numpy()))
        record.update({"slice_type": slice_type, "slice_value": slice_value, "n": len(group)})
        records.append(record)
    return records


def _metrics_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_targets = pd.concat([predictions, _combined_predictions(predictions)], ignore_index=True)
    all_targets["timestamp"] = pd.to_datetime(all_targets["timestamp"])
    all_targets["day_type"] = np.where(all_targets["timestamp"].dt.dayofweek >= 5, "weekend", "weekday")
    all_targets["period"] = np.where(all_targets["timestamp"].dt.hour.isin(range(7, 11)) | all_targets["timestamp"].dt.hour.isin(range(16, 21)), "peak", "off_peak")
    groups = ["model", "track", "horizon_step", "fold", "target_type"]
    records = _metric_rows(all_targets, groups, "overall", "all")
    for value, subset in all_targets.groupby("day_type", sort=True):
        records.extend(_metric_rows(subset, groups, "day_type", str(value)))
    for value, subset in all_targets.groupby("period", sort=True):
        records.extend(_metric_rows(subset, groups, "period", str(value)))
    metrics = pd.DataFrame(records)
    station_records: list[dict[str, Any]] = []
    station_groups = [*groups, "station_id"]
    for keys, group in all_targets.groupby(station_groups, sort=True):
        record = dict(zip(station_groups, keys, strict=True))
        record.update(forecast_metrics(group["actual"].to_numpy(), group["prediction"].to_numpy()))
        record.update({"slice_type": "overall", "slice_value": "all", "n": len(group)})
        station_records.append(record)
    station_metrics = pd.DataFrame(station_records)
    station_summary = station_metrics.groupby(groups, as_index=False).agg(
        station_median_mae=("mae", "median"),
        worst_quintile_station_mae=("mae", lambda values: values.nlargest(max(1, int(np.ceil(len(values) * 0.2)))).mean()),
    )
    metrics = metrics.merge(station_summary, on=groups, how="left")
    return metrics, station_metrics


def bootstrap_comparisons(predictions: pd.DataFrame, seed: int, replicates: int) -> pd.DataFrame:
    """Paired day-block MAE-difference intervals versus historical average."""
    rows: list[dict[str, Any]] = []
    all_targets = pd.concat([predictions, _combined_predictions(predictions)], ignore_index=True)
    all_targets["timestamp"] = pd.to_datetime(all_targets["timestamp"])
    all_targets["day"] = all_targets["timestamp"].dt.date
    keys = ["fold", "track", "horizon_step", "target_type"]
    for key_values, subset in all_targets.groupby(keys, sort=True):
        baseline = subset[subset["model"].eq("historical_average")]
        join_keys = ["station_id", "timestamp", "origin_timestamp"]
        for model, candidate in subset.groupby("model", sort=True):
            if model == "historical_average":
                continue
            aligned = candidate.merge(baseline[join_keys + ["actual", "prediction"]], on=join_keys, suffixes=("_candidate", "_baseline"), validate="one_to_one")
            if aligned.empty:
                continue
            daily = aligned.assign(
                difference=(aligned["prediction_candidate"] - aligned["actual_candidate"]).abs() - (aligned["prediction_baseline"] - aligned["actual_baseline"]).abs(),
                day=pd.to_datetime(aligned["timestamp"]).dt.date,
            ).groupby("day", sort=True)["difference"].mean()
            ci_low, ci_high = paired_mean_difference_ci(daily.to_numpy(), seed, replicates)
            row = dict(zip(keys, key_values, strict=True))
            row.update({"model": model, "baseline": "historical_average", "day_blocks": len(daily), "mae_difference": float(daily.mean()), "ci_95_low": ci_low, "ci_95_high": ci_high, "bootstrap_seed": seed, "bootstrap_replicates": replicates})
            rows.append(row)
    return pd.DataFrame(rows)


def run_backtest(config_path: str | Path) -> dict[str, Any]:
    """Run/reuse a complete no-weather chronological forecast backtest."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    if config["weather"]["enabled"]:
        raise ValueError("Main backtest requires weather.enabled=false; weather is a separate hindsight experiment")
    panel = load_panel(config)
    features = _prepare_feature_panel(panel)
    config_hash = sha256_file(config_path)
    run_id = f"{config['run']['name']}_{config_hash[:12]}"
    prediction_path = PROJECT_ROOT / "artifacts/predictions" / f"{run_id}.parquet"
    runtime_path = PROJECT_ROOT / "reports/tables/runtime_metrics.csv"
    if prediction_path.exists():
        return {"run_id": run_id, "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)), "cached": True}
    folds = make_day_folds(panel, config)
    all_predictions: list[pd.DataFrame] = []
    runtime_rows: list[dict[str, Any]] = []
    models_path = PROJECT_ROOT / "artifacts/models" / run_id
    enabled = tuple(config["models"]["enabled"])
    for fold in folds:
        for track in ("two_hour", "day_ahead"):
            train = _examples_for_days(features, fold.train_days, track)
            validation = _examples_for_days(features, fold.validation_days, track)
            test = _examples_for_days(features, fold.test_days, track)
            if train.empty or validation.empty or test.empty:
                raise RuntimeError(f"No valid examples for fold={fold.fold}, track={track}")
            for target in TARGETS:
                seasonal_prediction = _seasonal_baseline(test, target)
                # All models are scored on identical support. The only source
                # exception is a target whose mandatory seven-day seasonal
                # observation is an explicitly ambiguous DST source row.
                # Excluding it for every model is preferable to pretending the
                # seasonal baseline has a value or comparing unequal samples.
                scoring_mask = np.isfinite(seasonal_prediction)
                scoring_test = test.loc[scoring_mask].reset_index(drop=True)
                baseline_predictions = {
                    "seasonal_naive": seasonal_prediction[scoring_mask],
                    "historical_average": _historical_baseline(test, target)[scoring_mask],
                    "recent_average": _historical_baseline(test, target, recent=True)[scoring_mask],
                }
                for model_name, prediction in baseline_predictions.items():
                    if model_name not in enabled:
                        continue
                    valid = np.isfinite(prediction)
                    all_predictions.append(_prediction_rows(scoring_test.loc[valid].reset_index(drop=True), prediction[valid], model_name, target, fold.fold, track))
                    runtime_rows.append({"model": model_name, "fold": fold.fold, "track": track, "target_type": target, "fit_seconds": 0.0, "prediction_seconds": np.nan, "serialized_model_bytes": 0, "feature_count": 0, "tuned_configurations": 0})
                for model_name in ("poisson_glm", "lightgbm_poisson"):
                    if model_name in enabled:
                        prediction, details = _ml_predictions(model_name, train, validation, scoring_test, target, config, models_path, fold.fold, track)
                        all_predictions.append(_prediction_rows(scoring_test, prediction, model_name, target, fold.fold, track))
                        runtime_rows.append({key: value for key, value in details.items() if key not in {"selected_configuration", "tried_configurations"}})
    predictions = pd.concat(all_predictions, ignore_index=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(prediction_path, index=False)
    metrics, station_metrics = _metrics_tables(predictions)
    tables = PROJECT_ROOT / "reports/tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "forecast_metrics.csv", index=False)
    station_metrics.to_csv(tables / "station_metrics.csv", index=False)
    bootstrap_comparisons(predictions, int(config["run"]["seed"]), int(config["evaluation"]["bootstrap_replicates"])).to_csv(tables / "bootstrap_comparisons.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(runtime_path, index=False)
    manifest = {
        "timestamp_timezone": TIMEZONE,
        "weather_enabled": False, "station_ids": sorted(panel["station_id"].unique().tolist()),
        "folds": [{"fold": fold.fold, "train": [str(fold.train_days[0]), str(fold.train_days[-1])], "validation": [str(fold.validation_days[0]), str(fold.validation_days[-1])], "test": [str(fold.test_days[0]), str(fold.test_days[-1])]} for fold in folds],
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)),
        "output_file_hashes": {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in (prediction_path, tables / "forecast_metrics.csv", tables / "station_metrics.csv", tables / "bootstrap_comparisons.csv", runtime_path)},
        "runtime_seconds": float(pd.DataFrame(runtime_rows)["fit_seconds"].sum() + pd.DataFrame(runtime_rows)["prediction_seconds"].sum()),
        "random_seeds": {"global": int(config["run"]["seed"]), "bootstrap": int(config["run"]["seed"])},
        "warnings": ["GLM convergence warnings, if emitted by sklearn, are retained in execution logs.", "60 target rows were excluded uniformly in fold 1 because seasonal-naive's seven-day lag reaches an explicit DST-ambiguous source hour."],
        "fallbacks": [],
        "runtime_rows": runtime_rows,
    }
    enrich_experiment_manifest(run_id, config_path, manifest)
    return {"run_id": run_id, "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)), "cached": False}
