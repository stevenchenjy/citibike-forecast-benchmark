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
from sklearn.preprocessing import OneHotEncoder

from citibike_benchmark.config import load_config
from citibike_benchmark.constants import PROJECT_ROOT, TIMEZONE
from citibike_benchmark.evaluation.metrics import forecast_metrics
from citibike_benchmark.utils.io import sha256_file, write_json

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
    latest_start = len(complete_days) - test_days
    for fold in range(folds):
        test_end = latest_start - (folds - 1 - fold) * test_days
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


def _history_features(history: pd.DataFrame, origin: pd.Timestamp, capacity: int) -> dict[str, float]:
    available = history.loc[(history.index <= origin) & history["data_complete"]]
    features: dict[str, float] = {"station_capacity": float(capacity)}
    for demand in TARGETS:
        for lag in LAGS:
            instant = origin - pd.Timedelta(hours=lag - 1)
            features[f"{demand}_lag_{lag}"] = float(available.at[instant, demand]) if instant in available.index else np.nan
        for window in ROLLING:
            window_start = origin - pd.Timedelta(hours=window - 1)
            values = available.loc[window_start:origin, demand]
            features[f"{demand}_rolling_mean_{window}"] = float(values.mean()) if len(values) == window else np.nan
            features[f"{demand}_rolling_std_{window}"] = float(values.std(ddof=0)) if len(values) == window else np.nan
    balance = available.loc[origin - pd.Timedelta(hours=23):origin]
    features["recent_pickup_return_balance"] = float((balance["returns"] - balance["pickups"]).mean()) if len(balance) == 24 else np.nan
    return features


def _append_example(records: list[dict[str, Any]], station_id: str, history: pd.DataFrame, target_timestamp: pd.Timestamp, origin: pd.Timestamp, horizon_step: int) -> None:
    if target_timestamp not in history.index or origin not in history.index:
        return
    target = history.loc[target_timestamp]
    origin_row = history.loc[origin]
    if not bool(target["data_complete"]) or not bool(origin_row["data_complete"]):
        return
    record: dict[str, Any] = {
        "station_id": station_id, "timestamp": target_timestamp, "origin_timestamp": origin,
        "horizon_step": horizon_step, "target_hour": target_timestamp.hour,
        "target_day_of_week": target_timestamp.dayofweek, "target_is_weekend": int(target_timestamp.dayofweek >= 5),
        "actual_pickups": float(target["pickups"]), "actual_returns": float(target["returns"]),
    }
    record.update(_history_features(history, origin, int(target["station_capacity"])))
    records.append(record)


def _examples_for_days(panel: pd.DataFrame, days: tuple[object, ...], track: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    days_as_dates = set(days)
    for station_id, station in panel.groupby("station_id", sort=True):
        history = station.set_index("timestamp").sort_index()
        target_rows = station[station["date"].isin(days_as_dates)]
        for target_timestamp in target_rows["timestamp"]:
            if track == "two_hour":
                for horizon_step in (1, 2):
                    _append_example(records, station_id, history, target_timestamp, target_timestamp - pd.Timedelta(hours=horizon_step), horizon_step)
            elif track == "day_ahead":
                origin = target_timestamp.normalize() - pd.Timedelta(hours=1)
                _append_example(records, station_id, history, target_timestamp, origin, target_timestamp.hour + 1)
            else:
                raise ValueError(f"Unknown track: {track}")
    return pd.DataFrame.from_records(records)


def _historical_baseline(panel: pd.DataFrame, examples: pd.DataFrame, target: str, recent: bool = False) -> np.ndarray:
    output: list[float] = []
    for example in examples.itertuples(index=False):
        station = panel[panel["station_id"] == example.station_id]
        history = station[(station["timestamp"] < example.origin_timestamp) & station["data_complete"]]
        matching = history[(history["day_of_week"] == example.target_day_of_week) & (history["hour"] == example.target_hour)]
        if recent:
            selected = []
            for week in range(1, 5):
                timestamp = example.timestamp - pd.Timedelta(hours=168 * week)
                values = station[(station["timestamp"] == timestamp) & (station["timestamp"] < example.origin_timestamp) & station["data_complete"]][target]
                if len(values):
                    selected.append(float(values.iloc[0]))
            prediction = float(np.mean(selected)) if selected else float(matching[target].mean())
        else:
            prediction = float(matching[target].mean())
        output.append(prediction)
    return np.asarray(output, dtype=float)


def _seasonal_baseline(panel: pd.DataFrame, examples: pd.DataFrame, target: str) -> np.ndarray:
    output: list[float] = []
    for example in examples.itertuples(index=False):
        target_time = example.timestamp - pd.Timedelta(hours=168)
        values = panel[(panel["station_id"] == example.station_id) & (panel["timestamp"] == target_time) & panel["data_complete"]][target]
        output.append(float(values.iloc[0]) if len(values) and target_time < example.origin_timestamp else np.nan)
    return np.asarray(output, dtype=float)


def _feature_columns(examples: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "origin_timestamp", "actual_pickups", "actual_returns"}
    return [column for column in examples.columns if column not in excluded]


def _poisson_glm(train: pd.DataFrame, predict: pd.DataFrame, target: str, alpha: float) -> tuple[np.ndarray, Pipeline, float, float]:
    features = _feature_columns(train)
    numeric = [column for column in features if column != "station_id"]
    processor = ColumnTransformer([
        ("station", OneHotEncoder(handle_unknown="ignore"), ["station_id"]),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median"))]), numeric),
    ])
    model = Pipeline([("features", processor), ("model", PoissonRegressor(alpha=alpha, max_iter=300))])
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
        candidates = [{"alpha": alpha} for alpha in (0.001, 0.01, 0.1)]
        for candidate in candidates:
            prediction, _, fit_seconds, prediction_seconds = _poisson_glm(train, validation, target, float(candidate["alpha"]))
            candidate_rows.append({**candidate, "validation_mae": forecast_metrics(validation[f"actual_{target}"].to_numpy(), prediction)["mae"], "fit_seconds": fit_seconds, "prediction_seconds": prediction_seconds})
        chosen = min(candidate_rows, key=lambda row: row["validation_mae"])
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


def _metrics_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = ["model", "track", "horizon_step", "fold", "target_type"]
    records = []
    for keys, group in predictions.groupby(groups, sort=True):
        record = dict(zip(groups, keys, strict=True))
        record.update(forecast_metrics(group["actual"].to_numpy(), group["prediction"].to_numpy()))
        record["n"] = len(group)
        records.append(record)
    metrics = pd.DataFrame(records)
    station_records = []
    for keys, group in predictions.groupby([*groups, "station_id"], sort=True):
        record = dict(zip([*groups, "station_id"], keys, strict=True))
        record.update(forecast_metrics(group["actual"].to_numpy(), group["prediction"].to_numpy()))
        record["n"] = len(group)
        station_records.append(record)
    return metrics, pd.DataFrame(station_records)


def run_backtest(config_path: str | Path) -> dict[str, Any]:
    """Run/reuse a complete no-weather chronological forecast backtest."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path)
    if config["weather"]["enabled"]:
        raise ValueError("Main backtest requires weather.enabled=false; weather is a separate hindsight experiment")
    panel = load_panel(config)
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
            train = _examples_for_days(panel, fold.train_days, track)
            validation = _examples_for_days(panel, fold.validation_days, track)
            test = _examples_for_days(panel, fold.test_days, track)
            if train.empty or validation.empty or test.empty:
                raise RuntimeError(f"No valid examples for fold={fold.fold}, track={track}")
            for target in TARGETS:
                baseline_predictions = {
                    "seasonal_naive": _seasonal_baseline(panel, test, target),
                    "historical_average": _historical_baseline(panel, test, target),
                    "recent_average": _historical_baseline(panel, test, target, recent=True),
                }
                for model_name, prediction in baseline_predictions.items():
                    if model_name not in enabled:
                        continue
                    valid = np.isfinite(prediction)
                    all_predictions.append(_prediction_rows(test.loc[valid].reset_index(drop=True), prediction[valid], model_name, target, fold.fold, track))
                    runtime_rows.append({"model": model_name, "fold": fold.fold, "track": track, "target_type": target, "fit_seconds": 0.0, "prediction_seconds": np.nan, "serialized_model_bytes": 0, "feature_count": 0, "tuned_configurations": 0})
                for model_name in ("poisson_glm", "lightgbm_poisson"):
                    if model_name in enabled:
                        prediction, details = _ml_predictions(model_name, train, validation, test, target, config, models_path, fold.fold, track)
                        all_predictions.append(_prediction_rows(test, prediction, model_name, target, fold.fold, track))
                        runtime_rows.append({key: value for key, value in details.items() if key not in {"selected_configuration", "tried_configurations"}})
    predictions = pd.concat(all_predictions, ignore_index=True)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(prediction_path, index=False)
    metrics, station_metrics = _metrics_tables(predictions)
    tables = PROJECT_ROOT / "reports/tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / "forecast_metrics.csv", index=False)
    station_metrics.to_csv(tables / "station_metrics.csv", index=False)
    pd.DataFrame(runtime_rows).to_csv(runtime_path, index=False)
    manifest = {
        "run_id": run_id, "timestamp_timezone": TIMEZONE, "config_path": str(config_path.relative_to(PROJECT_ROOT)), "config_sha256": config_hash,
        "weather_enabled": False, "station_ids": sorted(panel["station_id"].unique().tolist()),
        "folds": [{"fold": fold.fold, "train": [str(fold.train_days[0]), str(fold.train_days[-1])], "validation": [str(fold.validation_days[0]), str(fold.validation_days[-1])], "test": [str(fold.test_days[0]), str(fold.test_days[-1])]} for fold in folds],
        "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)), "runtime_rows": runtime_rows,
    }
    write_json(PROJECT_ROOT / "artifacts/run_manifests" / f"{run_id}.json", manifest)
    return {"run_id": run_id, "prediction_path": str(prediction_path.relative_to(PROJECT_ROOT)), "cached": False}
