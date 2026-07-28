"""Compact, deterministic global Poisson GRU for the gated optional run."""
from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

MODEL_NAME = "poisson_gru"
ENABLED_BY_DEFAULT = False


def _torch() -> Any:
    """Import the optional modern PyTorch runtime only for the GRU milestone."""
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
        from torch.utils.data import DataLoader, Subset, TensorDataset
    except ImportError as exc:  # pragma: no cover - exercised only without deep extra
        raise RuntimeError("Poisson GRU requires `uv sync --extra deep`") from exc
    return torch, nn, functional, DataLoader, Subset, TensorDataset


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # MPS kernels can warn rather than fail if an operation has no deterministic
    # implementation. The seed and serialized run configuration are retained.
    torch.use_deterministic_algorithms(True, warn_only=True)


def _device(torch: Any, requested: str) -> Any:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "auto" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sequence_arrays(examples: pd.DataFrame, feature_panel: pd.DataFrame, sequence_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create origin-ending histories without reading target-time demand.

    Each step contains log1p pickup/return counts and a completeness flag. A
    source-DST-incomplete hour is zero-filled with its flag set to zero rather
    than removing an otherwise valid target. This retains the exact support
    used by the required baseline comparison.
    """
    required = {"station_id", "origin_sequence_index", "target_hour", "target_day_of_week", "horizon_step"}
    missing = required.difference(examples.columns)
    if missing:
        raise ValueError(f"GRU examples are missing required origin metadata: {sorted(missing)}")
    ordered = feature_panel.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    station_ids = sorted(ordered["station_id"].astype(str).unique())
    station_to_index = {station_id: index for index, station_id in enumerate(station_ids)}
    histories: dict[str, np.ndarray] = {}
    for station_id, group in ordered.groupby("station_id", sort=False):
        counts = group[["pickups", "returns"]].to_numpy(dtype=np.float32, copy=True)
        complete = group["data_complete"].to_numpy(dtype=np.float32)
        counts[complete == 0.0] = 0.0
        histories[str(station_id)] = np.column_stack([np.log1p(counts), complete])
    size = len(examples)
    sequences = np.zeros((size, sequence_length, 3), dtype=np.float32)
    stations = np.empty(size, dtype=np.int64)
    context = np.empty((size, 5), dtype=np.float32)
    for row_index, row in enumerate(examples[["station_id", "origin_sequence_index", "target_hour", "target_day_of_week", "horizon_step"]].itertuples(index=False)):
        station_id = str(row.station_id)
        origin = int(row.origin_sequence_index)
        history = histories[station_id]
        if origin < 0 or origin >= len(history):
            raise ValueError("GRU origin sequence index is outside its station history")
        start = max(0, origin - sequence_length + 1)
        selected = history[start:origin + 1]
        sequences[row_index, -len(selected):] = selected
        stations[row_index] = station_to_index[station_id]
        context[row_index] = (
            np.sin(2 * np.pi * int(row.target_hour) / 24),
            np.cos(2 * np.pi * int(row.target_hour) / 24),
            np.sin(2 * np.pi * int(row.target_day_of_week) / 7),
            np.cos(2 * np.pi * int(row.target_day_of_week) / 7),
            float(row.horizon_step) / 24.0,
        )
    return sequences, stations, context, np.array(station_ids)


def fit_predict_poisson_gru(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_panel: pd.DataFrame,
    target: str,
    settings: dict[str, Any],
    seed: int,
    model_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one capped global Poisson GRU and predict held-out examples."""
    torch, nn, functional, DataLoader, Subset, TensorDataset = _torch()
    _seed_everything(torch, seed)
    device = _device(torch, str(settings["device"]))
    sequence_length = int(settings["sequence_length"])
    train_sequence, train_station, train_context, station_ids = _sequence_arrays(train, feature_panel, sequence_length)
    validation_sequence, validation_station, validation_context, validation_station_ids = _sequence_arrays(validation, feature_panel, sequence_length)
    test_sequence, test_station, test_context, test_station_ids = _sequence_arrays(test, feature_panel, sequence_length)
    if not np.array_equal(station_ids, validation_station_ids) or not np.array_equal(station_ids, test_station_ids):
        raise ValueError("GRU station encoding differs across train, validation, and test")

    class CompactPoissonGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.GRU(input_size=3, hidden_size=int(settings["hidden_size"]), num_layers=1, batch_first=True)
            self.station_embedding = nn.Embedding(len(station_ids), int(settings["station_embedding_size"]))
            combined = int(settings["hidden_size"]) + int(settings["station_embedding_size"]) + 5
            self.head = nn.Sequential(nn.Linear(combined, int(settings["head_size"])), nn.ReLU(), nn.Linear(int(settings["head_size"]), 1))

        def forward(self, sequence: Any, station: Any, context: Any) -> Any:
            encoded, _ = self.encoder(sequence)
            joined = torch.cat([encoded[:, -1], self.station_embedding(station), context], dim=1)
            return torch.nn.functional.softplus(self.head(joined).squeeze(1)).clamp(min=1e-6, max=500.0)

    def dataset(sequence: np.ndarray, station: np.ndarray, context: np.ndarray, labels: np.ndarray) -> Any:
        return TensorDataset(torch.from_numpy(sequence), torch.from_numpy(station), torch.from_numpy(context), torch.from_numpy(labels.astype(np.float32)))

    train_data = dataset(train_sequence, train_station, train_context, train[f"actual_{target}"].to_numpy())
    validation_data = dataset(validation_sequence, validation_station, validation_context, validation[f"actual_{target}"].to_numpy())
    test_data = dataset(test_sequence, test_station, test_context, np.zeros(len(test), dtype=np.float32))

    def capped(data: Any, maximum: int) -> Any:
        if len(data) <= maximum:
            return data
        # Uniform chronological subsampling is deterministic and never reads
        # validation/test values when constructing the training sample.
        indices = np.linspace(0, len(data) - 1, maximum, dtype=np.int64)
        return Subset(data, indices.tolist())

    train_data = capped(train_data, int(settings["max_train_examples"]))
    validation_data = capped(validation_data, int(settings["max_validation_examples"]))
    batch_size = int(settings["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model = CompactPoissonGRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]), weight_decay=float(settings["weight_decay"]))

    def loss_for(loader: Any, training: bool) -> float:
        total = 0.0
        observations = 0
        if training:
            model.train()
        else:
            model.eval()
        for sequence, station, context, labels in loader:
            sequence, station, context, labels = sequence.to(device), station.to(device), context.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                rates = model(sequence, station, context)
                loss = functional.poisson_nll_loss(rates, labels, log_input=False, full=False, reduction="mean")
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
            total += float(loss.detach().cpu()) * len(labels)
            observations += len(labels)
        return total / observations

    start = perf_counter()
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    epochs_trained = 0
    for epoch in range(int(settings["max_epochs"])):
        loss_for(train_loader, training=True)
        validation_loss = loss_for(validation_loader, training=False)
        epochs_trained = epoch + 1
        if validation_loss < best_loss - float(settings["min_delta"]):
            best_loss = validation_loss
            best_state = deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(settings["patience"]):
                break
    if best_state is None:  # pragma: no cover - defensive against invalid floating point hardware
        raise RuntimeError("Poisson GRU did not produce an early-stopping checkpoint")
    model.load_state_dict(best_state)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for sequence, station, context, _ in test_loader:
            rates = model(sequence.to(device), station.to(device), context.to(device))
            predictions.append(rates.detach().cpu().numpy())
    prediction = np.maximum(np.concatenate(predictions), 0.0)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "station_ids": station_ids.tolist(), "target": target, "settings": settings,
        "seed": seed, "device": str(device), "best_validation_poisson_nll": best_loss,
    }, model_path)
    return prediction, {
        "fit_seconds": fit_seconds, "prediction_seconds": perf_counter() - start,
        "serialized_model_bytes": model_path.stat().st_size,
        "feature_count": sequence_length * 3 + 5,
        "tuned_configurations": 0,
        "selected_configuration": settings,
        "tried_configurations": [],
        "device": str(device), "epochs_trained": epochs_trained,
        "best_validation_poisson_nll": best_loss,
        "train_examples_used": len(train_data), "validation_examples_used": len(validation_data),
    }
