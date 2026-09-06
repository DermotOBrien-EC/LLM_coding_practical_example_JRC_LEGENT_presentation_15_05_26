from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from pytorch_lightning import seed_everything
from sklearn.preprocessing import MaxAbsScaler

from common import ForecastResult, QUANTILES, SEED, mape_pct

EPOCH_CANDIDATES = (6, 12, 18, 24, 30)
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_MIN_DELTA_PCT = 0.01
TRAINING_STRIDE = 4
BATCH_SIZE = 512

ModelFactory = Callable[[dict[str, Any]], TorchForecastingModel]


def to_darts_series(series: pd.Series) -> TimeSeries:
    return TimeSeries.from_times_and_values(
        series.index,
        series.to_numpy(dtype=np.float32),
        columns=["load"],
        fill_missing_dates=False,
    )


def _trainer_configuration() -> dict[str, Any]:
    return {
        "accelerator": "cpu",
        "devices": 1,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
        "deterministic": True,
    }


def _common_model_parameters(n_epochs: int) -> dict[str, Any]:
    return {
        "input_chunk_length": 168,
        "output_chunk_length": 168,
        "n_epochs": n_epochs,
        "batch_size": BATCH_SIZE,
        "optimizer_kwargs": {"lr": 1e-3},
        "random_state": SEED,
        "pl_trainer_kwargs": _trainer_configuration(),
        "save_checkpoints": False,
        "force_reset": True,
    }


def _quantile_component(series: TimeSeries, probability: float) -> str:
    suffix = f"_q{probability:.3f}"
    matches = [str(component) for component in series.components if str(component).endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one Darts component ending in {suffix}, found {matches}")
    return matches[0]


def _recursive_quantile_parameters(
    model: TorchForecastingModel,
    horizon: int,
    history: TimeSeries,
) -> dict[float, np.ndarray]:
    context = history
    chunks: dict[float, list[np.ndarray]] = {probability: [] for probability in QUANTILES}
    remaining = horizon
    while remaining > 0:
        chunk_length = min(168, remaining)
        parameters = model.predict(
            n=chunk_length,
            series=context,
            num_samples=1,
            predict_likelihood_parameters=True,
            verbose=False,
            show_warnings=False,
        )
        for probability in QUANTILES:
            component = _quantile_component(parameters, probability)
            chunks[probability].append(
                np.asarray(parameters[component].values(copy=False), dtype=float).reshape(-1)
            )
        median_component = _quantile_component(parameters, 0.5)
        median = parameters[median_component].with_columns_renamed(
            median_component, str(context.components[0])
        )
        context = context.append(median)
        remaining -= chunk_length
    return {
        probability: np.concatenate(probability_chunks)
        for probability, probability_chunks in chunks.items()
    }


def _inverse_quantile_arrays(
    scaler: Scaler,
    index: pd.DatetimeIndex,
    scaled_quantiles: dict[float, np.ndarray],
) -> dict[float, np.ndarray]:
    restored: dict[float, np.ndarray] = {}
    for probability, values in scaled_quantiles.items():
        scaled_series = TimeSeries.from_times_and_values(
            index,
            values.astype(np.float32),
            columns=["load"],
            fill_missing_dates=False,
        )
        restored_series = scaler.inverse_transform(scaled_series)
        restored[probability] = np.asarray(restored_series.values(copy=False), dtype=float).reshape(
            -1
        )
    median = restored[0.5]
    restored[0.1] = np.minimum(restored[0.1], median)
    restored[0.025] = np.minimum(restored[0.025], restored[0.1])
    restored[0.9] = np.maximum(restored[0.9], median)
    restored[0.975] = np.maximum(restored[0.975], restored[0.9])
    return restored


def run_deep_model(
    name: str,
    model_factory: ModelFactory,
    architecture_parameters: dict[str, Any],
    train: pd.Series,
    validation: pd.Series,
    train_validation: pd.Series,
    test_index: pd.DatetimeIndex,
) -> ForecastResult:
    started = time.perf_counter()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    train_scaler = Scaler(MaxAbsScaler())
    train_scaled = train_scaler.fit_transform(to_darts_series(train))
    validation_records: list[dict[str, float | int]] = []
    best_epochs: int | None = None
    best_validation_mape = float("inf")
    non_improving_candidates = 0

    for candidate_epochs in EPOCH_CANDIDATES:
        seed_everything(SEED, workers=True, verbose=False)
        candidate_model = model_factory(_common_model_parameters(candidate_epochs))
        candidate_model.fit(
            train_scaled,
            verbose=False,
            stride=TRAINING_STRIDE,
        )
        scaled_quantiles = _recursive_quantile_parameters(
            candidate_model,
            len(validation),
            train_scaled,
        )
        validation_quantiles = _inverse_quantile_arrays(
            train_scaler,
            pd.DatetimeIndex(validation.index),
            scaled_quantiles,
        )
        candidate_mape = mape_pct(validation.to_numpy(), validation_quantiles[0.5])
        validation_records.append(
            {
                "epochs": candidate_epochs,
                "validation_mape_pct": candidate_mape,
            }
        )
        if candidate_mape < best_validation_mape - EARLY_STOPPING_MIN_DELTA_PCT:
            best_validation_mape = candidate_mape
            best_epochs = candidate_epochs
            non_improving_candidates = 0
        else:
            non_improving_candidates += 1
            if non_improving_candidates >= EARLY_STOPPING_PATIENCE:
                break

    if best_epochs is None:
        raise RuntimeError(f"{name} validation produced no epoch candidate")

    seed_everything(SEED, workers=True, verbose=False)
    final_scaler = Scaler(MaxAbsScaler())
    train_validation_scaled = final_scaler.fit_transform(to_darts_series(train_validation))
    final_model = model_factory(_common_model_parameters(best_epochs))
    final_model.fit(
        train_validation_scaled,
        verbose=False,
        stride=TRAINING_STRIDE,
    )
    scaled_quantiles = _recursive_quantile_parameters(
        final_model,
        len(test_index),
        train_validation_scaled,
    )
    quantiles = _inverse_quantile_arrays(final_scaler, test_index, scaled_quantiles)
    return ForecastResult(
        name=name,
        point=quantiles[0.5],
        quantiles=quantiles,
        runtime_seconds=time.perf_counter() - started,
        hyperparameters={
            **architecture_parameters,
            "input_chunk_length": 168,
            "output_chunk_length": 168,
            "epoch_candidates": list(EPOCH_CANDIDATES),
            "selected_epochs": best_epochs,
            "early_stopping_patience_candidates": EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta_pct": EARLY_STOPPING_MIN_DELTA_PCT,
            "early_stopping_monitor": "fixed_origin_recursive_validation_mape_pct",
            "validation_mape_pct": best_validation_mape,
            "validation_candidates": validation_records,
            "training_stride": TRAINING_STRIDE,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "learning_rate": 1e-3,
            "scaler": "MaxAbsScaler",
            "likelihood": "quantile_regression",
            "quantile_levels": list(QUANTILES),
            "probabilistic_output": "direct_quantile_parameters",
            "accelerator": "cpu",
        },
    )
