from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from darts.models.forecasting.nbeats import NBEATSModel
from darts.utils.likelihood_models.torch import QuantileRegression
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics.regression import MeanAbsolutePercentageError

from common import (
    QUANTILES,
    SEED,
    DataSplits,
    ForecastResult,
    PositiveMeanScaler,
    darts_samples_to_quantiles,
    make_darts_series,
    mean_absolute_percentage_error,
)

INPUT_CHUNK_LENGTH = 168
OUTPUT_CHUNK_LENGTH = 168
MAX_EPOCHS = 6
BATCH_SIZE = 1024
N_SAMPLES = 500
MONITOR = "val_MeanAbsolutePercentageError"


class ValidationMapeTracker(pl.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.best_mape = float("inf")
        self.best_epoch = 0

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        del pl_module
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get(MONITOR)
        if metric is None:
            return
        value = float(metric.detach().cpu().item())
        if np.isfinite(value) and value < self.best_mape:
            self.best_mape = value
            self.best_epoch = int(trainer.current_epoch)


def _make_model(
    n_epochs: int,
    accelerator: str,
    devices: int | str,
    callbacks: list[pl.Callback],
) -> NBEATSModel:
    return NBEATSModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        n_epochs=n_epochs,
        batch_size=BATCH_SIZE,
        random_state=SEED,
        likelihood=QuantileRegression(quantiles=list(QUANTILES)),
        torch_metrics=MeanAbsolutePercentageError(),
        optimizer_kwargs={"lr": 1e-3},
        pl_trainer_kwargs={
            "accelerator": accelerator,
            "devices": devices,
            "deterministic": True,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
            "callbacks": callbacks,
        },
        save_checkpoints=False,
        force_reset=True,
        show_warnings=True,
    )


def _fit_validation_model(
    train_series: Any,
    validation_series: Any,
    accelerator: str,
    devices: int | str,
) -> tuple[NBEATSModel, ValidationMapeTracker]:
    tracker = ValidationMapeTracker()
    stopper = EarlyStopping(
        monitor=MONITOR,
        mode="min",
        patience=2,
        min_delta=1e-4,
        check_finite=True,
    )
    model = _make_model(MAX_EPOCHS, accelerator, devices, [tracker, stopper])
    model.fit(
        train_series,
        val_series=validation_series,
        dataloader_kwargs={"num_workers": 0},
        verbose=False,
    )
    if not np.isfinite(tracker.best_mape):
        raise RuntimeError(f"N-BEATS did not report the monitored metric {MONITOR}.")
    return model, tracker


def run(splits: DataSplits) -> ForecastResult:
    started = time.perf_counter()
    torch.set_float32_matmul_precision("high")

    train_scaler = PositiveMeanScaler.fit(splits.train)
    train_series = make_darts_series(train_scaler.transform(splits.train))
    validation_series = make_darts_series(
        train_scaler.transform(splits.validation)
    )

    accelerator, devices = "cpu", "auto"
    accelerator_fallback = (
        "CPU selected after the MPS backend exited with code 139 during the initial fit."
        if torch.backends.mps.is_available()
        else None
    )
    validation_model, tracker = _fit_validation_model(
        train_series, validation_series, accelerator, devices
    )

    validation_prediction = validation_model.predict(
        len(splits.validation),
        series=train_series,
        num_samples=200,
        verbose=False,
    )
    validation_quantiles = darts_samples_to_quantiles(
        validation_prediction, train_scaler, splits.validation.index
    )
    external_validation_mape = mean_absolute_percentage_error(
        splits.validation, validation_quantiles[0.5]
    )
    selected_epochs = max(1, tracker.best_epoch + 1)

    final_scaler = PositiveMeanScaler.fit(splits.train_validation)
    final_series = make_darts_series(
        final_scaler.transform(splits.train_validation)
    )
    final_model = _make_model(selected_epochs, accelerator, devices, [])
    final_model.fit(
        final_series,
        dataloader_kwargs={"num_workers": 0},
        verbose=False,
    )
    prediction = final_model.predict(
        len(splits.test),
        series=final_series,
        num_samples=N_SAMPLES,
        verbose=False,
    )
    quantiles = darts_samples_to_quantiles(prediction, final_scaler, splits.test.index)

    hyperparameters: dict[str, Any] = {
        "input_chunk_length": INPUT_CHUNK_LENGTH,
        "output_chunk_length": OUTPUT_CHUNK_LENGTH,
        "generic_architecture": True,
        "num_stacks": 30,
        "num_blocks": 1,
        "num_layers": 4,
        "layer_widths": 256,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "selected_epochs": selected_epochs,
        "early_stopping_monitor": MONITOR,
        "internal_best_validation_mape_pct": tracker.best_mape * 100.0,
        "validation_mape_pct": external_validation_mape,
        "likelihood": "quantile regression",
        "quantiles": list(QUANTILES),
        "prediction_samples": N_SAMPLES,
        "accelerator": accelerator,
        "accelerator_fallback": accelerator_fallback,
    }
    return ForecastResult(
        name="nbeats",
        forecast=quantiles[0.5].rename("forecast"),
        runtime_seconds=time.perf_counter() - started,
        hyperparameters=hyperparameters,
        quantiles=quantiles,
    )
