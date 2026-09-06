from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torchmetrics.regression import MeanAbsolutePercentageError

from code.common import DataSplits, ForecastResult, make_utc_series

SCALE_MW = 100_000.0
QUANTILES = [0.025, 0.1, 0.5, 0.9, 0.975]
MAX_EPOCHS = 8
TRAINING_STRIDE_HOURS = 336


class BestValidationMape(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.best_score = float("inf")
        self.best_epoch = 0

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get("val_MeanAbsolutePercentageError")
        if metric is None:
            return
        score = float(metric.detach().cpu())
        if score < self.best_score:
            self.best_score = score
            self.best_epoch = int(trainer.current_epoch)


def _to_darts(series: pd.Series) -> TimeSeries:
    naive_utc = series.index.tz_convert("UTC").tz_localize(None)
    scaled = (series.to_numpy(dtype="float32") / SCALE_MW).reshape(-1, 1)
    return TimeSeries.from_times_and_values(
        naive_utc,
        scaled,
        columns=["load_scaled"],
        fill_missing_dates=False,
        freq="h",
    )


def _trainer_kwargs(callbacks: list[Callback] | None = None) -> dict[str, Any]:
    accelerator = "cpu"
    return {
        "accelerator": accelerator,
        "devices": 1,
        "callbacks": callbacks or [],
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "deterministic": "warn",
        "precision": "32-true",
        "num_sanity_val_steps": 0,
        "gradient_clip_val": 1.0,
    }


def _build(n_epochs: int, callbacks: list[Callback] | None = None) -> NBEATSModel:
    return NBEATSModel(
        input_chunk_length=168,
        output_chunk_length=168,
        n_epochs=n_epochs,
        batch_size=8,
        likelihood=QuantileRegression(quantiles=QUANTILES),
        torch_metrics=MeanAbsolutePercentageError(),
        optimizer_kwargs={"lr": 1e-3},
        random_state=2020,
        save_checkpoints=False,
        force_reset=True,
        show_warnings=False,
        pl_trainer_kwargs=_trainer_kwargs(callbacks),
    )


def _quantile_series(
    samples: TimeSeries,
    test_index: pd.DatetimeIndex,
) -> dict[float, pd.Series]:
    output: dict[float, pd.Series] = {}
    for quantile in QUANTILES:
        values = samples.quantile(quantile).univariate_values() * SCALE_MW
        output[quantile] = make_utc_series(values, test_index, f"nbeats_q{quantile}")
    return output


def run(splits: DataSplits) -> ForecastResult:
    started = perf_counter()
    pl.seed_everything(2020, workers=True)
    train = _to_darts(splits.train)
    validation = _to_darts(splits.validation)
    train_validation = _to_darts(splits.train_validation)

    tracker = BestValidationMape()
    stopper = EarlyStopping(
        monitor="val_MeanAbsolutePercentageError",
        mode="min",
        patience=2,
        min_delta=0.0001,
        check_finite=True,
    )
    selection_model = _build(MAX_EPOCHS, callbacks=[tracker, stopper])
    selection_model.fit(
        train,
        val_series=validation,
        verbose=False,
        stride=TRAINING_STRIDE_HOURS,
        dataloader_kwargs={"num_workers": 0, "pin_memory": False},
    )
    selected_epochs = max(1, tracker.best_epoch + 1)
    if not np.isfinite(tracker.best_score):
        raise RuntimeError("N-BEATS validation MAPE was not logged")

    pl.seed_everything(2020, workers=True)
    final_model = _build(selected_epochs)
    final_model.fit(
        train_validation,
        verbose=False,
        stride=TRAINING_STRIDE_HOURS,
        dataloader_kwargs={"num_workers": 0, "pin_memory": False},
    )
    samples = final_model.predict(
        n=len(splits.test),
        series=train_validation,
        num_samples=500,
        verbose=False,
        random_state=2020,
        dataloader_kwargs={"num_workers": 0, "pin_memory": False},
    )
    quantiles = _quantile_series(samples, splits.test.index)

    hyperparameters = {
        "input_chunk_length": 168,
        "output_chunk_length": 168,
        "architecture": "default generic N-BEATS stacks",
        "likelihood": "quantile_regression",
        "quantiles": QUANTILES,
        "max_epochs": MAX_EPOCHS,
        "selected_epochs": selected_epochs,
        "early_stopping_metric": "validation MAPE",
        "early_stopping_patience": 2,
        "training_stride_hours": TRAINING_STRIDE_HOURS,
        "batch_size": 8,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "accelerator": _trainer_kwargs()["accelerator"],
        "probabilistic_samples": 500,
    }
    return ForecastResult(
        name="nbeats",
        forecast=quantiles[0.5].rename("nbeats"),
        runtime_seconds=perf_counter() - started,
        hyperparameters=hyperparameters,
        quantiles=quantiles,
        validation_mape_pct=tracker.best_score * 100.0,
    )
