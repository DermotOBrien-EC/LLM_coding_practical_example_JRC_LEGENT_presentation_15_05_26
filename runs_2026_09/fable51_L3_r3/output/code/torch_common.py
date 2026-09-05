"""Shared training loop for the two neural models (N-BEATS and TSMixer).

Both models are darts "torch forecasting models" and are trained the same
way: scale the series to 0-1, train on Train with early stopping on the
validation window, pick the candidate configuration with the best rolling
one-week validation MAPE, then retrain on Train + Validation for the epoch
count early stopping found, and forecast the test week. Both produce
quantile forecasts directly (quantile regression), so the point forecast is
the median and the intervals come from the outer quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torchmetrics import MeanAbsolutePercentageError

from common import (
    HORIZON,
    QUANTILES,
    SEED,
    ForecastResult,
    Timer,
    rolling_validation_mape,
    validation_origins,
)

MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 1024
MONITOR = "val_MeanAbsolutePercentageError"
# Apple-silicon GPU. TSMixer at width 128 costs about 30 s per epoch on the
# CPU and 11 s on the GPU; N-BEATS roughly halves as well. Seeds are fixed,
# but bit-for-bit repeatability across hardware is not guaranteed.
ACCELERATOR = "mps"

ModelFactory = Callable[[dict[str, Any], int, list[Callback]], TorchForecastingModel]


@dataclass
class MinMax:
    """Min-max scaling fitted on the training history only."""

    low: float
    high: float

    @classmethod
    def fit(cls, series: pd.Series) -> "MinMax":
        return cls(float(series.min()), float(series.max()))

    def transform(self, series: pd.Series) -> TimeSeries:
        scaled = (series - self.low) / (self.high - self.low)
        scaled.index = scaled.index.tz_localize(None)
        return TimeSeries.from_series(scaled.astype(np.float32))

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * (self.high - self.low) + self.low


class MetricHistory(Callback):
    """Records the monitored validation metric after every epoch."""

    def __init__(self) -> None:
        self.values: list[float] = []

    # on_validation_end runs after the module has logged its epoch metrics;
    # the earlier on_validation_epoch_end hook would see the previous epoch.
    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(MONITOR)
        if value is not None:
            self.values.append(float(value))


def trainer_kwargs(callbacks: list[Callback]) -> dict[str, Any]:
    return {
        "accelerator": ACCELERATOR,
        "callbacks": callbacks,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }


def common_model_kwargs(n_epochs: int, callbacks: list[Callback], lr: float) -> dict[str, Any]:
    """Arguments every darts torch model in this study shares."""
    return {
        "input_chunk_length": HORIZON,
        "output_chunk_length": HORIZON,
        "n_epochs": n_epochs,
        "batch_size": BATCH_SIZE,
        "likelihood": QuantileRegression(quantiles=list(QUANTILES)),
        "optimizer_kwargs": {"lr": lr},
        "torch_metrics": MeanAbsolutePercentageError(),
        "random_state": SEED,
        "pl_trainer_kwargs": trainer_kwargs(callbacks),
        "save_checkpoints": False,
        "force_reset": True,
    }


def quantile_frame(ts: TimeSeries, scaler: MinMax, name: str) -> pd.DataFrame:
    """Turn a darts quantile-parameter forecast into a MW-scale DataFrame."""
    values = scaler.inverse(ts.all_values()[:, :, 0])
    if values.shape[1] != len(QUANTILES):
        raise ValueError(f"{name}: expected {len(QUANTILES)} quantile columns, got {values.shape[1]}")
    values.sort(axis=1)
    index = pd.DatetimeIndex(ts.time_index).tz_localize("UTC")
    return pd.DataFrame(values, index=index, columns=list(QUANTILES))


def rolling_validation_forecasts(
    model: TorchForecastingModel, scaler: MinMax, history: pd.Series, val: pd.Series, horizon: int
) -> list[pd.Series]:
    """Median forecasts for each validation week, without retraining."""
    forecasts = model.historical_forecasts(
        series=scaler.transform(history),
        start=val.index[0].tz_localize(None),
        forecast_horizon=horizon,
        stride=horizon,
        retrain=False,
        last_points_only=False,
        predict_likelihood_parameters=True,
        verbose=False,
    )
    return [quantile_frame(fc, scaler, model.__class__.__name__)[0.5] for fc in forecasts]


def select_refit_forecast(
    name: str,
    candidates: list[dict[str, Any]],
    factory: ModelFactory,
    train: pd.Series,
    val: pd.Series,
    horizon: int = HORIZON,
) -> ForecastResult:
    torch.set_num_threads(8)
    with Timer() as timer:
        origins = validation_origins(val, horizon)
        history = pd.concat([train, val])
        # The validation series for early stopping starts one week before
        # the validation window so the first validation week is scored too.
        train_scaler = MinMax.fit(train)
        train_ts = train_scaler.transform(train)
        val_ts = train_scaler.transform(pd.concat([train.iloc[-HORIZON:], val]))

        scores: list[tuple[float, dict[str, Any], int]] = []
        for params in candidates:
            history_cb = MetricHistory()
            stopper = EarlyStopping(monitor=MONITOR, patience=PATIENCE, min_delta=1e-4, mode="min")
            model = factory(params, MAX_EPOCHS, [history_cb, stopper])
            model.fit(train_ts, val_series=val_ts, verbose=False)
            best_epoch = int(np.argmin(history_cb.values)) + 1
            forecasts = rolling_validation_forecasts(model, train_scaler, history, val, horizon)
            score = rolling_validation_mape(val, forecasts, origins)
            scores.append((score, params, best_epoch))
            print(
                f"  {name} {params}: validation MAPE {score:.2f} % "
                f"(best epoch {best_epoch} of {len(history_cb.values)})",
                flush=True,
            )
        best_score, best_params, best_epoch = min(scores, key=lambda s: s[0])

        # Refit on Train + Validation for the epoch count early stopping chose.
        full_scaler = MinMax.fit(history)
        model = factory(best_params, best_epoch, [])
        model.fit(full_scaler.transform(history), verbose=False)
        forecast = model.predict(horizon, predict_likelihood_parameters=True, verbose=False)
        frame = quantile_frame(forecast, full_scaler, name)

    return ForecastResult(
        name=name,
        point=frame[0.5].rename(name),
        quantiles={tau: frame[tau] for tau in QUANTILES},
        runtime_seconds=timer.seconds,
        hyperparameters={
            **best_params,
            "input_chunk_length": HORIZON,
            "output_chunk_length": HORIZON,
            "epochs_trained": best_epoch,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "likelihood": f"quantile regression {list(QUANTILES)}",
            "scaling": "min-max on the fitting history",
            "candidates_scored": len(candidates),
        },
        validation_mape_pct=best_score,
        extras={"validation_scores": scores},
    )
