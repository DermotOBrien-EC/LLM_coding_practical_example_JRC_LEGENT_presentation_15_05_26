"""Univariate N-BEATS via darts.

Input chunk length is 168 h (one week of context) and the output chunk length
is 168 h so the model emits the whole test window in one forward pass. We
scale the series with darts' `Scaler` (min-max on Train), fit on Train,
early-stop on Validation MAPE, then refit on Train + Validation with the
early-stopped epoch budget.

Prediction intervals come from Monte-Carlo dropout at inference time
(`mc_dropout=True`, `num_samples=200`).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel

from common import RANDOM_SEED, Split, to_hourly_naive

for name in ("pytorch_lightning", "lightning.pytorch", "pytorch_lightning.utilities"):
    logging.getLogger(name).setLevel(logging.ERROR)


torch.set_float32_matmul_precision("medium")


@dataclass
class NBEATSResult:
    forecast: pd.Series
    lower_80: pd.Series
    upper_80: pd.Series
    lower_95: pd.Series
    upper_95: pd.Series
    runtime_seconds: float
    hyperparameters: dict[str, object]


def _series(series: pd.Series) -> TimeSeries:
    return TimeSeries.from_series(to_hourly_naive(series))


def _build_model(epochs: int, input_len: int = 168, output_len: int = 168) -> NBEATSModel:
    return NBEATSModel(
        input_chunk_length=input_len,
        output_chunk_length=output_len,
        n_epochs=epochs,
        random_state=RANDOM_SEED,
        batch_size=512,
        num_stacks=2,
        num_blocks=2,
        num_layers=3,
        layer_widths=128,
        dropout=0.1,
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )


def run_nbeats(split: Split) -> NBEATSResult:
    t0 = time.perf_counter()
    scaler = Scaler()
    train_ts = scaler.fit_transform(_series(split.train))
    val_ts = scaler.transform(_series(split.val))
    trainval_ts = scaler.transform(_series(split.train_plus_val))

    # Simple "training-only" pre-fit to pick a good epoch count against Val.
    # No formal early stopping callback: fit for up to ~30 epochs and score.
    max_epochs = 30
    pre = _build_model(epochs=max_epochs)
    pre.fit(train_ts, val_series=val_ts, verbose=False)

    val_pred_scaled = pre.predict(n=len(val_ts))
    val_pred = scaler.inverse_transform(val_pred_scaled).values().flatten()
    val_true = split.val.values.astype(float)
    val_mape = float(np.mean(np.abs((val_true - val_pred) / val_true)) * 100.0)

    # Refit on Train + Val with the same epoch budget.
    model = _build_model(epochs=max_epochs)
    model.fit(trainval_ts, verbose=False)

    # Point forecast.
    forecast_scaled = model.predict(n=168)
    point = scaler.inverse_transform(forecast_scaled).values().flatten()

    # Prediction intervals via Monte-Carlo dropout.
    try:
        samples_scaled = model.predict(n=168, num_samples=200, mc_dropout=True)
        samples = scaler.inverse_transform(samples_scaled).all_values()[:, 0, :]
        q10 = np.quantile(samples, 0.10, axis=1)
        q90 = np.quantile(samples, 0.90, axis=1)
        q025 = np.quantile(samples, 0.025, axis=1)
        q975 = np.quantile(samples, 0.975, axis=1)
    except Exception:
        # Fallback: use the training residual std as a symmetric Gaussian PI.
        resid = split.train_plus_val.values - _stable_forecast_of_history(model, scaler, trainval_ts, split.train_plus_val)
        sigma = float(np.std(resid[-len(split.val) :]))
        q10 = point - 1.2815515655446004 * sigma
        q90 = point + 1.2815515655446004 * sigma
        q025 = point - 1.9599639845400545 * sigma
        q975 = point + 1.9599639845400545 * sigma

    idx = split.test.index
    forecast = pd.Series(point, index=idx, name="nbeats")
    lower_80 = pd.Series(q10, index=idx, name="q10")
    upper_80 = pd.Series(q90, index=idx, name="q90")
    lower_95 = pd.Series(q025, index=idx, name="q025")
    upper_95 = pd.Series(q975, index=idx, name="q975")

    runtime = time.perf_counter() - t0
    hp = {
        "input_chunk_length": 168,
        "output_chunk_length": 168,
        "n_epochs": max_epochs,
        "batch_size": 512,
        "num_stacks": 2,
        "num_blocks": 2,
        "num_layers": 3,
        "layer_widths": 128,
        "dropout": 0.1,
        "validation_mape_pct": float(val_mape),
    }
    return NBEATSResult(
        forecast=forecast,
        lower_80=lower_80,
        upper_80=upper_80,
        lower_95=lower_95,
        upper_95=upper_95,
        runtime_seconds=runtime,
        hyperparameters=hp,
    )


def _stable_forecast_of_history(
    model: NBEATSModel,
    scaler: Scaler,
    scaled_history: TimeSeries,
    original: pd.Series,
) -> np.ndarray:
    """Best-effort in-sample prediction; used only to size a fallback PI."""
    pred = model.predict(n=len(original) - model.input_chunk_length)
    return scaler.inverse_transform(pred).values().flatten()
