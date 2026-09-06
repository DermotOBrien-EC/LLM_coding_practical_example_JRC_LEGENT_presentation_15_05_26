"""TSMixer, in place of PatchTST.

The installed darts version (0.41.0) does not expose PatchTSTModel, so the
prompt's fallback applies: we use darts.models.TSMixerModel, another modern
univariate forecaster. This substitution is recorded in transcript.md.

Same input/output chunk lengths as N-BEATS (168 h context, 168 h horizon) and
~30 epochs.
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
from darts.models import TSMixerModel

from common import RANDOM_SEED, Split, to_hourly_naive

for name in ("pytorch_lightning", "lightning.pytorch", "pytorch_lightning.utilities"):
    logging.getLogger(name).setLevel(logging.ERROR)

torch.set_float32_matmul_precision("medium")


@dataclass
class TSMixerResult:
    forecast: pd.Series
    lower_80: pd.Series
    upper_80: pd.Series
    lower_95: pd.Series
    upper_95: pd.Series
    runtime_seconds: float
    hyperparameters: dict[str, object]


def _series(series: pd.Series) -> TimeSeries:
    return TimeSeries.from_series(to_hourly_naive(series))


def _build_model(epochs: int) -> TSMixerModel:
    return TSMixerModel(
        input_chunk_length=168,
        output_chunk_length=168,
        n_epochs=epochs,
        batch_size=512,
        random_state=RANDOM_SEED,
        hidden_size=64,
        ff_size=64,
        num_blocks=2,
        activation="ReLU",
        dropout=0.1,
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
        },
    )


def run_tsmixer(split: Split) -> TSMixerResult:
    t0 = time.perf_counter()
    scaler = Scaler()
    train_ts = scaler.fit_transform(_series(split.train))
    val_ts = scaler.transform(_series(split.val))
    trainval_ts = scaler.transform(_series(split.train_plus_val))

    max_epochs = 15
    pre = _build_model(epochs=max_epochs)
    pre.fit(train_ts, val_series=val_ts, verbose=False)
    val_pred = scaler.inverse_transform(pre.predict(n=len(val_ts))).values().flatten()
    val_true = split.val.values.astype(float)
    val_mape = float(np.mean(np.abs((val_true - val_pred) / val_true)) * 100.0)

    model = _build_model(epochs=max_epochs)
    model.fit(trainval_ts, verbose=False)
    point = scaler.inverse_transform(model.predict(n=168)).values().flatten()

    # PIs via MC dropout.
    try:
        samples_scaled = model.predict(n=168, num_samples=100, mc_dropout=True)
        samples = scaler.inverse_transform(samples_scaled).all_values()[:, 0, :]
        q10 = np.quantile(samples, 0.10, axis=1)
        q90 = np.quantile(samples, 0.90, axis=1)
        q025 = np.quantile(samples, 0.025, axis=1)
        q975 = np.quantile(samples, 0.975, axis=1)
    except Exception:
        sigma = float(np.std(val_true - val_pred))
        q10 = point - 1.2815515655446004 * sigma
        q90 = point + 1.2815515655446004 * sigma
        q025 = point - 1.9599639845400545 * sigma
        q975 = point + 1.9599639845400545 * sigma

    idx = split.test.index
    forecast = pd.Series(point, index=idx, name="patchtst")
    lower_80 = pd.Series(q10, index=idx, name="q10")
    upper_80 = pd.Series(q90, index=idx, name="q90")
    lower_95 = pd.Series(q025, index=idx, name="q025")
    upper_95 = pd.Series(q975, index=idx, name="q975")

    runtime = time.perf_counter() - t0
    hp = {
        "substituted_for": "PatchTSTModel (not exposed in installed darts 0.41.0)",
        "model_class": "TSMixerModel",
        "input_chunk_length": 168,
        "output_chunk_length": 168,
        "n_epochs": max_epochs,
        "batch_size": 512,
        "hidden_size": 64,
        "ff_size": 64,
        "num_blocks": 2,
        "dropout": 0.1,
        "validation_mape_pct": float(val_mape),
    }
    return TSMixerResult(
        forecast=forecast,
        lower_80=lower_80,
        upper_80=upper_80,
        lower_95=lower_95,
        upper_95=upper_95,
        runtime_seconds=runtime,
        hyperparameters=hp,
    )
