"""Patch-style univariate forecaster via darts.

The prompt asks for PatchTST. The installed darts version (0.41.0) does
not expose `PatchTSTModel`; the prompt explicitly accepts
`darts.models.TSMixerModel` as the alternative, so we use TSMixer here.
TSMixer is a recent MLP-mixer-style architecture in darts's catalogue:
it interleaves time-mixing and feature-mixing layers and is designed for
long-horizon forecasting on the same `input_chunk_length` / `output_chunk_length`
contract as PatchTST. The substitution is logged in `transcript.md`.

The training recipe mirrors `nbeats.py` so the two deep models compete on
an equal footing: same context length, same horizon, same scaler, same
likelihood, same epoch budget.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TSMixerModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.callbacks import EarlyStopping

from code.common import N_TEST, RANDOM_SEED, ModelForecast, mape

for name in ("pytorch_lightning", "lightning.pytorch", "darts"):
    logging.getLogger(name).setLevel(logging.WARNING)


QUANTILES: list[float] = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]

INPUT_CHUNK: int = 168
OUTPUT_CHUNK: int = 168
MAX_EPOCHS: int = 15  # match N-BEATS for fair comparison; ~4 min per fit on CPU


class _EpochPrinter(pl_callbacks.Callback):
    """One line of progress per epoch — keeps the orchestrator log alive."""

    def __init__(self, tag: str) -> None:
        self._tag = tag
        self._t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module) -> None:  # type: ignore[override]
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()
                   if isinstance(v, (int, float)) or hasattr(v, "item")}
        train_loss = metrics.get("train_loss", float("nan"))
        val_loss = metrics.get("val_loss", float("nan"))
        elapsed = time.perf_counter() - self._t0
        print(
            f"      {self._tag} epoch {trainer.current_epoch + 1}/{trainer.max_epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} elapsed={elapsed:.1f}s",
            flush=True,
        )


def _build_model(epochs: int, *, validation: bool, tag: str) -> TSMixerModel:
    cbs: list = [_EpochPrinter(tag)]
    if validation:
        cbs.append(EarlyStopping(monitor="val_loss", patience=5, mode="min"))
    return TSMixerModel(
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=OUTPUT_CHUNK,
        n_epochs=epochs,
        batch_size=128,
        likelihood=QuantileRegression(quantiles=QUANTILES),
        random_state=RANDOM_SEED,
        pl_trainer_kwargs={
            "callbacks": cbs,
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
        },
        save_checkpoints=False,
        force_reset=True,
    )


def run_patchtst(train: pd.Series, val: pd.Series, trainval: pd.Series) -> ModelForecast:
    started = time.perf_counter()
    torch.manual_seed(RANDOM_SEED)

    train_ts = TimeSeries.from_series(train)
    val_ts = TimeSeries.from_series(val)
    trainval_ts = TimeSeries.from_series(trainval)

    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_ts)
    val_scaled = scaler.transform(val_ts)

    selector = _build_model(epochs=MAX_EPOCHS, validation=True, tag="TSMixer sel")
    selector.fit(series=train_scaled, val_series=val_scaled, verbose=False)
    epochs_used = int(getattr(selector.trainer, "current_epoch", MAX_EPOCHS))
    epochs_used = max(epochs_used, 5)
    val_pred = selector.predict(n=len(val), num_samples=200)
    val_pred = scaler.inverse_transform(val_pred)
    val_mape = mape(val.to_numpy(), val_pred.values().ravel())

    scaler_final = Scaler()
    trainval_scaled = scaler_final.fit_transform(trainval_ts)
    final = _build_model(epochs=epochs_used, validation=False, tag="TSMixer refit")
    final.fit(series=trainval_scaled, verbose=False)

    pred_scaled = final.predict(n=N_TEST, num_samples=500)
    pred = scaler_final.inverse_transform(pred_scaled)
    samples = pred.all_values(copy=False)[:, 0, :]
    point = samples.mean(axis=1)
    q10 = np.quantile(samples, 0.10, axis=1)
    q50 = np.quantile(samples, 0.50, axis=1)
    q90 = np.quantile(samples, 0.90, axis=1)
    lower80 = q10
    upper80 = q90
    lower95 = np.quantile(samples, 0.025, axis=1)
    upper95 = np.quantile(samples, 0.975, axis=1)

    return ModelForecast(
        name="patchtst",
        point=point,
        q10=q10,
        q50=q50,
        q90=q90,
        lower80=lower80,
        upper80=upper80,
        lower95=lower95,
        upper95=upper95,
        runtime_seconds=time.perf_counter() - started,
        hyperparameters={
            "substitute_for": "PatchTST",
            "darts_model_class": "TSMixerModel",
            "darts_version": "0.41.0",
            "input_chunk_length": INPUT_CHUNK,
            "output_chunk_length": OUTPUT_CHUNK,
            "epochs_used_refit": epochs_used,
            "max_epochs_selector": MAX_EPOCHS,
            "batch_size": 128,
            "quantiles": QUANTILES,
            "validation_mape_pct": val_mape,
            "scaler": "standard",
        },
    )
