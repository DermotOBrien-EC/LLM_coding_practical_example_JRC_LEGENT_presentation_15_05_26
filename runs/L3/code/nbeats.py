"""N-BEATS univariate forecaster via darts.

Plain language: N-BEATS is a stack of MLP blocks where each block predicts
a piece of the forecast and subtracts it from a "backcast" of recent
history, so subsequent blocks only have to model what is left. We use
the default architecture, scale the load to mean-0 / std-1 before fitting
(otherwise the network's weights waste capacity on absolute magnitude),
and add a `QuantileRegression` likelihood so the model emits prediction
intervals directly, not just point forecasts.

Selection on the validation window is done via early stopping on the
validation loss. Once the early-stopped run gives us a sensible epoch
count, we refit on Train + Validation combined for the same number of
epochs without early stopping (no held-out signal left to stop on), and
forecast the 168 test hours.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.callbacks import EarlyStopping

from code.common import N_TEST, RANDOM_SEED, ModelForecast, mape

# Silence Lightning / Darts info spam so the orchestrator output stays
# readable. We keep WARNING+ so anything genuinely bad still surfaces.
for name in ("pytorch_lightning", "lightning.pytorch", "darts"):
    logging.getLogger(name).setLevel(logging.WARNING)


QUANTILES: list[float] = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]

INPUT_CHUNK: int = 168
OUTPUT_CHUNK: int = 168
MAX_EPOCHS: int = 15  # Apple Silicon CPU budget: 15 epochs ≈ 4.5 min per fit


class _EpochPrinter(pl_callbacks.Callback):
    """One line of progress per epoch so the orchestrator log stays alive."""

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


def _build_model(epochs: int, *, validation: bool, tag: str) -> NBEATSModel:
    cbs: list = [_EpochPrinter(tag)]
    if validation:
        cbs.append(EarlyStopping(monitor="val_loss", patience=5, mode="min"))
    return NBEATSModel(
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


def run_nbeats(train: pd.Series, val: pd.Series, trainval: pd.Series) -> ModelForecast:
    started = time.perf_counter()
    torch.manual_seed(RANDOM_SEED)

    train_ts = TimeSeries.from_series(train)
    val_ts = TimeSeries.from_series(val)
    trainval_ts = TimeSeries.from_series(trainval)

    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_ts)
    val_scaled = scaler.transform(val_ts)

    # Phase 1: train with early stopping on val to find a sensible epoch count.
    selector = _build_model(epochs=MAX_EPOCHS, validation=True, tag="N-BEATS sel")
    selector.fit(series=train_scaled, val_series=val_scaled, verbose=False)
    epochs_used = int(getattr(selector.trainer, "current_epoch", MAX_EPOCHS))  # 0-indexed
    epochs_used = max(epochs_used, 5)  # never refit for fewer than a handful
    val_pred = selector.predict(n=len(val), num_samples=200)
    val_pred = scaler.inverse_transform(val_pred)
    val_mape = mape(val.to_numpy(), val_pred.values().ravel())

    # Phase 2: refit on train+val for the same epoch count, no validation.
    scaler_final = Scaler()
    trainval_scaled = scaler_final.fit_transform(trainval_ts)
    final = _build_model(epochs=epochs_used, validation=False, tag="N-BEATS refit")
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
        name="nbeats",
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
