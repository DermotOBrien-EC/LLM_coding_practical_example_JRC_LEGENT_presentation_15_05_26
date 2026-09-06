"""PatchTST slot, served by TSMixer.

The study asks for PatchTST, a transformer-based forecaster. The installed darts
version (0.41.0) does not expose `PatchTSTModel`, so we use the model the prompt
names as the first fallback, `TSMixerModel`. TSMixer is a modern all-MLP mixing
architecture rather than a transformer, so this is a substitution, recorded here
and in `transcript.md`. It plays the same role in the bake-off: a recent deep
sequence model fed only the raw load history, with the same one-week-in,
one-week-out setup as N-BEATS, so the two deep models are compared fairly.

Everything else mirrors the N-BEATS module: scale the load, choose epochs by
early stopping on the validation window, refit on train+validation, and read
prediction intervals from a Gaussian likelihood head.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TSMixerModel
from darts.utils.likelihood_models import GaussianLikelihood
from pytorch_lightning.callbacks import EarlyStopping

import common as c

INPUT_LEN = 168
OUTPUT_LEN = 168
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 1024
N_SAMPLES = 500

# The darts model actually used, recorded for transparency.
BACKEND_MODEL = "TSMixerModel"
SUBSTITUTION_NOTE = (
    "PatchTSTModel is not available in darts 0.41.0; TSMixerModel is used as the "
    "prompt-named fallback. TSMixer is an MLP-mixing model, not a transformer."
)


def _to_ts(series: pd.Series) -> TimeSeries:
    return TimeSeries.from_series(series.asfreq("h"), freq="h").astype(np.float32)


def _trainer_kwargs(callbacks: list | None = None) -> dict:
    kwargs = {
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }
    if callbacks is not None:
        kwargs["callbacks"] = callbacks
    return kwargs


def _build(n_epochs: int, callbacks: list | None = None) -> TSMixerModel:
    return TSMixerModel(
        input_chunk_length=INPUT_LEN,
        output_chunk_length=OUTPUT_LEN,
        n_epochs=n_epochs,
        batch_size=BATCH_SIZE,
        random_state=c.SEED,
        likelihood=GaussianLikelihood(),
        pl_trainer_kwargs=_trainer_kwargs(callbacks),
    )


def run(series: pd.Series) -> c.ModelResult:
    """Choose epochs on validation, refit on train+val, forecast the test week."""
    start = time.perf_counter()
    torch.manual_seed(c.SEED)

    train_scaler = Scaler()
    train_scaled = train_scaler.fit_transform(_to_ts(c.train_series(series)))
    val_scaled = train_scaler.transform(_to_ts(c.val_series(series)))

    stopper = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
    select_model = _build(MAX_EPOCHS, callbacks=[stopper])
    select_model.fit(train_scaled, val_series=val_scaled, verbose=False)

    if stopper.stopped_epoch:
        chosen_epochs = max(3, stopper.stopped_epoch - PATIENCE + 1)
    else:
        chosen_epochs = MAX_EPOCHS

    tv_scaler = Scaler()
    tv_scaled = tv_scaler.fit_transform(_to_ts(c.train_plus_val_series(series)))
    final = _build(chosen_epochs)
    final.fit(tv_scaled, verbose=False)

    pred_scaled = final.predict(n=c.HORIZON, num_samples=N_SAMPLES)
    pred = tv_scaler.inverse_transform(pred_scaled)

    quantiles = {
        q: pred.quantile(q).values().flatten().astype(float)
        for q in c.WINNER_QUANTILES
    }
    point = quantiles[0.5]

    runtime = time.perf_counter() - start
    return c.ModelResult(
        name="patchtst",
        point=point,
        runtime_seconds=runtime,
        hyperparameters={
            "backend_model": BACKEND_MODEL,
            "substitution_note": SUBSTITUTION_NOTE,
            "input_chunk_length": INPUT_LEN,
            "output_chunk_length": OUTPUT_LEN,
            "max_epochs": MAX_EPOCHS,
            "chosen_epochs": int(chosen_epochs),
            "batch_size": BATCH_SIZE,
            "likelihood": "Gaussian",
            "num_samples": N_SAMPLES,
            "early_stopping_patience": PATIENCE,
            "selection_metric": "validation_loss",
        },
        quantiles=quantiles,
    )
