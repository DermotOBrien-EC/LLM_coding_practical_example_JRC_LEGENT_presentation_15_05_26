"""N-BEATS: a deep neural network built specifically for forecasting.

N-BEATS stacks many small fully-connected blocks that each try to explain part
of the signal and pass the leftover to the next block. It learns purely from the
shape of the past load, with no hand-built features and no calendar knowledge:
we give it one week of history (168 hours) and ask it to produce the next week
(168 hours) in a single shot. It is the first genuinely "deep learning" entry in
the study and a good test of whether raw pattern-learning beats engineered
features on a short horizon.

We scale the load to a friendly numeric range before training (neural networks
learn better that way) and undo the scaling afterwards. The number of training
epochs is chosen with early stopping on the validation window, then the model is
retrained on train+validation for that many epochs.

Prediction intervals come from a Gaussian likelihood head: the network predicts
a distribution per hour, which we sample to get quantiles.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel
from darts.utils.likelihood_models import GaussianLikelihood
from pytorch_lightning.callbacks import EarlyStopping

import common as c

INPUT_LEN = 168  # one week of context
OUTPUT_LEN = 168  # forecast one week at once
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 1024
N_SAMPLES = 500


def _to_ts(series: pd.Series) -> TimeSeries:
    return TimeSeries.from_series(series.asfreq("h"), freq="h").astype(np.float32)


def _trainer_kwargs(callbacks: list | None = None) -> dict:
    """Force CPU training and quiet, reproducible runs."""
    kwargs = {
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }
    if callbacks is not None:
        kwargs["callbacks"] = callbacks
    return kwargs


def _build(n_epochs: int, callbacks: list | None = None) -> NBEATSModel:
    return NBEATSModel(
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

    # --- Step 1: fit on train, early-stop on validation loss. ---------------
    train_scaler = Scaler()
    train_scaled = train_scaler.fit_transform(_to_ts(c.train_series(series)))
    val_scaled = train_scaler.transform(_to_ts(c.val_series(series)))

    stopper = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
    select_model = _build(MAX_EPOCHS, callbacks=[stopper])
    select_model.fit(train_scaled, val_series=val_scaled, verbose=False)

    # The best epoch is roughly `patience` steps before it stopped; if it never
    # triggered, all MAX_EPOCHS were useful.
    if stopper.stopped_epoch:
        chosen_epochs = max(3, stopper.stopped_epoch - PATIENCE + 1)
    else:
        chosen_epochs = MAX_EPOCHS

    # --- Step 2: refit on train+val for the chosen number of epochs. --------
    tv_scaler = Scaler()
    tv_scaled = tv_scaler.fit_transform(_to_ts(c.train_plus_val_series(series)))
    final = _build(chosen_epochs)
    final.fit(tv_scaled, verbose=False)

    # --- Step 3: probabilistic forecast of the test week. -------------------
    pred_scaled = final.predict(n=c.HORIZON, num_samples=N_SAMPLES)
    pred = tv_scaler.inverse_transform(pred_scaled)

    quantiles = {
        q: pred.quantile(q).values().flatten().astype(float)
        for q in c.WINNER_QUANTILES
    }
    point = quantiles[0.5]

    runtime = time.perf_counter() - start
    return c.ModelResult(
        name="nbeats",
        point=point,
        runtime_seconds=runtime,
        hyperparameters={
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
