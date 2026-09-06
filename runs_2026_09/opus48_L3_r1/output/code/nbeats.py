"""N-BEATS: a deep neural forecaster built from stacked fully-connected blocks.

N-BEATS learns temporal structure purely from the numbers, with no hand-built
calendar or lag features. It reads a fixed window of recent history (here one
week, 168 hours) and outputs the next week (168 hours) in one shot. The network
is a tall stack of simple blocks that each try to explain part of the signal
and pass the leftover to the next block.

We follow the prompt: the default stack configuration, one week in, one week
out, at most 30 training passes with early stopping watching the validation
loss. The "hyperparameter" we actually select is how many passes to keep: we
train on Train while watching Validation, note where the validation loss stops
improving, then retrain on Train+Val for that many passes. Prediction intervals
come from a quantile-regression output head sampled at forecast time.
"""

from __future__ import annotations

import logging
import time
import warnings

import numpy as np
import pandas as pd

import common as c

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

import torch                                                    # noqa: E402
from darts import TimeSeries                                    # noqa: E402
from darts.dataprocessing.transformers import Scaler            # noqa: E402
from darts.models import NBEATSModel                            # noqa: E402
from darts.utils.likelihood_models import QuantileRegression    # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping           # noqa: E402

QUANTILE_LEVELS: list[float] = [0.025, 0.1, 0.5, 0.9, 0.975]
INPUT_CHUNK = 168
OUTPUT_CHUNK = 168
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 512
NUM_SAMPLES = 500
# MPS (Apple GPU) cannot do float64, and it is markedly faster here, so the
# torch models run in float32 on MPS when available, otherwise on CPU.
ACCELERATOR = "mps" if torch.backends.mps.is_available() else "cpu"


def _to_ts(series: pd.Series) -> TimeSeries:
    return TimeSeries.from_series(series, freq="h").astype(np.float32)


def _new_model(n_epochs: int, callbacks: list | None) -> NBEATSModel:
    return NBEATSModel(
        input_chunk_length=INPUT_CHUNK,
        output_chunk_length=OUTPUT_CHUNK,
        n_epochs=n_epochs,
        batch_size=BATCH_SIZE,
        random_state=c.SEED,
        likelihood=QuantileRegression(quantiles=QUANTILE_LEVELS),
        pl_trainer_kwargs={
            "accelerator": ACCELERATOR,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
            "callbacks": callbacks or [],
        },
    )


def run(splits: c.Splits) -> c.ModelResult:
    start = time.time()

    # ---- how many epochs? train on Train, watch Validation ------------
    scaler_tune = Scaler()
    train_s = scaler_tune.fit_transform(_to_ts(splits.train))
    val_s = scaler_tune.transform(_to_ts(splits.val))
    stopper = EarlyStopping(monitor="val_loss", patience=PATIENCE, min_delta=1e-4, mode="min")
    tuner = _new_model(MAX_EPOCHS, callbacks=[stopper])
    tuner.fit(train_s, val_series=val_s, verbose=False)
    epochs_used = int(tuner.trainer.current_epoch)  # epochs actually completed
    epochs_used = max(1, min(epochs_used, MAX_EPOCHS))

    # ---- refit on Train+Val for the chosen number of epochs -----------
    scaler = Scaler()
    trainval_s = scaler.fit_transform(_to_ts(splits.trainval))
    model = _new_model(epochs_used, callbacks=None)
    model.fit(trainval_s, verbose=False)

    sampled = scaler.inverse_transform(model.predict(len(splits.test), num_samples=NUM_SAMPLES))
    point = pd.Series(sampled.quantile(0.5).values().ravel(), index=splits.test.index, name="nbeats")
    quantiles: dict[float, pd.Series] = {}
    for q in QUANTILE_LEVELS:
        quantiles[q] = pd.Series(sampled.quantile(q).values().ravel(), index=splits.test.index, name=f"nbeats_q{q}")

    return c.ModelResult(
        name="nbeats",
        point=point,
        runtime_s=time.time() - start,
        hyperparameters={
            "input_chunk_length": INPUT_CHUNK,
            "output_chunk_length": OUTPUT_CHUNK,
            "max_epochs": MAX_EPOCHS,
            "epochs_used_after_early_stopping": epochs_used,
            "batch_size": BATCH_SIZE,
            "accelerator": ACCELERATOR,
            "num_samples": NUM_SAMPLES,
        },
        quantiles=quantiles,
    )
