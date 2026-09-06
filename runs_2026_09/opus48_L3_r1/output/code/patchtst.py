"""Transformer forecaster (substituting for PatchTST).

The prompt asks for PatchTST, a transformer built for time series. The installed
darts version (0.41.0) does not expose ``PatchTSTModel``, so we substitute the
vanilla ``darts.models.TransformerModel``: a genuine encoder-decoder transformer
applied to the univariate load series, which fills the same role in the bake-off
(a self-attention deep model to sit next to the MLP-based N-BEATS). This
substitution is recorded in transcript.md.

Like N-BEATS it reads one week of history and emits the next week directly, uses
no hand-built features, and gets its prediction intervals from a
quantile-regression head. We shrink the default transformer (smaller model
width and fewer layers) so 30 epochs finish quickly; the shape of the study does
not need a giant model, only a representative transformer.
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
from darts.models import TransformerModel                       # noqa: E402
from darts.utils.likelihood_models import QuantileRegression    # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping           # noqa: E402

QUANTILE_LEVELS: list[float] = [0.025, 0.1, 0.5, 0.9, 0.975]
INPUT_CHUNK = 168
OUTPUT_CHUNK = 168
MAX_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 512
NUM_SAMPLES = 500
ACCELERATOR = "mps" if torch.backends.mps.is_available() else "cpu"

# A deliberately compact transformer: enough self-attention to be representative
# without the default model's per-epoch cost.
_ARCH = dict(
    d_model=64,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    dropout=0.1,
)


def _to_ts(series: pd.Series) -> TimeSeries:
    return TimeSeries.from_series(series, freq="h").astype(np.float32)


def _new_model(n_epochs: int, callbacks: list | None) -> TransformerModel:
    return TransformerModel(
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
        **_ARCH,
    )


def run(splits: c.Splits) -> c.ModelResult:
    start = time.time()

    scaler_tune = Scaler()
    train_s = scaler_tune.fit_transform(_to_ts(splits.train))
    val_s = scaler_tune.transform(_to_ts(splits.val))
    stopper = EarlyStopping(monitor="val_loss", patience=PATIENCE, min_delta=1e-4, mode="min")
    tuner = _new_model(MAX_EPOCHS, callbacks=[stopper])
    tuner.fit(train_s, val_series=val_s, verbose=False)
    epochs_used = int(tuner.trainer.current_epoch)
    epochs_used = max(1, min(epochs_used, MAX_EPOCHS))

    scaler = Scaler()
    trainval_s = scaler.fit_transform(_to_ts(splits.trainval))
    model = _new_model(epochs_used, callbacks=None)
    model.fit(trainval_s, verbose=False)

    sampled = scaler.inverse_transform(model.predict(len(splits.test), num_samples=NUM_SAMPLES))
    point = pd.Series(sampled.quantile(0.5).values().ravel(), index=splits.test.index, name="patchtst")
    quantiles: dict[float, pd.Series] = {}
    for q in QUANTILE_LEVELS:
        quantiles[q] = pd.Series(sampled.quantile(q).values().ravel(), index=splits.test.index, name=f"patchtst_q{q}")

    return c.ModelResult(
        name="patchtst",
        point=point,
        runtime_s=time.time() - start,
        hyperparameters={
            "substituted_model": "darts.models.TransformerModel",
            "reason": "PatchTSTModel not available in darts 0.41.0",
            "input_chunk_length": INPUT_CHUNK,
            "output_chunk_length": OUTPUT_CHUNK,
            "max_epochs": MAX_EPOCHS,
            "epochs_used_after_early_stopping": epochs_used,
            "batch_size": BATCH_SIZE,
            "accelerator": ACCELERATOR,
            "num_samples": NUM_SAMPLES,
            **_ARCH,
        },
        quantiles=quantiles,
    )
