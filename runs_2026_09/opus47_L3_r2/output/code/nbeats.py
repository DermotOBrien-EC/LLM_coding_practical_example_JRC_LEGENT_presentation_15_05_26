"""N-BEATS on the univariate load series via darts.

One week of context (168 h) predicts one week (168 h). The model is
trained on Train, early-stopped on the first 168 h of Validation, then
refit on Train + Validation with the same epoch budget as the early-stopped
run and used to forecast the test week.

Prediction intervals come from the residual distribution on the validation
window (an empirical Normal fit to residual std), since NBEATSModel is a
point forecaster by default.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import EarlyStopping

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel

from common import Splits, align, mape


warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

RANDOM_SEED = 42
INPUT_LEN = 168
OUTPUT_LEN = 168
MAX_EPOCHS = 20
BATCH_SIZE = 1024


def _to_ts(s: pd.Series) -> TimeSeries:
    s = s.copy()
    s.index = s.index.tz_convert("UTC").tz_localize(None)
    return TimeSeries.from_series(s, freq="h")


def _make_trainer_kwargs(patience: int) -> dict:
    """Trainer knobs shared by validation and refit runs."""
    early = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
    return {
        "accelerator": "cpu",
        "callbacks": [early],
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }


def _build_model(max_epochs: int, patience: int, use_val: bool) -> NBEATSModel:
    return NBEATSModel(
        input_chunk_length=INPUT_LEN,
        output_chunk_length=OUTPUT_LEN,
        n_epochs=max_epochs,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_SEED,
        pl_trainer_kwargs=_make_trainer_kwargs(patience),
        save_checkpoints=False,
        model_name=f"nbeats_{'val' if use_val else 'refit'}",
        force_reset=True,
    )


def run(splits: Splits) -> dict[str, object]:
    """Train with early stopping, refit on train+val, forecast test."""
    scaler = Scaler()
    train_ts = _to_ts(splits.train)
    val_ts = _to_ts(splits.val)
    trainval_ts = _to_ts(splits.trainval)
    test_ts = _to_ts(splits.test)

    # Fit the scaler on train only, apply to all series.
    scaler.fit(train_ts)
    train_s = scaler.transform(train_ts)
    val_s = scaler.transform(val_ts)
    trainval_s = scaler.transform(trainval_ts)

    # Early-stop the training-only run on the first 168 h of val to pick epochs.
    model_val = _build_model(max_epochs=MAX_EPOCHS, patience=3, use_val=True)
    model_val.fit(train_s, val_series=val_s, verbose=False)
    chosen_epochs = getattr(model_val, "epochs_trained", MAX_EPOCHS)

    # Score on the first 168 h of val for the record and residual sigma.
    val_pred_s = model_val.predict(n=min(168, len(val_s)), series=train_s)
    val_pred = scaler.inverse_transform(val_pred_s).values(copy=False).flatten()
    val_actual = splits.val.iloc[:len(val_pred)].values
    val_mape_pct = mape(val_actual, val_pred)
    residual_sigma = float(np.std(val_actual - val_pred, ddof=1))

    # Refit on train + validation, using the epoch budget the early-stopper
    # settled on (or the cap if it never triggered).
    refit_epochs = max(5, min(int(chosen_epochs) or MAX_EPOCHS, MAX_EPOCHS))
    model_final = NBEATSModel(
        input_chunk_length=INPUT_LEN,
        output_chunk_length=OUTPUT_LEN,
        n_epochs=refit_epochs,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_SEED,
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
        },
        save_checkpoints=False,
        model_name="nbeats_final",
        force_reset=True,
    )
    model_final.fit(trainval_s, verbose=False)

    pred_s = model_final.predict(n=len(splits.test), series=trainval_s)
    pred = scaler.inverse_transform(pred_s).values(copy=False).flatten()

    from scipy.stats import norm
    forecast = align(pred, splits.test.index)
    lower_80 = align(pred + norm.ppf(0.10) * residual_sigma, splits.test.index)
    upper_80 = align(pred + norm.ppf(0.90) * residual_sigma, splits.test.index)
    lower_95 = align(pred + norm.ppf(0.025) * residual_sigma, splits.test.index)
    upper_95 = align(pred + norm.ppf(0.975) * residual_sigma, splits.test.index)
    q10 = align(pred + norm.ppf(0.10) * residual_sigma, splits.test.index)
    q50 = forecast
    q90 = align(pred + norm.ppf(0.90) * residual_sigma, splits.test.index)

    return {
        "name": "nbeats",
        "forecast": forecast,
        "lower_80": lower_80,
        "upper_80": upper_80,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "quantiles": {"0.1": q10, "0.5": q50, "0.9": q90},
        "hyperparameters": {
            "input_chunk_length": INPUT_LEN,
            "output_chunk_length": OUTPUT_LEN,
            "max_epochs": MAX_EPOCHS,
            "refit_epochs": refit_epochs,
            "batch_size": BATCH_SIZE,
            "random_state": RANDOM_SEED,
            "validation_mape_pct": val_mape_pct,
            "residual_sigma_mw": residual_sigma,
            "interval_method": "empirical_normal_from_val_residuals",
        },
    }
