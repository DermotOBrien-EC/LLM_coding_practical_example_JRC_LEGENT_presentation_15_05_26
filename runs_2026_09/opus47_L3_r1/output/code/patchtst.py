"""Transformer-family model via darts.

PatchTSTModel is not exported by darts 0.41.0 (checked at run start).
We substitute darts.models.TSMixerModel, another modern univariate deep
forecaster from the same family (patch-and-mix architecture). The
substitution is recorded in transcript.md.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from common import RANDOM_SEED, Splits


INPUT_LEN: int = 168
OUTPUT_LEN: int = 168
MAX_EPOCHS: int = 15  # early stopping usually halts around 6-10 on this data


def _to_ts(s: pd.Series):
    from darts import TimeSeries
    ser = s.copy()
    ser.index = ser.index.tz_convert("UTC").tz_localize(None)
    return TimeSeries.from_series(ser.astype(float), freq="h")


def _make_model(seed: int):
    from darts.models import TSMixerModel
    from pytorch_lightning.callbacks import EarlyStopping

    torch.manual_seed(seed)
    early = EarlyStopping(
        monitor="val_loss", patience=3, mode="min", min_delta=0.0
    )
    return TSMixerModel(
        input_chunk_length=INPUT_LEN,
        output_chunk_length=OUTPUT_LEN,
        n_epochs=MAX_EPOCHS,
        random_state=seed,
        batch_size=512,
        hidden_size=48,
        ff_size=48,
        num_blocks=2,
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "callbacks": [early],
            "log_every_n_steps": 50,
        },
    )


def forecast(splits: Splits) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    train_val_ts = _to_ts(splits.train_val)

    mu = float(splits.train.mean())
    sd = float(splits.train.std())
    def norm(ts):
        return (ts - mu) / sd
    def denorm_arr(a: np.ndarray) -> np.ndarray:
        return a * sd + mu

    # Single fit on train+val with a 3-week proxy val tail for early stopping.
    fit_series = norm(train_val_ts)
    tail_len = 3 * 7 * 24
    proxy_val = fit_series[-tail_len:]
    fit_head = fit_series[:-tail_len]

    m = _make_model(RANDOM_SEED)
    m.fit(series=fit_head, val_series=proxy_val, verbose=False)

    val_pred = m.predict(n=len(proxy_val), series=fit_head)
    val_pred_arr = denorm_arr(val_pred.values().reshape(-1))
    val_obs_arr = denorm_arr(proxy_val.values().reshape(-1))
    resid = val_obs_arr - val_pred_arr
    resid_sd = float(np.std(resid))

    pred_norm = m.predict(n=OUTPUT_LEN, series=fit_series)
    pred_arr = denorm_arr(pred_norm.values().reshape(-1))
    mean = pd.Series(pred_arr, index=splits.test.index, name="patchtst")
    z80 = 1.2816
    z95 = 1.9600
    intervals = pd.DataFrame({
        "lower_80": pred_arr - z80 * resid_sd,
        "upper_80": pred_arr + z80 * resid_sd,
        "lower_95": pred_arr - z95 * resid_sd,
        "upper_95": pred_arr + z95 * resid_sd,
    }, index=splits.test.index)
    hp = {
        "model_class": "TSMixerModel (substitute for PatchTST)",
        "input_chunk_length": INPUT_LEN,
        "output_chunk_length": OUTPUT_LEN,
        "max_epochs": MAX_EPOCHS,
        "batch_size": 64,
        "residual_sd_mw": resid_sd,
    }
    return mean, intervals, hp
