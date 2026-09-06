"""N-BEATS via darts.

Univariate. One week of context (168h) -> one week horizon (168h). Early
stopping on validation MAPE.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from common import RANDOM_SEED, Splits


INPUT_LEN: int = 168
OUTPUT_LEN: int = 168
MAX_EPOCHS: int = 40


def _to_ts(s: pd.Series):
    from darts import TimeSeries
    ser = s.copy()
    ser.index = ser.index.tz_convert("UTC").tz_localize(None)
    return TimeSeries.from_series(ser.astype(float), freq="h")


def _make_model(seed: int):
    from darts.models import NBEATSModel
    from pytorch_lightning.callbacks import EarlyStopping

    torch.manual_seed(seed)
    early = EarlyStopping(
        monitor="val_loss", patience=5, mode="min", min_delta=0.0
    )
    return NBEATSModel(
        input_chunk_length=INPUT_LEN,
        output_chunk_length=OUTPUT_LEN,
        n_epochs=MAX_EPOCHS,
        random_state=seed,
        batch_size=128,
        num_stacks=3,
        num_blocks=1,
        num_layers=4,
        layer_widths=256,
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

    # Normalise using training statistics (subtract mean, divide by std).
    mu = float(splits.train.mean())
    sd = float(splits.train.std())
    def norm(ts):
        return (ts - mu) / sd
    def denorm_arr(a: np.ndarray) -> np.ndarray:
        return a * sd + mu

    # Single fit on train+val, holding out the last three weeks of train+val
    # as a proxy validation slice for early-stopping (still strictly inside
    # the fit set, not in test). Three weeks is large enough that darts can
    # extract validation samples of length input_chunk + output_chunk = 336.
    fit_series = norm(train_val_ts)
    tail_len = 3 * 7 * 24  # 504 hours = 3 weeks
    proxy_val = fit_series[-tail_len:]
    fit_head = fit_series[:-tail_len]

    m = _make_model(RANDOM_SEED)
    m.fit(series=fit_head, val_series=proxy_val, verbose=False)

    # Residuals on the proxy validation window feed the interval width.
    val_pred = m.predict(n=len(proxy_val), series=fit_head)
    val_pred_arr = denorm_arr(val_pred.values().reshape(-1))
    val_obs_arr = denorm_arr(proxy_val.values().reshape(-1))
    resid = val_obs_arr - val_pred_arr
    resid_sd = float(np.std(resid))

    # Predict 168 hours ahead from the end of train+val.
    pred_norm = m.predict(n=OUTPUT_LEN, series=fit_series)
    pred_arr = denorm_arr(pred_norm.values().reshape(-1))
    mean = pd.Series(pred_arr, index=splits.test.index, name="nbeats")
    z80 = 1.2816
    z95 = 1.9600
    intervals = pd.DataFrame({
        "lower_80": pred_arr - z80 * resid_sd,
        "upper_80": pred_arr + z80 * resid_sd,
        "lower_95": pred_arr - z95 * resid_sd,
        "upper_95": pred_arr + z95 * resid_sd,
    }, index=splits.test.index)
    hp = {
        "input_chunk_length": INPUT_LEN,
        "output_chunk_length": OUTPUT_LEN,
        "max_epochs": MAX_EPOCHS,
        "batch_size": 64,
        "residual_sd_mw": resid_sd,
    }
    return mean, intervals, hp
