from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import TSMixerModel
from pytorch_lightning.callbacks import EarlyStopping

from common import (
    LOAD_COL,
    MODEL_LABELS,
    TEST_END,
    TEST_START,
    TIME_COL,
    TRAIN_END,
    VAL_END,
    VAL_START,
    ForecastResult,
    mape_pct,
)

CANDIDATES: list[dict[str, Any]] = [
    {"hidden_size": 16, "ff_size": 16, "num_blocks": 1, "dropout": 0.10, "n_epochs": 2},
    {"hidden_size": 32, "ff_size": 32, "num_blocks": 1, "dropout": 0.10, "n_epochs": 3},
]


def _to_timeseries(df: pd.DataFrame) -> TimeSeries:
    local = df[[TIME_COL, LOAD_COL]].copy()
    local[TIME_COL] = pd.to_datetime(local[TIME_COL]).dt.tz_localize(None)
    return TimeSeries.from_dataframe(local, time_col=TIME_COL, value_cols=LOAD_COL, fill_missing_dates=False, freq="h")


def _fit(train_series: TimeSeries, val_series: TimeSeries, params: dict[str, Any], scale: float) -> TSMixerModel:
    early_stopper = EarlyStopping(monitor="val_loss", patience=3, min_delta=1e-5, mode="min")
    model = TSMixerModel(
        input_chunk_length=168,
        output_chunk_length=168,
        random_state=42,
        batch_size=64,
        optimizer_kwargs={"lr": 1e-3},
        pl_trainer_kwargs={
            "accelerator": "cpu",
            "enable_checkpointing": False,
            "callbacks": [early_stopper],
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
        },
        **params,
    )
    model.fit(train_series / scale, val_series=val_series / scale, verbose=False, dataloader_kwargs={"pin_memory": False, "num_workers": 0})
    return model


def _predict(model: TSMixerModel, n: int, index: pd.DatetimeIndex, scale: float) -> pd.Series:
    pred = model.predict(n, verbose=False, dataloader_kwargs={"pin_memory": False, "num_workers": 0})
    return pd.Series(pred.values().ravel() * scale, index=index, name="forecast")


def _select_hyperparameters(train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, float | str]]]:
    train_ts = _to_timeseries(train)
    val_ts = _to_timeseries(val)
    val_actual = pd.Series(val[LOAD_COL].to_numpy(dtype=float), index=pd.DatetimeIndex(val[TIME_COL]))
    scale = float(train[LOAD_COL].mean())
    scores: list[dict[str, float | str]] = []
    best_score = np.inf
    best_params = CANDIDATES[0]
    for i, params in enumerate(CANDIDATES):
        model = _fit(train_ts, val_ts, params, scale)
        forecast = _predict(model, len(val), val_actual.index, scale)
        score = mape_pct(val_actual, forecast)
        scores.append({"candidate": str(i), "validation_mape_pct": float(score)})
        if score < best_score:
            best_score = score
            best_params = params
    return best_params, scores


def _residual_intervals(train_val: pd.DataFrame, params: dict[str, Any]) -> dict[str, float]:
    train_part = train_val.iloc[:-2_208].copy()
    holdout = train_val.iloc[-2_208:].copy()
    scale = float(train_part[LOAD_COL].mean())
    quick = _fit(_to_timeseries(train_part), _to_timeseries(holdout), params, scale)
    pred = _predict(quick, len(holdout), pd.DatetimeIndex(holdout[TIME_COL]), scale)
    residual = holdout[LOAD_COL].to_numpy(dtype=float) - pred.to_numpy(dtype=float)
    return {
        "q025": float(np.quantile(residual, 0.025)),
        "q10": float(np.quantile(residual, 0.10)),
        "q90": float(np.quantile(residual, 0.90)),
        "q975": float(np.quantile(residual, 0.975)),
    }


def run(df: pd.DataFrame) -> ForecastResult:
    start = time.perf_counter()
    train = df[df[TIME_COL] <= TRAIN_END].copy()
    val = df[(df[TIME_COL] >= VAL_START) & (df[TIME_COL] <= VAL_END)].copy()
    train_val = df[df[TIME_COL] <= VAL_END].copy()
    test = df[(df[TIME_COL] >= TEST_START) & (df[TIME_COL] <= TEST_END)].copy()
    test_index = pd.DatetimeIndex(test[TIME_COL])

    best_params, scores = _select_hyperparameters(train, val)
    scale = float(train_val[LOAD_COL].mean())
    model = _fit(_to_timeseries(train_val), _to_timeseries(train_val.iloc[-2_208:]), best_params, scale)
    forecast = _predict(model, len(test), test_index, scale)
    residual_q = _residual_intervals(train_val, best_params)

    q10 = (forecast + residual_q["q10"]).rename("q10")
    q50 = forecast.rename("q50")
    q90 = (forecast + residual_q["q90"]).rename("q90")
    lower80 = q10.rename("lower80")
    upper80 = q90.rename("upper80")
    lower95 = (forecast + residual_q["q025"]).rename("lower95")
    upper95 = (forecast + residual_q["q975"]).rename("upper95")

    return ForecastResult(
        name="patchtst",
        display_name=MODEL_LABELS["patchtst"],
        forecast=forecast,
        runtime_seconds=time.perf_counter() - start,
        hyperparameters={
            **best_params,
            "substitution": "PatchTSTModel was unavailable in Darts 0.41.0; used TSMixerModel",
            "validation_scores": scores,
            "interval_method": "validation-window residual quantiles",
        },
        q10=q10,
        q50=q50,
        q90=q90,
        lower80=lower80,
        upper80=upper80,
        lower95=lower95,
        upper95=upper95,
        fitted_model=model,
        diagnostics={"residual_quantiles": residual_q},
    )
