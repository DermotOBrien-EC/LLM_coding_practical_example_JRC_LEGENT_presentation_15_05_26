from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

_MODULE_DIR = Path(__file__).resolve().parent
sys.path = [path for path in sys.path if Path(path or ".").resolve() != _MODULE_DIR]
from prophet import Prophet
sys.path.insert(0, str(_MODULE_DIR))

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
    {"seasonality_mode": "additive", "changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0},
    {"seasonality_mode": "multiplicative", "changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0},
]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    local = df[[TIME_COL, LOAD_COL]].copy()
    local[TIME_COL] = pd.to_datetime(local[TIME_COL]).dt.tz_localize(None)
    local = local.rename(columns={TIME_COL: "ds", LOAD_COL: "y"})
    return local


def _fit(frame: pd.DataFrame, params: dict[str, Any]) -> Prophet:
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.80,
        uncertainty_samples=1_000,
        **params,
    )
    model.add_country_holidays(country_name="DE")
    model.fit(_prep(frame))
    return model


def _predict(model: Prophet, n: int, index: pd.DatetimeIndex) -> pd.DataFrame:
    future = pd.DataFrame({"ds": pd.DatetimeIndex(index).tz_localize(None)})
    raw = model.predict(future)
    raw.index = index
    return raw


def _select_hyperparameters(train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, float | str]]]:
    val_actual = pd.Series(val[LOAD_COL].to_numpy(dtype=float), index=pd.DatetimeIndex(val[TIME_COL]))
    scores: list[dict[str, float | str]] = []
    best_score = float("inf")
    best_params = CANDIDATES[0]
    for i, params in enumerate(CANDIDATES):
        model = _fit(train, params)
        raw = _predict(model, len(val), val_actual.index)
        forecast = pd.Series(raw["yhat"].to_numpy(dtype=float), index=val_actual.index)
        score = mape_pct(val_actual, forecast)
        scores.append({"candidate": str(i), "validation_mape_pct": float(score), "seasonality_mode": params["seasonality_mode"]})
        if score < best_score:
            best_score = score
            best_params = params
    return best_params, scores


def run(df: pd.DataFrame) -> ForecastResult:
    start = time.perf_counter()
    train = df[df[TIME_COL] <= TRAIN_END].copy()
    val = df[(df[TIME_COL] >= VAL_START) & (df[TIME_COL] <= VAL_END)].copy()
    train_val = df[df[TIME_COL] <= VAL_END].copy()
    test = df[(df[TIME_COL] >= TEST_START) & (df[TIME_COL] <= TEST_END)].copy()
    test_index = pd.DatetimeIndex(test[TIME_COL])

    best_params, scores = _select_hyperparameters(train, val)
    model = _fit(train_val, best_params)
    raw = _predict(model, len(test), test_index)
    forecast = pd.Series(raw["yhat"].to_numpy(dtype=float), index=test_index, name="forecast")
    lower80 = pd.Series(raw["yhat_lower"].to_numpy(dtype=float), index=test_index, name="lower80")
    upper80 = pd.Series(raw["yhat_upper"].to_numpy(dtype=float), index=test_index, name="upper80")
    span80 = upper80 - lower80
    lower95 = (lower80 - span80 * 0.46875).rename("lower95")
    upper95 = (upper80 + span80 * 0.46875).rename("upper95")
    q10 = lower80.rename("q10")
    q50 = forecast.rename("q50")
    q90 = upper80.rename("q90")

    return ForecastResult(
        name="prophet",
        display_name=MODEL_LABELS["prophet"],
        forecast=forecast,
        runtime_seconds=time.perf_counter() - start,
        hyperparameters={**best_params, "country_holidays": "DE", "validation_scores": scores},
        q10=q10,
        q50=q50,
        q90=q90,
        lower80=lower80,
        upper80=upper80,
        lower95=lower95,
        upper95=upper95,
        fitted_model=model,
    )
