from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
    series_from_frame,
)

CANDIDATES: list[dict[str, Any]] = [
    {"order": (1, 0, 1), "seasonal_order": (1, 1, 0, 24)},
    {"order": (2, 0, 1), "seasonal_order": (1, 1, 0, 24)},
    {"order": (1, 0, 1), "seasonal_order": (0, 1, 1, 24)},
]


def _fit(series: pd.Series, params: dict[str, Any]) -> Any:
    model = SARIMAX(
        series,
        order=params["order"],
        seasonal_order=params["seasonal_order"],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=35)


def _forecast_with_intervals(fitted: Any, n: int, index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    pred = fitted.get_forecast(steps=n)
    mean = pd.Series(np.asarray(pred.predicted_mean, dtype=float), index=index, name="forecast")
    se = np.sqrt(np.asarray(pred.var_pred_mean, dtype=float))
    q10 = pd.Series(mean.to_numpy() - 1.2815515655446004 * se, index=index, name="q10")
    q50 = mean.rename("q50")
    q90 = pd.Series(mean.to_numpy() + 1.2815515655446004 * se, index=index, name="q90")
    lower80 = q10.rename("lower80")
    upper80 = q90.rename("upper80")
    lower95 = pd.Series(mean.to_numpy() - 1.959963984540054 * se, index=index, name="lower95")
    upper95 = pd.Series(mean.to_numpy() + 1.959963984540054 * se, index=index, name="upper95")
    return mean, q10, q50, q90, lower80, upper80, lower95, upper95


def _select_hyperparameters(train: pd.DataFrame, val: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, float | str]]]:
    train_series = series_from_frame(train).asfreq("h")
    val_series = series_from_frame(val).asfreq("h")
    scores: list[dict[str, float | str]] = []
    best_params = CANDIDATES[0]
    best_score = np.inf
    for i, params in enumerate(CANDIDATES):
        fitted = _fit(train_series, params)
        forecast, *_ = _forecast_with_intervals(fitted, len(val_series), pd.DatetimeIndex(val_series.index))
        score = mape_pct(val_series, forecast)
        scores.append({"candidate": str(i), "validation_mape_pct": float(score), "aic": float(fitted.aic)})
        if score < best_score:
            best_score = score
            best_params = params
    return best_params, scores


def run(df: pd.DataFrame) -> ForecastResult:
    start = time.perf_counter()
    train = df[df[TIME_COL] <= TRAIN_END].copy()
    val = df[(df[TIME_COL] >= VAL_START) & (df[TIME_COL] <= VAL_END)].copy()
    train_val = df[df[TIME_COL] <= VAL_END].copy()
    test_index = pd.DatetimeIndex(df.loc[(df[TIME_COL] >= TEST_START) & (df[TIME_COL] <= TEST_END), TIME_COL])

    best_params, scores = _select_hyperparameters(train, val)
    fitted = _fit(series_from_frame(train_val).asfreq("h"), best_params)
    forecast, q10, q50, q90, lower80, upper80, lower95, upper95 = _forecast_with_intervals(fitted, len(test_index), test_index)
    serialisable_params = {
        "order": list(best_params["order"]),
        "seasonal_order": list(best_params["seasonal_order"]),
        "validation_scores": scores,
        "maxiter": 35,
    }
    return ForecastResult(
        name="sarima",
        display_name=MODEL_LABELS["sarima"],
        forecast=forecast,
        runtime_seconds=time.perf_counter() - start,
        hyperparameters=serialisable_params,
        q10=q10,
        q50=q50,
        q90=q90,
        lower80=lower80,
        upper80=upper80,
        lower95=lower95,
        upper95=upper95,
        fitted_model=fitted,
    )
