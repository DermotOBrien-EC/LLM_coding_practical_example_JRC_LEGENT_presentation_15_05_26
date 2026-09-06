from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import (
    FINAL_TRAIN_END,
    ModelOutput,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    mape_pct,
)

CANDIDATES = [
    {"order": (1, 0, 1), "seasonal_order": (0, 0, 0, 24)},
    {"order": (1, 0, 0), "seasonal_order": (1, 0, 0, 24)},
    {"order": (2, 0, 1), "seasonal_order": (0, 0, 0, 24)},
]


def fit_sarimax(series: pd.Series, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]) -> object:
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=True,
    )
    return model.fit(disp=False, maxiter=30, method="lbfgs")


def select_params_with_grid(series: pd.Series) -> dict[str, object]:
    train = series.loc[TRAIN_START:TRAIN_END]
    val = series.loc[VAL_START:VAL_END]
    best = CANDIDATES[0]
    best_score = float("inf")
    for candidate in CANDIDATES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit_sarimax(train, candidate["order"], candidate["seasonal_order"])
            forecast = result.get_forecast(steps=len(val)).predicted_mean
        forecast.index = val.index
        score = mape_pct(val, forecast)
        if score < best_score:
            best_score = score
            best = candidate
    return {**best, "validation_mape_pct": best_score, "selection_method": "statsmodels_grid"}


def select_params(series: pd.Series) -> dict[str, object]:
    return select_params_with_grid(series)


def run(series: pd.Series) -> ModelOutput:
    selected = select_params(series)
    order = tuple(int(x) for x in selected["order"])
    seasonal_order = tuple(int(x) for x in selected["seasonal_order"])
    final_train = series.loc[TRAIN_START:FINAL_TRAIN_END]
    test_index = pd.date_range(TEST_START, TEST_END, freq="h")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_sarimax(final_train, order, seasonal_order)
        pred = result.get_forecast(steps=len(test_index))
    forecast = pred.predicted_mean
    forecast.index = test_index
    intervals = pred.conf_int(alpha=0.2)
    lower_80 = pd.Series(intervals.iloc[:, 0].to_numpy(dtype=float), index=test_index)
    upper_80 = pd.Series(intervals.iloc[:, 1].to_numpy(dtype=float), index=test_index)
    intervals_95 = pred.conf_int(alpha=0.05)
    lower_95 = pd.Series(intervals_95.iloc[:, 0].to_numpy(dtype=float), index=test_index)
    upper_95 = pd.Series(intervals_95.iloc[:, 1].to_numpy(dtype=float), index=test_index)
    std = (upper_95 - forecast) / 1.959963984540054
    q10 = forecast + (-1.2815515655446004) * std
    q50 = forecast.copy()
    q90 = forecast + 1.2815515655446004 * std
    return ModelOutput(
        name="sarima",
        forecast=forecast,
        runtime_seconds=0.0,
        hyperparameters=selected,
        lower_80=lower_80,
        upper_80=upper_80,
        lower_95=lower_95,
        upper_95=upper_95,
        q10=q10,
        q50=q50,
        q90=q90,
        fitted_model=result,
    )
