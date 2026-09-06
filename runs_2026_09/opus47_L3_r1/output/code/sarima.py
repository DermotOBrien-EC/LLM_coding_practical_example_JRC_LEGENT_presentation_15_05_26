"""SARIMA baseline via statsmodels SARIMAX.

pmdarima's auto_arima at s=24 on 40k hourly rows is impractical (hours per
candidate order); we run a small manual grid on the validation window and
refit the winner on train+val for the test forecast.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from common import Splits, mape

warnings.filterwarnings("ignore")

# Candidate seasonal orders, s=24 daily seasonality. Weekly structure is
# expected to come out of first differencing plus day-of-week regularity in
# the seasonal MA term. Keep the grid small to stay under a few minutes.
CANDIDATE_ORDERS: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = [
    ((1, 0, 1), (1, 1, 1, 24)),
    ((2, 0, 1), (1, 1, 1, 24)),
    ((1, 1, 1), (1, 1, 1, 24)),
]


def _fit(order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int], y: pd.Series):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=50, method="lbfgs")


def _pick_order(splits: Splits) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """Score each candidate on the validation window; pick lowest val MAPE."""
    # Fit on train, forecast the full val window in one shot, score.
    y_train = splits.train.astype(float)
    best = None
    best_mape = float("inf")
    for order, seasonal_order in CANDIDATE_ORDERS:
        try:
            res = _fit(order, seasonal_order, y_train)
            fc = res.forecast(steps=len(splits.val))
            fc.index = splits.val.index
            m = mape(splits.val.to_numpy(), fc.to_numpy())
        except Exception:
            m = float("inf")
        if m < best_mape:
            best_mape = m
            best = (order, seasonal_order)
    assert best is not None
    return best


def forecast(splits: Splits) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Fit SARIMA, return point forecast, interval frame, and hyperparameters.

    The interval frame has columns [lower_80, upper_80, lower_95, upper_95]
    indexed like splits.test.
    """
    order, seasonal_order = _pick_order(splits)
    # Final refit on train + val.
    y_fit = splits.train_val.astype(float)
    res = _fit(order, seasonal_order, y_fit)
    steps = len(splits.test)
    fc_res = res.get_forecast(steps=steps)
    mean = fc_res.predicted_mean
    mean.index = splits.test.index
    mean.name = "sarima"
    ci80 = fc_res.conf_int(alpha=0.20)
    ci95 = fc_res.conf_int(alpha=0.05)
    ci80.index = splits.test.index
    ci95.index = splits.test.index
    intervals = pd.DataFrame({
        "lower_80": ci80.iloc[:, 0].to_numpy(),
        "upper_80": ci80.iloc[:, 1].to_numpy(),
        "lower_95": ci95.iloc[:, 0].to_numpy(),
        "upper_95": ci95.iloc[:, 1].to_numpy(),
    }, index=splits.test.index)
    hp = {"order": list(order), "seasonal_order": list(seasonal_order)}
    return mean, intervals, hp
