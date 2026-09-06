"""SARIMA(X) with a small validation-scored grid.

Weekly seasonality (m=168) makes state-space SARIMA impractical on ~50 k
hours. Standard practice for hourly-load SARIMA is a *daily* seasonal
period m=24 and let weekly structure show up through non-seasonal
differencing plus AR terms. This is what the spec recommends.

We validate a compact grid of (p, d, q)(P, D, Q, 24) candidates on the
Oct-Dec 2019 window (a 168-hour one-week rolling forecast starting on
each candidate's fit output) and pick the one with the lowest validation
MAPE. Then we refit on Train + Validation and forecast the test week.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import Splits, align, mape


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning)


CANDIDATE_ORDERS: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = [
    ((2, 0, 1), (1, 1, 1, 24)),
    ((2, 1, 1), (1, 1, 1, 24)),
    ((1, 1, 1), (1, 1, 1, 24)),
]


def _fit(series: pd.Series, order: tuple[int, int, int], seasonal: tuple[int, int, int, int]):
    """Fit a SARIMAX model on the given series."""
    model = SARIMAX(
        series.astype("float64"),
        order=order,
        seasonal_order=seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=50, method="lbfgs")


def _score_on_val(
    train: pd.Series, val: pd.Series, order, seasonal
) -> float:
    """Fit on train, forecast the first 168 h of val, return MAPE.

    Using the first week of validation keeps this cheap; the horizon
    matches the test-week horizon, so the score is representative.
    """
    fit = _fit(train, order, seasonal)
    horizon = min(168, len(val))
    fc = fit.forecast(steps=horizon)
    return mape(val.iloc[:horizon].values, np.asarray(fc.values))


def run(splits: Splits) -> dict[str, object]:
    """Pick an order by validation MAPE, refit on train+val, forecast test."""
    val_scores: dict[tuple, float] = {}
    for order, seasonal in CANDIDATE_ORDERS:
        try:
            val_scores[(order, seasonal)] = _score_on_val(
                splits.train, splits.val, order, seasonal
            )
        except Exception as exc:  # noqa: BLE001 - record and move on
            val_scores[(order, seasonal)] = float("inf")
            print(f"[sarima] {order}{seasonal} failed on val: {exc}")

    best_key = min(val_scores, key=val_scores.get)
    best_order, best_seasonal = best_key

    final = _fit(splits.trainval, best_order, best_seasonal)
    horizon = len(splits.test)
    forecast_res = final.get_forecast(steps=horizon)
    forecast = align(forecast_res.predicted_mean.values, splits.test.index)

    ci80 = forecast_res.conf_int(alpha=0.20)
    ci95 = forecast_res.conf_int(alpha=0.05)
    lower_80 = align(ci80.iloc[:, 0].values, splits.test.index)
    upper_80 = align(ci80.iloc[:, 1].values, splits.test.index)
    lower_95 = align(ci95.iloc[:, 0].values, splits.test.index)
    upper_95 = align(ci95.iloc[:, 1].values, splits.test.index)

    # Analytic quantile forecasts for pinball loss (Normal from PI).
    # SARIMAX PIs use z at half-alpha; recover sigma from the 95% band width.
    sigma = (upper_95.values - lower_95.values) / (2.0 * 1.959963984540054)
    from scipy.stats import norm
    q10 = align(forecast.values + norm.ppf(0.10) * sigma, splits.test.index)
    q50 = forecast
    q90 = align(forecast.values + norm.ppf(0.90) * sigma, splits.test.index)

    return {
        "name": "sarima",
        "forecast": forecast,
        "lower_80": lower_80,
        "upper_80": upper_80,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "quantiles": {"0.1": q10, "0.5": q50, "0.9": q90},
        "hyperparameters": {
            "order": list(best_order),
            "seasonal_order": list(best_seasonal),
            "validation_mape_pct": val_scores[best_key],
            "candidates_scored": {
                f"{k[0]}x{k[1]}": v for k, v in val_scores.items()
            },
        },
    }
