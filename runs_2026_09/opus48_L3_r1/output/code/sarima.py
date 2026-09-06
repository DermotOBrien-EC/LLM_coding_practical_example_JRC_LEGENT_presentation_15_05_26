"""Classical seasonal ARIMA via statsmodels SARIMAX.

SARIMA is the textbook statistical forecaster: it models the series as its own
past values (autoregression), past forecast errors (moving average), and a
repeating seasonal pattern. Hourly electricity load really has two seasons, a
daily one and a weekly one, but a weekly season at hourly resolution (period
168) makes the state-space model enormous and slow. Following the prompt we use
the daily season (period 24) explicitly and let the weekly and longer structure
show up through differencing and the model's own memory.

We try a few (p,d,q)(P,D,Q,24) orders, keep the one with the lowest validation
MAPE, refit it on train+validation, and forecast the test week. Prediction
intervals come from the model's own forecast standard errors under a normal
assumption.

Two practical notes. First, we do NOT use ``simple_differencing``: with it on,
statsmodels returns forecasts of the differenced series (values near zero)
rather than of the load itself, which silently breaks the level forecast.
Second, we score candidate orders on the first week (168 hours) of the
validation window rather than the whole three months, because a SARIMA forecast
run thousands of hours ahead just reverts to its average seasonal shape and
stops telling the orders apart; the 168-hour horizon is also the one the test
uses.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX

import common as c

warnings.filterwarnings("ignore")

# (order, seasonal_order) candidates. Kept small on purpose: each fit on ~42k
# hourly points is expensive, and the goal is a fair classical baseline, not a
# grand order search.
CANDIDATES: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = [
    ((2, 1, 2), (1, 0, 1, 24)),
    ((2, 1, 2), (1, 1, 1, 24)),
    ((1, 1, 2), (0, 1, 1, 24)),
]

QUANTILE_LEVELS: list[float] = [0.025, 0.1, 0.5, 0.9, 0.975]
_MAXITER: int = 30


def _fit(series: pd.Series, order, seasonal_order) -> "SARIMAX":
    model = SARIMAX(
        series.to_numpy(),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=_MAXITER, method="lbfgs")


def run(splits: c.Splits) -> c.ModelResult:
    start = time.time()

    # ---- order selection on the validation window ----------------------
    best = None
    best_val_mape = np.inf
    sweep: list[dict] = []
    val_week = splits.val.iloc[:168]  # first validation week; matches the test horizon
    for order, seasonal_order in CANDIDATES:
        fitted = _fit(splits.train, order, seasonal_order)
        fc = np.asarray(fitted.get_forecast(steps=168).predicted_mean, dtype=float)
        val_mape = c.mape(val_week.to_numpy(), fc)
        sweep.append({"order": order, "seasonal_order": seasonal_order, "val_mape": round(val_mape, 4)})
        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best = (order, seasonal_order)

    assert best is not None
    order, seasonal_order = best

    # ---- refit on Train+Val and forecast the 168 test hours ------------
    fitted = _fit(splits.trainval, order, seasonal_order)
    forecast = fitted.get_forecast(steps=len(splits.test))
    mean = np.asarray(forecast.predicted_mean, dtype=float)
    se = np.asarray(forecast.se_mean, dtype=float)

    point = pd.Series(mean, index=splits.test.index, name="sarima")
    quantiles: dict[float, pd.Series] = {}
    for q in QUANTILE_LEVELS:
        vals = mean + norm.ppf(q) * se
        quantiles[q] = pd.Series(vals, index=splits.test.index, name=f"sarima_q{q}")

    return c.ModelResult(
        name="sarima",
        point=point,
        runtime_s=time.time() - start,
        hyperparameters={
            "order": list(order),
            "seasonal_order": list(seasonal_order),
            "selected_by_val_mape": round(best_val_mape, 4),
            "maxiter": _MAXITER,
            "sweep": sweep,
        },
        quantiles=quantiles,
    )
