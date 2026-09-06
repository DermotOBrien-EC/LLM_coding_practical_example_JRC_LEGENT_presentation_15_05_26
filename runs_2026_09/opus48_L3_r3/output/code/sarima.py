"""SARIMA: a classical statistical time-series model.

SARIMA describes a series by how each value relates to recent past values
(the AR part), to recent forecast errors (the MA part), and to the same
structure one season ago (the seasonal part). Here the natural season is one
day (24 hours). The weekly pattern is left to emerge through differencing and
the daily terms rather than being modelled directly, because a seasonal period
of 168 is computationally impractical for this class of model.

Two practical choices worth stating plainly:

1. We fit on the most recent 90 days of the allowed data rather than all 4.75
   years. Fitting a seasonal ARIMA by maximum likelihood on ~42,000 hourly
   points is very slow, and load structure is dominated by recent seasonality,
   so a recent window is the standard, tractable choice. It still never touches
   the test window.
2. We choose the model order by which candidate gives the lowest error on the
   validation week, exactly as the study asks.

Prediction intervals come from the model's own forecast distribution, treated
as Gaussian (mean plus standard error), which is what SARIMAX provides.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX

import common as c

# How many recent days to fit on. 90 days = 2160 hourly points.
FIT_DAYS: int = 90
FIT_HOURS: int = FIT_DAYS * 24

# A short list of candidate (p,d,q)(P,D,Q,s) orders with daily season s=24.
# Kept small on purpose: each is a full maximum-likelihood fit.
CANDIDATE_ORDERS: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = [
    ((2, 0, 1), (1, 1, 1, 24)),
    ((2, 1, 2), (1, 1, 1, 24)),
    ((3, 0, 1), (2, 1, 0, 24)),
]


def _fit(endog: pd.Series, order, seasonal_order) -> "SARIMAXResults":
    """Fit one SARIMAX model, quietly, with a capped iteration budget."""
    model = SARIMAX(
        endog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False, maxiter=50, method="lbfgs")


def _recent(series: pd.Series, end: pd.Timestamp) -> pd.Series:
    """The last FIT_HOURS observations ending at (and including) `end`."""
    window = series.loc[:end].iloc[-FIT_HOURS:]
    return window.asfreq("h")  # label the hourly frequency so statsmodels is quiet


def run(series: pd.Series) -> c.ModelResult:
    """Select an order on validation, refit on train+val, forecast the test week."""
    start = time.perf_counter()

    # --- Step 1: choose the order by validation-week error. -----------------
    # Fit on the 90 days ending at the train boundary, forecast the first 168
    # hours of the validation window, and score against the validation actuals.
    fit_for_selection = _recent(series, c.TRAIN_END)
    val_target = c.val_series(series).iloc[: c.HORIZON]

    scored: list[tuple[float, tuple, tuple]] = []
    for order, seasonal_order in CANDIDATE_ORDERS:
        try:
            res = _fit(fit_for_selection, order, seasonal_order)
            fc = res.get_forecast(steps=c.HORIZON).predicted_mean.to_numpy()
            score = c.mape(val_target.to_numpy(), fc)
        except Exception:
            score = float("inf")
        scored.append((score, order, seasonal_order))

    scored.sort(key=lambda row: row[0])
    best_score, best_order, best_seasonal = scored[0]
    val_mapes = {f"{o}{s}": round(sc, 3) for sc, o, s in scored}

    # --- Step 2: refit the chosen order on the 90 days ending at val end. ----
    fit_final = _recent(series, c.VAL_END)
    final_res = _fit(fit_final, best_order, best_seasonal)

    # --- Step 3: forecast the 168 test hours with a Gaussian interval. -------
    forecast = final_res.get_forecast(steps=c.HORIZON)
    mean = forecast.predicted_mean.to_numpy(dtype=float)
    se = forecast.se_mean.to_numpy(dtype=float)

    quantiles = {q: mean + norm.ppf(q) * se for q in c.WINNER_QUANTILES}
    quantiles[0.5] = mean  # the mean is the centre of a Gaussian forecast

    runtime = time.perf_counter() - start
    return c.ModelResult(
        name="sarima",
        point=mean,
        runtime_seconds=runtime,
        hyperparameters={
            "order": list(best_order),
            "seasonal_order": list(best_seasonal),
            "fit_window_days": FIT_DAYS,
            "selection_metric": "validation_mape_pct",
            "validation_mape_by_order": val_mapes,
        },
        quantiles=quantiles,
    )
