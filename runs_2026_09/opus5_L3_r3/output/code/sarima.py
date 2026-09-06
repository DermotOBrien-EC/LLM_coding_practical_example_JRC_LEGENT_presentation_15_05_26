"""SARIMA: the classical statistical entry in the bake-off.

A seasonal ARIMA describes the series purely in terms of its own recent
history: how today's load relates to the last few hours, and how it relates
to the same hour yesterday. Orders are written (p,d,q)(P,D,Q,s) where the
first triple is the short-memory part and the second is the seasonal part
repeating every s hours.

Two practical decisions are worth stating up front.

1. The seasonal period is s = 24, not 168. A weekly state-space seasonal
   term needs a hidden state of roughly 168 dimensions, which turns each
   likelihood evaluation into minutes of linear algebra. The study brief
   therefore asks for daily seasonality and lets the weekly rhythm show up,
   if it can, through differencing. It largely cannot, and that is one of
   the findings rather than a bug.

2. The model is fitted on a trailing window of the training data rather than
   all 41,616 hours. A Kalman filter pass costs time proportional to the
   sample, and an ARIMA has no long memory anyway: hours from 2015 carry
   almost no information about January 2020. The length of that window is
   itself treated as a hyperparameter and chosen on the validation weeks.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import (
    HORIZON,
    QUANTILE_LEVELS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    VAL_END,
    ForecastResult,
    mape,
    validation_origins,
)

# Candidate orders. Kept small and deliberately conventional: a low-order
# short-memory part, one seasonal difference at lag 24 to remove the daily
# shape, and at most one seasonal AR / MA term.
CANDIDATE_ORDERS: tuple[tuple[tuple[int, int, int], tuple[int, int, int, int]], ...] = (
    ((1, 1, 1), (0, 1, 1, 24)),
    ((2, 1, 1), (0, 1, 1, 24)),
    ((2, 1, 2), (0, 1, 1, 24)),
    ((1, 1, 1), (1, 1, 1, 24)),
    ((2, 1, 2), (1, 1, 1, 24)),
    ((3, 1, 1), (1, 1, 0, 24)),
)

# Trailing-window lengths to try, in hours: thirteen weeks and twenty-six weeks.
CANDIDATE_WINDOWS: tuple[int, ...] = (2184, 4368)

# Number of validation weeks used for scoring. Four evenly spaced weeks keep
# the sweep (6 orders x 2 windows x 4 weeks) inside a few minutes.
N_VALIDATION_BLOCKS: int = 4


def _fit(endog: pd.Series, order: Any, seasonal_order: Any) -> Any:
    """Fit one SARIMAX, silencing the optimiser's routine complaints."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model = SARIMAX(
            endog,
            order=order,
            seasonal_order=seasonal_order,
            trend="n",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=80)


def _strip_freq(y: pd.Series) -> pd.Series:
    """statsmodels wants a plain index; the tz-aware one confuses its warnings."""
    out = y.copy()
    out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out.asfreq("h")


def _select(series: pd.Series) -> tuple[Any, Any, int, list[dict[str, Any]]]:
    """Choose order and window length by mean MAPE over validation weeks.

    Parameters are estimated once per candidate on data ending 2019-09-30.
    For each validation week the *same* parameters are then re-run over the
    history up to that week's start (statsmodels calls this `apply`), which
    updates the filter's state without re-estimating anything. That keeps the
    validation weeks strictly out of the fitting step while still giving each
    forecast an honest starting point.
    """
    origins = validation_origins(N_VALIDATION_BLOCKS)
    log: list[dict[str, Any]] = []

    for window in CANDIDATE_WINDOWS:
        train_tail = _strip_freq(series.loc[:TRAIN_END].iloc[-window:])
        for order, seasonal_order in CANDIDATE_ORDERS:
            t0 = time.perf_counter()
            try:
                fitted = _fit(train_tail, order, seasonal_order)
            except Exception as exc:  # a divergent optimiser is a rejected candidate
                log.append(
                    {
                        "order": list(order),
                        "seasonal_order": list(seasonal_order),
                        "window_hours": window,
                        "val_mape_pct": float("nan"),
                        "error": str(exc)[:120],
                    }
                )
                continue

            scores: list[float] = []
            for origin in origins:
                history = _strip_freq(series.loc[: origin - pd.Timedelta(hours=1)].iloc[-window:])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    applied = fitted.apply(history)
                    pred = applied.get_forecast(HORIZON).predicted_mean
                actual = series.loc[origin : origin + pd.Timedelta(hours=HORIZON - 1)]
                scores.append(mape(actual.to_numpy(), np.asarray(pred)))

            log.append(
                {
                    "order": list(order),
                    "seasonal_order": list(seasonal_order),
                    "window_hours": window,
                    "val_mape_pct": float(np.mean(scores)),
                    "fit_seconds": round(time.perf_counter() - t0, 1),
                }
            )

    usable = [row for row in log if np.isfinite(row.get("val_mape_pct", np.nan))]
    if not usable:
        raise RuntimeError("no SARIMA candidate converged")
    best = min(usable, key=lambda row: row["val_mape_pct"])
    return tuple(best["order"]), tuple(best["seasonal_order"]), int(best["window_hours"]), log


def run(series: pd.Series) -> ForecastResult:
    """Select orders on validation, refit on train+validation, forecast the test week."""
    start = time.perf_counter()

    order, seasonal_order, window, log = _select(series)

    # Refit on the trailing window of train + validation, i.e. everything
    # available up to 2019-12-31 23:00.
    final_tail = _strip_freq(series.loc[:VAL_END].iloc[-window:])
    fitted = _fit(final_tail, order, seasonal_order)

    test_index = series.loc[TEST_START:TEST_END].index
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = fitted.get_forecast(HORIZON)
    point = pd.Series(np.asarray(forecast.predicted_mean), index=test_index)

    # Analytic Gaussian intervals: the state-space model gives a forecast
    # standard error at each horizon, which converts straight to quantiles.
    mean = np.asarray(forecast.predicted_mean, dtype="float64")
    stderr = np.asarray(forecast.se_mean, dtype="float64")
    from scipy.stats import norm

    quantiles = {
        float(level): pd.Series(mean + norm.ppf(level) * stderr, index=test_index)
        for level in QUANTILE_LEVELS
    }

    return ForecastResult(
        name="sarima",
        point=point,
        quantiles=quantiles,
        hyperparameters={
            "order": list(order),
            "seasonal_order": list(seasonal_order),
            "training_window_hours": window,
            "selection_metric": "mean MAPE over 4 held-out validation weeks",
            "interval_method": "analytic Gaussian from state-space forecast standard errors",
        },
        runtime_seconds=time.perf_counter() - start,
        selection_log=log,
    )
