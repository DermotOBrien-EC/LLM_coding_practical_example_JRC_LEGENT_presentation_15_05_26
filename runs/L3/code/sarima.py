"""SARIMA on hourly load.

We use `statsmodels.SARIMAX` with explicit `(p,d,q)(P,D,Q,s=24)` orders. The
prompt allows either pmdarima.auto_arima or statsmodels SARIMAX; we picked
the latter because:

  * pmdarima.auto_arima with seasonal=True and m=24 over ~43,000 hourly
    rows runs for hours in practice — the stepwise search refits dozens
    of candidate SARIMAX models on the full series.
  * A small hand-curated grid evaluated on the validation window gives a
    transparent, reproducible choice in minutes, which is what this study
    actually needs.

Daily seasonality is captured directly through `s=24`. Weekly structure
emerges through the AR/MA terms — strict weekly seasonal m=168 is
intractable for SARIMAX on this volume, and the prompt explicitly accepts
this trade-off.

For runtime sanity we train each candidate on the last `MAX_TRAIN_HOURS`
of available history (one year). That is enough to capture daily and
weekly structure while keeping each fit under a few minutes.
"""

from __future__ import annotations

import time
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from code.common import (
    N_TEST,
    TEST_END,
    TEST_START,
    ModelForecast,
    mape,
)

MAX_TRAIN_HOURS: int = 8760  # one full year of context per fit
CANDIDATE_ORDERS: Sequence[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = (
    ((1, 0, 1), (1, 1, 1, 24)),
    ((2, 0, 1), (1, 1, 1, 24)),
    ((2, 1, 2), (0, 1, 1, 24)),
)


def _fit_sarimax(endog: pd.Series, order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]):
    """Fit one SARIMAX. Warnings about convergence are noisy but expected."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model = SARIMAX(
            endog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=50)


def run_sarima(train: pd.Series, val: pd.Series, trainval: pd.Series) -> ModelForecast:
    """Pick an order on validation, refit on train+val, forecast the test week."""
    started = time.perf_counter()

    train_tail = train.iloc[-MAX_TRAIN_HOURS:]

    val_mapes: dict[str, float] = {}
    best_order = None
    best_seasonal = None
    best_mape = float("inf")
    for order, seasonal in CANDIDATE_ORDERS:
        fit_res = _fit_sarimax(train_tail, order, seasonal)
        f = fit_res.get_forecast(steps=len(val)).predicted_mean
        f.index = val.index
        m = mape(val.to_numpy(), f.to_numpy())
        key = f"{order}x{seasonal}"
        val_mapes[key] = m
        if m < best_mape:
            best_mape = m
            best_order = order
            best_seasonal = seasonal

    # Refit on last MAX_TRAIN_HOURS of train+val combined.
    refit_tail = trainval.iloc[-MAX_TRAIN_HOURS:]
    final_fit = _fit_sarimax(refit_tail, best_order, best_seasonal)
    fcast = final_fit.get_forecast(steps=N_TEST)
    point = fcast.predicted_mean.to_numpy()
    se = fcast.var_pred_mean.to_numpy() ** 0.5

    # Analytic normal intervals on the level of the load itself.
    z80 = 1.2815515655446004
    z95 = 1.959963984540054
    z10 = 1.2815515655446004  # 10th percentile is mean - z10*se
    z90 = 1.2815515655446004

    return ModelForecast(
        name="sarima",
        point=point,
        q10=point - z10 * se,
        q50=point,
        q90=point + z90 * se,
        lower80=point - z80 * se,
        upper80=point + z80 * se,
        lower95=point - z95 * se,
        upper95=point + z95 * se,
        runtime_seconds=time.perf_counter() - started,
        hyperparameters={
            "order": list(best_order),
            "seasonal_order": list(best_seasonal),
            "train_hours_used": MAX_TRAIN_HOURS,
            "validation_mape_by_order": val_mapes,
            "selected_validation_mape": best_mape,
        },
    )
