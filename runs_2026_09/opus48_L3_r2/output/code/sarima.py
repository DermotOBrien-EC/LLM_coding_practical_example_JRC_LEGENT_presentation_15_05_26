"""Classical seasonal ARIMA (SARIMA) via statsmodels SARIMAX.

SARIMA is the textbook statistical forecaster. It models the series as its own
recent values (autoregression), recent shocks (moving average), and a repeating
seasonal shape, after differencing away trends and seasonal levels.

Two practical choices, both documented in transcript.md:

1. Seasonal period. Hourly load has a strong daily cycle (period 24) and a
   weekly cycle (period 168). A seasonal period of 168 is not practical to fit,
   so we use m = 24 (daily) as the prompt suggests and let the weekly shape
   come through differencing and the AR/MA terms.

2. Fitting window. Seasonal maximum-likelihood fitting over all 41,000+ hours
   is very slow and, for a one-week-ahead forecast, unnecessary: SARIMA leans
   on recent structure. We fit on a trailing window (the most recent ~90 days)
   ending at the fitting cutoff. This keeps each fit to seconds while still
   seeing several months of daily and weekly rhythm.

Prediction intervals come from the model's own Gaussian forecast distribution
(mean plus standard error), which is the natural, analytic way SARIMA quantifies
uncertainty.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX

import common as C

# (p, d, q)(P, D, Q, s) candidates. s = 24 (daily). We pick between them by
# validation MAPE, exactly as the prompt asks.
CANDIDATE_ORDERS: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = [
    ((2, 1, 2), (1, 1, 1, 24)),
    ((1, 1, 1), (1, 1, 1, 24)),
    ((2, 1, 1), (0, 1, 1, 24)),
]

TRAIN_WINDOW_HOURS = 24 * 90  # ~3 months of recent history


def _fit(endog: np.ndarray, order, seasonal_order) -> "SARIMAX":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            endog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False, maxiter=50)


def _forecast_quantiles(fitted, horizon: int) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Point forecast (mean) plus Gaussian quantiles from the forecast s.e."""
    fc = fitted.get_forecast(steps=horizon)
    mean = np.asarray(fc.predicted_mean, dtype=float)
    se = np.asarray(fc.se_mean, dtype=float)
    quantiles = {q: mean + norm.ppf(q) * se for q in C.QUANTILE_LEVELS}
    return mean, quantiles


def run(bundle: C.DataBundle) -> C.ForecastResult:
    start = time.time()

    # --- Select the order on the first validation week -----------------------
    train_hist = C.slice_inclusive(bundle.full, C.TRAIN_START, C.TRAIN_END)
    train_win = train_hist.to_numpy()[-TRAIN_WINDOW_HOURS:]
    val_actual = bundle.val_select.to_numpy()

    best_order, best_val = None, np.inf
    for order, seasonal_order in CANDIDATE_ORDERS:
        try:
            fitted = _fit(train_win, order, seasonal_order)
            mean, _ = _forecast_quantiles(fitted, C.HORIZON)
            score = C.mape(val_actual, mean)
        except Exception:
            score = np.inf
        if score < best_val:
            best_val, best_order = score, (order, seasonal_order)

    if best_order is None:  # every candidate blew up; fall back to a safe one
        best_order = CANDIDATE_ORDERS[1]

    order, seasonal_order = best_order

    # --- Refit on Train+Validation (trailing window) and forecast the test ---
    full_hist = C.slice_inclusive(bundle.full, C.TRAIN_START, C.VAL_END)
    full_win = full_hist.to_numpy()[-TRAIN_WINDOW_HOURS:]
    fitted = _fit(full_win, order, seasonal_order)
    point, quantiles = _forecast_quantiles(fitted, C.HORIZON)

    runtime = time.time() - start
    return C.ForecastResult(
        name="sarima",
        point=point,
        runtime_seconds=runtime,
        hyperparameters={
            "order": list(order),
            "seasonal_order": list(seasonal_order),
            "train_window_hours": TRAIN_WINDOW_HOURS,
        },
        val_mape=float(best_val),
        quantiles=quantiles,
    )


if __name__ == "__main__":
    b = C.build_bundle()
    res = run(b)
    print("sarima order:", res.hyperparameters, "val MAPE:", res.val_mape)
    print("sarima test MAPE:", C.mape(b.test.to_numpy(), res.point))
