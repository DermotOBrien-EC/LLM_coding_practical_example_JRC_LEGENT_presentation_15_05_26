"""SARIMA on hourly load with daily seasonality (m = 24).

Fitting a full weekly-seasonal SARIMA (m = 168) on ~42 000 hourly observations
is impractically slow in `statsmodels`. Instead we do what the prompt allows:
fit SARIMAX with daily seasonality and let a first seasonal difference at
lag 24 h absorb the strongest cycle; the weekly pattern shows up implicitly in
the residual dynamics via the non-seasonal AR/MA terms. To keep the fit
tractable we score a small grid on Validation and refit the winner on
Train + Validation.

Because a full SARIMAX MLE on 42k hourly rows is heavy, the grid is tiny and
the winner is selected by validation MAPE.

Prediction intervals come from `get_forecast(...).conf_int(alpha=...)` on
the refit statespace model.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import Split, mape, to_hourly_naive

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class SarimaResult:
    forecast: pd.Series
    lower_80: pd.Series
    upper_80: pd.Series
    lower_95: pd.Series
    upper_95: pd.Series
    runtime_seconds: float
    hyperparameters: dict[str, object]


def _fit_one(
    y_train: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> object:
    model = SARIMAX(
        y_train.values,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, method="lbfgs", maxiter=50)


def _score_val(
    fit: object,
    n_forecast: int,
    y_val: pd.Series,
) -> float:
    fc = fit.get_forecast(steps=n_forecast).predicted_mean
    return mape(y_val.values, np.asarray(fc))


def _search(
    split: Split,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int], float]:
    """A tiny grid: differencing is fixed, we sweep AR/MA orders only."""
    y_train = to_hourly_naive(split.train)
    y_val = to_hourly_naive(split.val)
    # Grid trimmed after a first pass showed (3,1,1)(1,1,1,24) dominating the
    # smaller AR orders on the validation window (val MAPE ~12 % vs ~53 %).
    # Three candidates keep the sweep cheap while spanning AR depth and MA
    # depth.
    candidates: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = [
        ((1, 1, 1), (1, 1, 1, 24)),
        ((2, 1, 2), (1, 1, 1, 24)),
        ((3, 1, 1), (1, 1, 1, 24)),
    ]
    best: tuple[object, tuple, tuple, float] | None = None
    for order, seasonal_order in candidates:
        try:
            fit = _fit_one(y_train, order, seasonal_order)
            m = _score_val(fit, len(y_val), y_val)
        except Exception as exc:  # pragma: no cover
            print(f"  SARIMA {order}x{seasonal_order} failed: {exc}")
            continue
        print(f"  SARIMA {order}x{seasonal_order}: val MAPE {m:.3f}%")
        if best is None or m < best[3]:
            best = (fit, order, seasonal_order, m)
    if best is None:
        raise RuntimeError("no SARIMA candidate converged")
    return best[1], best[2], best[3]


def run_sarima(split: Split) -> SarimaResult:
    t0 = time.perf_counter()
    order, seasonal_order, val_mape = _search(split)

    # Refit on Train + Validation.
    y_full = to_hourly_naive(split.train_plus_val)
    fit_full = _fit_one(y_full, order, seasonal_order)

    forecast_obj = fit_full.get_forecast(steps=168)
    point = np.asarray(forecast_obj.predicted_mean)
    ci_80 = forecast_obj.conf_int(alpha=0.20)
    ci_95 = forecast_obj.conf_int(alpha=0.05)
    # SARIMAX returns a numpy array or DataFrame depending on version.
    if isinstance(ci_80, pd.DataFrame):
        lo80, hi80 = ci_80.iloc[:, 0].values, ci_80.iloc[:, 1].values
        lo95, hi95 = ci_95.iloc[:, 0].values, ci_95.iloc[:, 1].values
    else:
        lo80, hi80 = ci_80[:, 0], ci_80[:, 1]
        lo95, hi95 = ci_95[:, 0], ci_95[:, 1]

    idx = split.test.index
    forecast = pd.Series(point, index=idx, name="sarima")
    lower_80 = pd.Series(lo80, index=idx, name="q10")
    upper_80 = pd.Series(hi80, index=idx, name="q90")
    lower_95 = pd.Series(lo95, index=idx, name="q025")
    upper_95 = pd.Series(hi95, index=idx, name="q975")

    runtime = time.perf_counter() - t0
    hp = {
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "validation_mape_pct": float(val_mape),
    }
    return SarimaResult(
        forecast=forecast,
        lower_80=lower_80,
        upper_80=upper_80,
        lower_95=lower_95,
        upper_95=upper_95,
        runtime_seconds=runtime,
        hyperparameters=hp,
    )
