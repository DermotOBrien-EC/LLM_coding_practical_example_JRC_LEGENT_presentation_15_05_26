from __future__ import annotations

import warnings
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .common import QUANTILES, TEST_INDEX, TRAIN_END, ForecastResult, mape, validation_blocks

CANDIDATES = [(1, 0, 0), (2, 0, 0)]
SEASONAL_ORDER = (1, 1, 0, 24)


def weekly_fourier(index: pd.DatetimeIndex) -> np.ndarray:
    hours = (index - pd.Timestamp("2015-01-01", tz="UTC")).total_seconds().to_numpy() / 3600.0
    return np.column_stack(
        [
            function(2.0 * np.pi * harmonic * hours / 168.0)
            for harmonic in [1, 2, 3]
            for function in [np.sin, np.cos]
        ]
    )


def fit(series: pd.Series, order: tuple[int, int, int], scale: float) -> tuple[Any, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = SARIMAX(
            series.to_numpy() / scale,
            exog=weekly_fourier(series.index),
            order=order,
            seasonal_order=SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
            concentrate_scale=False,
        )
        result = model.fit(disp=False, maxiter=60, low_memory=False)
    messages = sorted({str(warning.message) for warning in caught})
    return result, messages


def run(pretest: pd.Series) -> ForecastResult:
    start = perf_counter()
    train = pretest.loc[:TRAIN_END]
    scale = float(train.mean())
    records = []
    for order in CANDIDATES:
        fitted, messages = fit(train, order, scale)
        state = fitted
        predictions, actuals = [], []
        for _, target in validation_blocks(pretest):
            predicted = state.get_forecast(len(target), exog=weekly_fourier(target.index))
            predictions.extend(np.asarray(predicted.predicted_mean) * scale)
            actuals.extend(target.values)
            # Filtering advances the origin with known history; coefficients stay frozen.
            if target.index[-1] < pretest.index[-1]:
                state = state.extend(target.to_numpy() / scale, exog=weekly_fourier(target.index))
                np.testing.assert_array_equal(state.params, fitted.params)
        value = mape(np.array(actuals), np.array(predictions))
        records.append(
            {
                "order": list(order),
                "mape_validation_pct": value,
                "converged": bool(fitted.mle_retvals.get("converged", False)),
                "warnings": messages,
                "iterations": int(fitted.mle_retvals.get("iterations", 0)),
            }
        )
        print(
            f"sarima validation {order}: {value:.4f}% converged={records[-1]['converged']}",
            flush=True,
        )
    eligible = [record for record in records if record["converged"]]
    if not eligible:
        raise RuntimeError("No SARIMA candidate converged; do not report an unverified fit")
    selected = tuple(min(eligible, key=lambda record: record["mape_validation_pct"])["order"])
    final_scale = float(pretest.mean())
    final, messages = fit(pretest, selected, final_scale)
    if not final.mle_retvals.get("converged", False):
        raise RuntimeError("Final SARIMA refit failed to converge")
    forecast = final.get_forecast(168, exog=weekly_fourier(TEST_INDEX))
    point = np.asarray(forecast.predicted_mean) * final_scale
    sigma = np.asarray(forecast.se_mean) * final_scale
    quantiles = point[:, None] + sigma[:, None] * norm.ppf(QUANTILES)[None, :]
    return ForecastResult(
        name="sarima",
        point=point,
        quantiles=quantiles,
        runtime_seconds=perf_counter() - start,
        hyperparameters={
            "order": list(selected),
            "seasonal_order": list(SEASONAL_ORDER),
            "weekly_fourier_harmonics": 3,
            "maxiter": 60,
            "enforce_stationarity": False,
            "enforce_invertibility": False,
            "concentrate_scale": False,
            "intervals": "Gaussian state-space forecast errors, fixed estimated parameters",
        },
        validation=records,
        diagnostics={
            "final_converged": True,
            "final_warnings": messages,
            "final_iterations": int(final.mle_retvals.get("iterations", 0)),
        },
    )
