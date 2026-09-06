from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults

from common import ForecastResult, QUANTILES, mape_pct

Candidate = tuple[tuple[int, int, int], tuple[int, int, int, int]]
CANDIDATES: tuple[Candidate, ...] = (
    ((1, 0, 1), (1, 0, 0, 24)),
    ((2, 0, 1), (0, 0, 1, 24)),
    ((1, 0, 2), (1, 0, 0, 24)),
)


def weekly_difference(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    return np.asarray(values[168:] - values[:-168], dtype=float)


def _fit_sarima(values: np.ndarray, candidate: Candidate) -> SARIMAXResults:
    order, seasonal_order = candidate
    model = SARIMAX(
        values,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
        concentrate_scale=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return model.fit(method="lbfgs", maxiter=35, disp=False)


def _restore_levels(difference_forecast: np.ndarray, observed_history: pd.Series) -> np.ndarray:
    history = observed_history.to_numpy(dtype=float).tolist()
    restored = np.empty(len(difference_forecast), dtype=float)
    for index, difference in enumerate(np.asarray(difference_forecast, dtype=float)):
        value = history[-168] + difference
        history.append(float(value))
        restored[index] = value
    return restored


def _candidate_record(
    candidate: Candidate, validation_mape: float, fitted: SARIMAXResults
) -> dict[str, Any]:
    order, seasonal_order = candidate
    return {
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "validation_mape_pct": validation_mape,
        "converged": bool(fitted.mle_retvals.get("converged", False)),
        "aic": float(fitted.aic),
    }


def run_sarima(
    train: pd.Series,
    validation: pd.Series,
    train_validation: pd.Series,
    test_index: pd.DatetimeIndex,
) -> ForecastResult:
    started = time.perf_counter()
    train_differenced = weekly_difference(train)
    validation_records: list[dict[str, Any]] = []
    best_candidate: Candidate | None = None
    best_mape = float("inf")

    for candidate in CANDIDATES:
        fitted = _fit_sarima(train_differenced, candidate)
        difference_forecast = np.asarray(
            fitted.get_forecast(steps=len(validation)).predicted_mean, dtype=float
        )
        forecast = _restore_levels(difference_forecast, train)
        candidate_mape = mape_pct(validation.to_numpy(), forecast)
        candidate_record = _candidate_record(candidate, candidate_mape, fitted)
        validation_records.append(candidate_record)
        if not candidate_record["converged"]:
            continue
        if candidate_mape < best_mape:
            best_mape = candidate_mape
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("SARIMA validation produced no candidate")

    final_fitted = _fit_sarima(weekly_difference(train_validation), best_candidate)
    if not bool(final_fitted.mle_retvals.get("converged", False)):
        raise RuntimeError("Selected SARIMA configuration did not converge on the pre-test refit")
    forecast_distribution = final_fitted.get_forecast(steps=len(test_index))
    difference_mean = np.asarray(forecast_distribution.predicted_mean, dtype=float)
    difference_se = np.asarray(forecast_distribution.se_mean, dtype=float)
    point = _restore_levels(difference_mean, train_validation)
    quantiles = {
        probability: _restore_levels(
            difference_mean + norm.ppf(probability) * difference_se,
            train_validation,
        )
        for probability in QUANTILES
    }
    order, seasonal_order = best_candidate
    return ForecastResult(
        name="sarima",
        point=point,
        quantiles=quantiles,
        runtime_seconds=time.perf_counter() - started,
        hyperparameters={
            "weekly_difference_hours": 168,
            "order": list(order),
            "seasonal_order": list(seasonal_order),
            "trend": "constant",
            "maxiter": 35,
            "selection_metric": "validation_mape_pct",
            "validation_mape_pct": best_mape,
            "candidates": validation_records,
            "final_converged": bool(final_fitted.mle_retvals.get("converged", False)),
        },
    )
