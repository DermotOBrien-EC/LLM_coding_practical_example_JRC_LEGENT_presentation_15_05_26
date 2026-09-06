from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import (
    DataSplits,
    ForecastResult,
    mean_absolute_percentage_error,
    normal_quantile_series,
)

CANDIDATES: list[dict[str, Any]] = [
    {"order": (1, 0, 1), "seasonal_order": (1, 1, 1, 24)},
    {"order": (2, 0, 1), "seasonal_order": (1, 1, 1, 24)},
    {"order": (1, 1, 1), "seasonal_order": (1, 0, 1, 24)},
]
MAX_ITERATIONS = 25


def _fit(
    series: pd.Series,
    configuration: dict[str, Any],
    low_memory: bool = True,
) -> Any:
    model = SARIMAX(
        series.to_numpy(dtype=float),
        order=configuration["order"],
        seasonal_order=configuration["seasonal_order"],
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
        concentrate_scale=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(
            method="lbfgs",
            maxiter=MAX_ITERATIONS,
            disp=False,
            low_memory=low_memory,
        )


def _point_forecast(fitted: Any, index: pd.DatetimeIndex) -> pd.Series:
    values = fitted.get_forecast(steps=len(index)).predicted_mean
    return pd.Series(np.asarray(values, dtype=float), index=index, name="forecast")


def run(splits: DataSplits) -> ForecastResult:
    started = time.perf_counter()
    candidate_records: list[dict[str, Any]] = []
    fitted_candidates: list[Any | None] = []

    for configuration in CANDIDATES:
        candidate_started = time.perf_counter()
        try:
            fitted = _fit(splits.train, configuration)
            forecast = _point_forecast(fitted, splits.validation.index)
            score = mean_absolute_percentage_error(splits.validation, forecast)
            converged = bool(fitted.mle_retvals.get("converged", False))
            record = {
                **configuration,
                "validation_mape_pct": score,
                "converged": converged,
                "fit_runtime_seconds": time.perf_counter() - candidate_started,
            }
        except (ValueError, np.linalg.LinAlgError) as exc:
            fitted = None
            record = {
                **configuration,
                "validation_mape_pct": None,
                "converged": False,
                "fit_runtime_seconds": time.perf_counter() - candidate_started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        fitted_candidates.append(fitted)
        candidate_records.append(record)

    valid_positions = [
        position
        for position, record in enumerate(candidate_records)
        if record["validation_mape_pct"] is not None
        and np.isfinite(float(record["validation_mape_pct"]))
    ]
    if not valid_positions:
        raise RuntimeError("Every SARIMA validation candidate failed.")
    best_position = min(
        valid_positions,
        key=lambda position: float(candidate_records[position]["validation_mape_pct"]),
    )
    selected = CANDIDATES[best_position]

    final_fit = _fit(splits.train_validation, selected, low_memory=False)
    forecast_object = final_fit.get_forecast(steps=len(splits.test))
    forecast = pd.Series(
        np.asarray(forecast_object.predicted_mean, dtype=float),
        index=splits.test.index,
        name="forecast",
    )
    summary = forecast_object.summary_frame(alpha=0.05)
    standard_error = pd.Series(
        summary["mean_se"].to_numpy(dtype=float), index=splits.test.index
    )
    quantiles = normal_quantile_series(forecast, standard_error)

    hyperparameters: dict[str, Any] = {
        **selected,
        "seasonal_period_hours": 24,
        "max_iterations": MAX_ITERATIONS,
        "validation_mape_pct": candidate_records[best_position][
            "validation_mape_pct"
        ],
        "validation_candidates": candidate_records,
        "final_converged": bool(final_fit.mle_retvals.get("converged", False)),
        "final_aic": float(final_fit.aic),
    }
    return ForecastResult(
        name="sarima",
        forecast=forecast,
        runtime_seconds=time.perf_counter() - started,
        hyperparameters=hyperparameters,
        quantiles=quantiles,
    )
