from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from code.common import DataSplits, ForecastResult, mape_pct, make_utc_series

SCALE_MW = 10_000.0
SEASONAL_PERIOD_HOURS = 24
CANDIDATE_AR_ORDERS = [1, 24, 168]
N_PROBABILISTIC_PATHS = 2_000


def _daily_difference(series: pd.Series) -> np.ndarray:
    return series.diff(SEASONAL_PERIOD_HOURS).dropna().to_numpy(dtype=float) / SCALE_MW


def _fit(series: pd.Series, ar_order: int) -> ARIMAResults:
    differenced = _daily_difference(series)
    model = ARIMA(
        differenced,
        order=(ar_order, 0, 0),
        trend="n",
        enforce_stationarity=True,
    )
    return model.fit(
        method="burg",
        cov_type="none",
        low_memory=True,
    )


def _forecast_differences(
    fitted: ARIMAResults,
    differenced_history: np.ndarray,
    steps: int,
) -> np.ndarray:
    ar_parameters = np.asarray(fitted.arparams, dtype=float)
    state = differenced_history[-len(ar_parameters) :].astype(float).tolist()
    output: list[float] = []
    for _ in range(steps):
        recent = np.asarray(state[-len(ar_parameters) :][::-1], dtype=float)
        prediction = float(np.dot(ar_parameters, recent))
        output.append(prediction)
        state.append(prediction)
    return np.asarray(output, dtype=float)


def _integrate_daily_differences(
    difference_forecast: np.ndarray,
    load_history: pd.Series,
) -> np.ndarray:
    base_day = load_history.iloc[-SEASONAL_PERIOD_HOURS:].to_numpy(dtype=float) / SCALE_MW
    reconstructed = np.empty(len(difference_forecast), dtype=float)
    for step, difference in enumerate(difference_forecast):
        seasonal_base = (
            base_day[step]
            if step < SEASONAL_PERIOD_HOURS
            else reconstructed[step - SEASONAL_PERIOD_HOURS]
        )
        reconstructed[step] = seasonal_base + difference
    return reconstructed * SCALE_MW


def _simulate_load_paths(
    fitted: ARIMAResults,
    differenced_history: np.ndarray,
    load_history: pd.Series,
    steps: int,
    repetitions: int,
) -> np.ndarray:
    ar_parameters = np.asarray(fitted.arparams, dtype=float)
    ar_order = len(ar_parameters)
    state = np.repeat(
        differenced_history[-ar_order:][None, :],
        repetitions,
        axis=0,
    )
    innovation_std = float(np.sqrt(fitted.params[-1]))
    random = np.random.default_rng(2020)
    difference_paths = np.empty((steps, repetitions), dtype=float)
    for step in range(steps):
        conditional_mean = state[:, ::-1] @ ar_parameters
        simulated = conditional_mean + random.normal(
            loc=0.0,
            scale=innovation_std,
            size=repetitions,
        )
        difference_paths[step] = simulated
        state[:, :-1] = state[:, 1:]
        state[:, -1] = simulated

    base_day = load_history.iloc[-SEASONAL_PERIOD_HOURS:].to_numpy(dtype=float) / SCALE_MW
    load_paths = np.empty_like(difference_paths)
    for step in range(steps):
        seasonal_base = (
            base_day[step]
            if step < SEASONAL_PERIOD_HOURS
            else load_paths[step - SEASONAL_PERIOD_HOURS]
        )
        load_paths[step] = seasonal_base + difference_paths[step]
    return load_paths * SCALE_MW


def run(splits: DataSplits) -> ForecastResult:
    started = perf_counter()
    validation_scores: dict[str, float] = {}
    successful: list[tuple[float, int]] = []
    train_differences = _daily_difference(splits.train)

    for ar_order in CANDIDATE_AR_ORDERS:
        fitted = _fit(splits.train, ar_order)
        difference_forecast = _forecast_differences(
            fitted,
            train_differences,
            len(splits.validation),
        )
        validation_forecast = _integrate_daily_differences(
            difference_forecast,
            splits.train,
        )
        score = mape_pct(splits.validation, validation_forecast)
        validation_scores[f"p={ar_order}"] = score
        successful.append((score, ar_order))

    validation_mape, chosen_ar_order = min(successful, key=lambda item: item[0])
    final_fit = _fit(splits.train_validation, chosen_ar_order)
    combined_differences = _daily_difference(splits.train_validation)
    final_difference_forecast = _forecast_differences(
        final_fit,
        combined_differences,
        len(splits.test),
    )
    point_forecast = _integrate_daily_differences(
        final_difference_forecast,
        splits.train_validation,
    )
    paths = _simulate_load_paths(
        final_fit,
        combined_differences,
        splits.train_validation,
        len(splits.test),
        N_PROBABILISTIC_PATHS,
    )

    quantiles: dict[float, pd.Series] = {}
    for quantile in (0.025, 0.1, 0.5, 0.9, 0.975):
        values = np.quantile(paths, quantile, axis=1)
        quantiles[quantile] = make_utc_series(
            values,
            splits.test.index,
            f"sarima_q{quantile}",
        )

    hyperparameters = {
        "order": [chosen_ar_order, 0, 0],
        "seasonal_order": [0, 1, 0, SEASONAL_PERIOD_HOURS],
        "estimator": "Burg conditional AR fit after explicit daily differencing",
        "candidate_ar_orders": CANDIDATE_AR_ORDERS,
        "candidate_validation_mape_pct": validation_scores,
        "probabilistic_paths": N_PROBABILISTIC_PATHS,
        "interval_method": "Gaussian AR innovation simulation",
    }
    return ForecastResult(
        name="sarima",
        forecast=make_utc_series(point_forecast, splits.test.index, "sarima"),
        runtime_seconds=perf_counter() - started,
        hyperparameters=hyperparameters,
        quantiles=quantiles,
        validation_mape_pct=validation_mape,
    )
