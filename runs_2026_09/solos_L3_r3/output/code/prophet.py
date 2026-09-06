from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet

from common import ForecastResult, QUANTILES, SEED, enforce_quantile_order, mape_pct

CANDIDATE_CHANGEPOINT_PRIORS = (0.01, 0.05, 0.15)


def _to_darts_series(series: pd.Series) -> TimeSeries:
    return TimeSeries.from_times_and_values(
        series.index,
        series.to_numpy(dtype=np.float32),
        columns=["load"],
        fill_missing_dates=False,
    )


def _make_model(changepoint_prior_scale: float) -> Prophet:
    return Prophet(
        country_holidays="DE",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        uncertainty_samples=0,
        random_state=SEED,
        suppress_stdout_stderror=True,
    )


def _values(series: TimeSeries) -> np.ndarray:
    return np.asarray(series.values(copy=False), dtype=float).reshape(-1)


def run_prophet(
    train: pd.Series,
    validation: pd.Series,
    train_validation: pd.Series,
    test_index: pd.DatetimeIndex,
) -> ForecastResult:
    started = time.perf_counter()
    train_series = _to_darts_series(train)
    validation_records: list[dict[str, Any]] = []
    best_prior: float | None = None
    best_mape = float("inf")

    for prior in CANDIDATE_CHANGEPOINT_PRIORS:
        model = _make_model(prior)
        model.fit(train_series)
        forecast = _values(model.predict(len(validation), num_samples=1))
        candidate_mape = mape_pct(validation.to_numpy(), forecast)
        validation_records.append(
            {
                "changepoint_prior_scale": prior,
                "validation_mape_pct": candidate_mape,
            }
        )
        if candidate_mape < best_mape:
            best_mape = candidate_mape
            best_prior = prior

    if best_prior is None:
        raise RuntimeError("Prophet validation produced no candidate")

    final_model = _make_model(best_prior)
    final_model.fit(_to_darts_series(train_validation))
    point = _values(final_model.predict(n=len(test_index), num_samples=1))
    probabilistic = final_model.predict(
        n=len(test_index),
        num_samples=1_000,
        random_state=SEED,
    )
    samples = np.asarray(probabilistic.all_values(copy=False), dtype=float)[:, 0, :]
    quantiles = enforce_quantile_order(
        {probability: np.quantile(samples, probability, axis=1) for probability in QUANTILES}
    )

    future = pd.DataFrame({"ds": pd.DatetimeIndex(test_index)})
    component_frame = final_model.model.predict(future)
    decomposition = component_frame.set_index("ds")[["trend", "weekly", "yearly"]]
    return ForecastResult(
        name="prophet",
        point=point,
        quantiles=quantiles,
        runtime_seconds=time.perf_counter() - started,
        hyperparameters={
            "country_holidays": "DE",
            "daily_seasonality": True,
            "weekly_seasonality": True,
            "yearly_seasonality": True,
            "changepoint_prior_scale": best_prior,
            "seasonality_prior_scale": 10.0,
            "holidays_prior_scale": 10.0,
            "selection_metric": "validation_mape_pct",
            "validation_mape_pct": best_mape,
            "candidates": validation_records,
            "point_forecast": "deterministic_yhat",
            "forecast_samples": 1_000,
            "base_uncertainty_samples": 0,
        },
        extras={"decomposition": decomposition},
    )
