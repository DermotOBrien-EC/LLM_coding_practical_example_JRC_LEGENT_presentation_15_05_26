from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from darts.models.forecasting.prophet_model import Prophet

from common import (
    SEED,
    DataSplits,
    ForecastResult,
    Standardizer,
    darts_samples_to_quantiles,
    make_darts_series,
    mean_absolute_percentage_error,
)

CANDIDATES: list[dict[str, Any]] = [
    {"changepoint_prior_scale": 0.03, "seasonality_prior_scale": 10.0},
    {"changepoint_prior_scale": 0.10, "seasonality_prior_scale": 10.0},
    {"changepoint_prior_scale": 0.20, "seasonality_prior_scale": 5.0},
]
N_SAMPLES = 500


def _make_model(configuration: dict[str, Any]) -> Prophet:
    return Prophet(
        country_holidays="DE",
        random_state=SEED,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        interval_width=0.95,
        uncertainty_samples=N_SAMPLES,
        **configuration,
    )


def _point_series(prediction: Any, index: pd.DatetimeIndex) -> pd.Series:
    values = prediction.all_values(copy=False)
    return pd.Series(values[:, 0, 0].astype(float), index=index, name="forecast")


def _extract_decomposition(
    model: Prophet, scaler: Standardizer
) -> pd.DataFrame:
    if model.model is None:
        raise ValueError("The Prophet model has not been fitted.")

    component_ranges = {
        "trend": pd.date_range("2019-01-01", "2020-01-07 23:00:00", freq="h"),
        "weekly": pd.date_range("2020-01-01", periods=168, freq="h"),
        "yearly": pd.date_range("2019-01-01", "2019-12-31", freq="D"),
    }
    records: list[pd.DataFrame] = []
    for component, index in component_ranges.items():
        predicted = model.model.predict(pd.DataFrame({"ds": index}))
        if component not in predicted:
            raise ValueError(f"Prophet did not return its {component} component.")
        component_values = predicted[component].to_numpy(dtype=float)
        if component == "trend":
            component_values = scaler.inverse_array(component_values)
        else:
            component_values = component_values * scaler.standard_deviation
        records.append(
            pd.DataFrame(
                {
                    "timestamp": index,
                    "component": component,
                    "value": component_values,
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def run(splits: DataSplits) -> ForecastResult:
    started = time.perf_counter()

    train_scaler = Standardizer.fit(splits.train)
    train_scaled = train_scaler.transform(splits.train)
    train_series = make_darts_series(train_scaled)

    validation_records: list[dict[str, Any]] = []
    for configuration in CANDIDATES:
        candidate_started = time.perf_counter()
        candidate = _make_model(configuration)
        candidate.fit(train_series)
        prediction = candidate.predict(len(splits.validation), num_samples=1)
        scaled_forecast = _point_series(prediction, splits.validation.index)
        forecast = pd.Series(
            train_scaler.inverse_array(scaled_forecast.to_numpy(dtype=float)),
            index=splits.validation.index,
        )
        validation_records.append(
            {
                **configuration,
                "validation_mape_pct": mean_absolute_percentage_error(
                    splits.validation, forecast
                ),
                "fit_runtime_seconds": time.perf_counter() - candidate_started,
            }
        )

    best_position = int(
        np.argmin([record["validation_mape_pct"] for record in validation_records])
    )
    selected = dict(CANDIDATES[best_position])

    final_scaler = Standardizer.fit(splits.train_validation)
    final_series = make_darts_series(final_scaler.transform(splits.train_validation))
    final_model = _make_model(selected)
    final_model.fit(final_series)
    stochastic_prediction = final_model.predict(len(splits.test), num_samples=N_SAMPLES)
    quantiles = darts_samples_to_quantiles(
        stochastic_prediction, final_scaler, splits.test.index
    )
    point_forecast = quantiles[0.5].rename("forecast")
    decomposition = _extract_decomposition(final_model, final_scaler)

    hyperparameters: dict[str, Any] = {
        **selected,
        "country_holidays": "DE",
        "daily_seasonality": True,
        "weekly_seasonality": True,
        "yearly_seasonality": True,
        "seasonality_mode": "additive",
        "uncertainty_samples": N_SAMPLES,
        "validation_mape_pct": validation_records[best_position][
            "validation_mape_pct"
        ],
        "validation_candidates": validation_records,
    }
    return ForecastResult(
        name="prophet",
        forecast=point_forecast,
        runtime_seconds=time.perf_counter() - started,
        hyperparameters=hyperparameters,
        quantiles=quantiles,
        decomposition=decomposition,
    )
