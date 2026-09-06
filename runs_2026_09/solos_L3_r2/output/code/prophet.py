from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd
from darts import TimeSeries
from darts.models import Prophet

from code.common import DataSplits, ForecastResult, mape_pct, make_utc_series

CANDIDATES: list[dict[str, Any]] = [
    {"changepoint_prior_scale": 0.05, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.20, "seasonality_mode": "additive"},
    {"changepoint_prior_scale": 0.05, "seasonality_mode": "multiplicative"},
]


def _to_darts(series: pd.Series) -> TimeSeries:
    naive_utc = series.index.tz_convert("UTC").tz_localize(None)
    return TimeSeries.from_times_and_values(
        naive_utc,
        series.to_numpy(dtype="float32"),
        columns=["load_mw"],
        fill_missing_dates=False,
        freq="h",
    )


def _build(configuration: dict[str, Any]) -> Prophet:
    return Prophet(
        country_holidays="DE",
        random_state=2020,
        suppress_stdout_stderror=True,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=configuration["changepoint_prior_scale"],
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        seasonality_mode=configuration["seasonality_mode"],
        uncertainty_samples=1_000,
    )


def _configuration_label(configuration: dict[str, Any]) -> str:
    return (
        f"cps={configuration['changepoint_prior_scale']},mode={configuration['seasonality_mode']}"
    )


def run(splits: DataSplits) -> ForecastResult:
    started = perf_counter()
    train = _to_darts(splits.train)
    train_validation = _to_darts(splits.train_validation)
    validation_scores: dict[str, float] = {}
    successful: list[tuple[float, dict[str, Any]]] = []

    for configuration in CANDIDATES:
        model = _build(configuration)
        model.fit(train)
        forecast = model.predict(len(splits.validation), num_samples=1)
        score = mape_pct(splits.validation, forecast.univariate_values())
        validation_scores[_configuration_label(configuration)] = score
        successful.append((score, configuration))

    validation_mape, chosen = min(successful, key=lambda item: item[0])
    final_model = _build(chosen)
    final_model.fit(train_validation)
    point_forecast = final_model.predict(len(splits.test), num_samples=1)
    sample_forecast = final_model.predict(
        len(splits.test),
        num_samples=1_000,
        random_state=2020,
    )

    quantiles: dict[float, pd.Series] = {}
    for quantile in (0.025, 0.1, 0.5, 0.9, 0.975):
        values = sample_forecast.quantile(quantile).univariate_values()
        quantiles[quantile] = make_utc_series(
            values,
            splits.test.index,
            f"prophet_q{quantile}",
        )

    component_start = pd.Timestamp("2019-01-01 00:00:00")
    component_end = pd.Timestamp("2020-01-07 23:00:00")
    component_dates = pd.date_range(component_start, component_end, freq="h")
    component_frame = final_model.model.predict(pd.DataFrame({"ds": component_dates}))
    retained_columns = ["ds", "trend", "weekly", "yearly", "holidays", "yhat"]
    components = component_frame.loc[:, retained_columns].set_index("ds")

    hyperparameters = {
        "country_holidays": "DE",
        "daily_seasonality": True,
        "weekly_seasonality": True,
        "yearly_seasonality": True,
        "changepoint_prior_scale": chosen["changepoint_prior_scale"],
        "seasonality_mode": chosen["seasonality_mode"],
        "seasonality_prior_scale": 10.0,
        "holidays_prior_scale": 10.0,
        "candidate_validation_mape_pct": validation_scores,
        "probabilistic_samples": 1_000,
    }
    return ForecastResult(
        name="prophet",
        forecast=make_utc_series(
            point_forecast.univariate_values(),
            splits.test.index,
            "prophet",
        ),
        runtime_seconds=perf_counter() - started,
        hyperparameters=hyperparameters,
        quantiles=quantiles,
        validation_mape_pct=validation_mape,
        prophet_components=components,
    )
