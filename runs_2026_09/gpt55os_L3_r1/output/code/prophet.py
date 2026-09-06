from __future__ import annotations

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet

from common import ForecastResult, elapsed_seconds, mape_pct, timer


def _series(values: pd.Series) -> TimeSeries:
    return TimeSeries.from_series(values.astype(float), fill_missing_dates=False, freq="h")


def _fit(values: pd.Series, seasonality_prior_scale: float, changepoint_prior_scale: float) -> Prophet:
    model = Prophet(
        country_holidays="DE",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        seasonality_prior_scale=seasonality_prior_scale,
        changepoint_prior_scale=changepoint_prior_scale,
        random_state=20260906,
        suppress_stdout_stderror=True,
    )
    model.fit(_series(values), verbose=False)
    return model


def forecast(train: pd.Series, validation: pd.Series, train_validation: pd.Series, test: pd.Series) -> ForecastResult:
    start = timer()
    candidates = [
        {"seasonality_prior_scale": 5.0, "changepoint_prior_scale": 0.05},
        {"seasonality_prior_scale": 10.0, "changepoint_prior_scale": 0.05},
        {"seasonality_prior_scale": 10.0, "changepoint_prior_scale": 0.1},
    ]
    best_params: dict[str, float] | None = None
    best_score = float("inf")
    validation_scores: list[dict[str, float]] = []
    for params in candidates:
        model = _fit(train, **params)
        pred_series = model.predict(len(validation), num_samples=1, verbose=False)
        pred = pred_series.values(copy=False).reshape(-1).astype(float)
        score = mape_pct(validation.to_numpy(), pred)
        validation_scores.append({**params, "validation_mape_pct": score})
        if score < best_score:
            best_score = score
            best_params = params
    if best_params is None:
        raise RuntimeError("Prophet validation search produced no fitted model")

    model = _fit(train_validation, **best_params)
    pred_series = model.predict(len(test), num_samples=500, verbose=False, random_state=20260906)
    samples = pred_series.all_values(copy=False)[:, 0, :].astype(float)
    q025, q10, q25, q50, q75, q90, q975 = np.quantile(samples, [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975], axis=1)
    point = model.predict(len(test), num_samples=1, verbose=False).values(copy=False).reshape(-1).astype(float)
    raw_forecast = model.predict_raw(len(test), verbose=False)
    components = raw_forecast[[column for column in raw_forecast.columns if column in {"ds", "trend", "weekly", "yearly", "holidays"}]].copy()
    return ForecastResult(
        name="prophet",
        point=point,
        runtime_seconds=elapsed_seconds(start),
        hyperparameters={"selected": best_params, "validation_mape_pct": best_score, "validation_grid": validation_scores},
        validation_mape_pct=best_score,
        q025=q025,
        q10=q10,
        q25=q25,
        q50=q50,
        q75=q75,
        q90=q90,
        q975=q975,
        extra={"components": components},
    )
