from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet

from common import ModelResult, QUANTILES, SEED, mape


def make_model(mode: str) -> Prophet:
    return Prophet(country_holidays="DE", daily_seasonality=True,
                   weekly_seasonality=True, yearly_seasonality=True,
                   seasonality_mode=mode, changepoint_prior_scale=0.01,
                   holidays_prior_scale=10.0, uncertainty_samples=600,
                   random_state=SEED)


def as_series(y: pd.Series) -> TimeSeries:
    return TimeSeries.from_series(y, fill_missing_dates=False, freq="h")


def run(train: pd.Series, pretest: pd.Series, future: pd.DatetimeIndex) -> ModelResult:
    start = time.perf_counter()
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    candidates: list[dict[str, Any]] = []
    forecasts: list[np.ndarray] = []
    validation_frame = pd.DataFrame({"ds": pretest.index[len(train):]})
    for mode in ("additive", "multiplicative"):
        model = make_model(mode)
        model.fit(as_series(train))
        # Prophet's time-only prediction is identical across the fixed-origin blocks.
        prediction = model.model.predict(validation_frame)["yhat"].to_numpy()
        score = mape(pretest.iloc[len(train):].to_numpy(), prediction)
        candidates.append({"seasonality_mode": mode, "validation_mape_pct": score})
        forecasts.append(prediction)
        print(f"prophet mode={mode} validation MAPE={score:.3f}%", flush=True)
    chosen = int(np.argmin([c["validation_mape_pct"] for c in candidates]))
    mode = candidates[chosen]["seasonality_mode"]
    final = make_model(mode)
    final.fit(as_series(pretest))
    frame = pd.DataFrame({"ds": future})
    prediction_frame = final.model.predict(frame)
    np.random.seed(SEED)
    samples = final.model.predictive_samples(frame)["yhat"]
    quantiles = np.quantile(samples, QUANTILES, axis=1).T
    components = prediction_frame[["ds", "trend", "weekly", "yearly", "daily", "holidays"]].copy()
    reference_dates = {
        "decomposition_trend": pd.date_range(pretest.index[0], pretest.index[-1], freq="D"),
        "decomposition_weekly": pd.date_range("2019-01-07", periods=168, freq="h"),
        "decomposition_yearly": pd.date_range("2019-01-01", periods=365, freq="D"),
    }
    decomposition = {}
    for key, dates in reference_dates.items():
        component = key.removeprefix("decomposition_")
        frame_ref = final.model.predict(pd.DataFrame({"ds": dates}))
        decomposition[key] = frame_ref[["ds", component]]
    params = {"country_holidays": "DE", "daily_seasonality": True,
              "weekly_seasonality": True, "yearly_seasonality": True,
              "seasonality_mode": mode, "changepoint_prior_scale": 0.01,
              "holidays_prior_scale": 10.0, "uncertainty_samples": 600,
              "seed": SEED, "calendar_timezone": "UTC (library-naive timestamps)",
              "uncertainty": "Prophet predictive samples, MAP parameters, mcmc_samples=0"}
    return ModelResult("prophet", prediction_frame["yhat"].to_numpy(), quantiles,
                       forecasts[chosen], params, candidates, time.perf_counter() - start,
                       {"components": components, **decomposition})
