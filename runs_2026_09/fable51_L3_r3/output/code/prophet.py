"""Prophet with German public holidays, through the darts wrapper.

Prophet describes the series as a smooth trend plus daily, weekly and
yearly seasonal curves plus a bump on each public holiday. It does not
look at the most recent observations when forecasting, so its one-week
forecast is really a "what does a typical early-January week look like"
statement. The holiday terms are what should let it handle 1 January.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# common must come before darts: it fixes sys.path so that darts imports
# the real prophet package rather than this file (see common.py).
from common import (
    HORIZON,
    QUANTILES,
    SEED,
    ForecastResult,
    Timer,
    rolling_validation_mape,
    validation_origins,
)
from darts import TimeSeries  # noqa: E402
from darts.models import Prophet  # noqa: E402

N_SAMPLES = 1000

# Fourier orders control how wiggly each seasonal curve may be. Prophet's
# defaults (yearly 10, weekly 3, daily 4) are meant for daily data, so
# richer daily and weekly shapes are the main thing worth trying.
CANDIDATES: list[dict[str, Any]] = [
    {"yearly_seasonality": 10, "weekly_seasonality": 3, "daily_seasonality": 4,
     "changepoint_prior_scale": 0.05, "seasonality_mode": "additive"},
    {"yearly_seasonality": 10, "weekly_seasonality": 10, "daily_seasonality": 12,
     "changepoint_prior_scale": 0.05, "seasonality_mode": "additive"},
    {"yearly_seasonality": 10, "weekly_seasonality": 10, "daily_seasonality": 12,
     "changepoint_prior_scale": 0.05, "seasonality_mode": "multiplicative"},
    {"yearly_seasonality": 10, "weekly_seasonality": 10, "daily_seasonality": 12,
     "changepoint_prior_scale": 0.01, "seasonality_mode": "multiplicative"},
]


def to_timeseries(series: pd.Series) -> TimeSeries:
    """darts wants a timezone-naive index; the clock stays UTC."""
    naive = series.copy()
    naive.index = naive.index.tz_localize(None)
    return TimeSeries.from_series(naive)


def to_pandas(ts: TimeSeries, name: str) -> pd.Series:
    out = ts.to_series() if ts.n_components == 1 else ts.to_dataframe().iloc[:, 0]
    out.index = pd.DatetimeIndex(out.index).tz_localize("UTC")
    return out.rename(name)


def make_model(params: dict[str, Any]) -> Prophet:
    return Prophet(country_holidays="DE", random_state=SEED, **params)


def run(train: pd.Series, val: pd.Series, horizon: int = HORIZON) -> ForecastResult:
    with Timer() as timer:
        origins = validation_origins(val, horizon)
        n_val_hours = len(origins) * horizon
        train_ts = to_timeseries(train)

        # 1. Fit each candidate on Train, forecast the validation weeks.
        #    Prophet does not condition on recent data, so forecasting the
        #    13 weeks in one go is identical to 13 separate one-week forecasts.
        scores: list[tuple[float, dict[str, Any]]] = []
        for params in CANDIDATES:
            model = make_model(params)
            model.fit(train_ts)
            val_fc = to_pandas(model.predict(n_val_hours), "prophet")
            forecasts = [val_fc[o : o + pd.Timedelta(hours=horizon - 1)] for o in origins]
            score = rolling_validation_mape(val, forecasts, origins)
            scores.append((score, params))
            print(f"  prophet {params}: validation MAPE {score:.2f} %", flush=True)
        best_score, best_params = min(scores, key=lambda s: s[0])

        # 2. Refit on Train + Validation, forecast the test week.
        full_ts = to_timeseries(pd.concat([train, val]))
        model = make_model(best_params)
        model.fit(full_ts)
        point = to_pandas(model.predict(horizon, num_samples=1), "prophet")
        samples = model.predict(horizon, num_samples=N_SAMPLES).all_values()[:, 0, :]
        index = point.index
        quantiles = {
            tau: pd.Series(np.quantile(samples, tau, axis=1), index=index) for tau in QUANTILES
        }

    return ForecastResult(
        name="prophet",
        point=point,
        quantiles=quantiles,
        runtime_seconds=timer.seconds,
        hyperparameters={
            **best_params,
            "country_holidays": "DE",
            "uncertainty_samples": N_SAMPLES,
            "candidates_scored": len(CANDIDATES),
        },
        validation_mape_pct=best_score,
        extras={"prophet_model": model.model, "validation_scores": scores},
    )


if __name__ == "__main__":
    from common import load_series, split

    train, val, _ = split(load_series())
    result = run(train, val)
    print(f"validation MAPE {result.validation_mape_pct:.2f} %  runtime {result.runtime_seconds:.1f} s")
    print(result.point.head())
