"""Seasonal-naive baseline: the forecast for hour t is the load at t - 168 h.

There is nothing to fit or tune. The last observed week before the forecast
origin is simply repeated. Every other model has to beat this.
"""

from __future__ import annotations

import pandas as pd

from common import HORIZON, ForecastResult, Timer, rolling_validation_mape, validation_origins


def seasonal_naive(history: pd.Series, horizon: int) -> pd.Series:
    """Repeat the final week of `history` over the next `horizon` hours."""
    if len(history) < HORIZON:
        raise ValueError("need at least one week of history")
    last_week = history.iloc[-HORIZON:].values
    reps = -(-horizon // HORIZON)  # ceiling division
    values = list(last_week) * reps
    index = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    return pd.Series(values[:horizon], index=index, name="naive")


def run(train: pd.Series, val: pd.Series, horizon: int = HORIZON) -> ForecastResult:
    with Timer() as timer:
        # Validation score, for the record only: there is nothing to select.
        history = pd.concat([train, val])
        origins = validation_origins(val, horizon)
        val_forecasts = [seasonal_naive(history[: o - pd.Timedelta(hours=1)], horizon) for o in origins]
        val_mape = rolling_validation_mape(val, val_forecasts, origins)
        point = seasonal_naive(history, horizon)
    return ForecastResult(
        name="naive",
        point=point,
        quantiles=None,
        runtime_seconds=timer.seconds,
        hyperparameters={"lag_hours": HORIZON},
        validation_mape_pct=val_mape,
    )


if __name__ == "__main__":
    from common import load_series, split

    train, val, _ = split(load_series())
    result = run(train, val)
    print(f"validation MAPE {result.validation_mape_pct:.2f} %  runtime {result.runtime_seconds:.1f} s")
    print(result.point.head())
