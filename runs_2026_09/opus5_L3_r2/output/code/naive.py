"""Seasonal-naive baseline: today's forecast is what happened last week.

The forecast for hour t is simply the observed load at hour t - 168 (the same
hour, seven days earlier). It has no parameters, so there is nothing to tune.
It is here as the reference point: an approach that cannot beat "look at last
week" is not earning its complexity.

One thing worth flagging up front. The test week is 1-7 January 2020, so the
week it copies is 25-31 December 2019 - Christmas. German load over Christmas
is unusually low, so this particular baseline starts at a disadvantage. That
is not a flaw in the method, it is the method honestly showing what happens
when a naive rule meets a holiday period.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import HORIZON, ModelForecast, TEST_START, horizon_index


def seasonal_naive_forecast(
    history: pd.Series, origin: pd.Timestamp, horizon: int = HORIZON
) -> np.ndarray:
    """Predict the horizon by copying the 168 hours that ended at the origin."""
    lag = pd.Timedelta(hours=168)
    wanted = horizon_index(origin, horizon) - lag
    missing = wanted.difference(history.index)
    if len(missing) > 0:
        raise ValueError(f"seasonal naive needs {len(missing)} hours it cannot see")
    return history.loc[wanted].to_numpy(dtype=float)


def run(series: pd.Series) -> ModelForecast:
    """Fit (nothing to fit) and forecast the 168 test hours."""
    started = time.perf_counter()
    history = series.loc[series.index < TEST_START]
    values = seasonal_naive_forecast(history, TEST_START)
    point = pd.Series(values, index=horizon_index(TEST_START), name="naive")
    return ModelForecast(
        name="naive",
        point=point,
        hyperparameters={"seasonal_lag_hours": 168},
        runtime_seconds=time.perf_counter() - started,
        notes=(
            "No parameters and no validation step. The copied week is "
            "25-31 December 2019, which is Christmas week."
        ),
    )
