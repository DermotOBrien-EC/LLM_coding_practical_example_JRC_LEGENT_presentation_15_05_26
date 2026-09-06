"""Seasonal-naive baseline: forecast at hour t is the observed load at t - 168 h.

Same hour, one week earlier. This is the natural anchor for load forecasting
because the load pattern is dominated by day-of-week x hour-of-day structure
and holidays. Everything else in the study should beat this.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import Split


@dataclass
class NaiveResult:
    forecast: pd.Series
    runtime_seconds: float
    hyperparameters: dict[str, object]


def run_naive(split: Split) -> NaiveResult:
    """Take load(t - 168 h) for every test hour."""
    t0 = time.perf_counter()
    history = split.train_plus_val
    forecast_values = []
    lag = pd.Timedelta(hours=168)
    for ts in split.test.index:
        forecast_values.append(float(history.loc[ts - lag]))
    forecast = pd.Series(np.asarray(forecast_values, dtype=float), index=split.test.index, name="naive")
    runtime = time.perf_counter() - t0
    return NaiveResult(forecast=forecast, runtime_seconds=runtime, hyperparameters={"lag_hours": 168})
