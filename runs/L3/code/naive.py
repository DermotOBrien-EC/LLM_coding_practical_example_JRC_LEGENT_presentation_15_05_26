"""Seasonal-naive baseline.

The forecast for hour t is whatever was observed at hour t - 168 (same hour
of the same weekday, one week earlier). This is the cheapest defensible
hourly-load baseline and it sets the floor that every other model is asked
to beat. No fitting, no hyperparameters — only a lag lookup.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from code.common import N_TEST, TEST_END, TEST_START, ModelForecast


def run_naive(series: pd.Series) -> ModelForecast:
    """Return the seasonal-naive forecast for the test window."""
    started = time.perf_counter()
    one_week = pd.Timedelta("168h")
    point_index = pd.date_range(TEST_START, TEST_END, freq="h")
    assert len(point_index) == N_TEST
    point = series.loc[point_index - one_week].to_numpy()
    return ModelForecast(
        name="naive",
        point=point,
        runtime_seconds=time.perf_counter() - started,
        hyperparameters={"lag_hours": 168},
    )
