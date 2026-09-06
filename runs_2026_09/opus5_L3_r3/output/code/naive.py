"""Seasonal-naive baseline: this hour next week looks like this hour last week.

The forecast for hour t is simply the load observed at t minus 168 hours.
It has no parameters, so there is nothing to tune and nothing to refit; the
only thing it needs is the week of data immediately before the test window.

It is here as the anchor. A model that cannot beat "same hour last week" has
not earned its complexity.
"""

from __future__ import annotations

import time

import pandas as pd

from common import HORIZON, TEST_END, TEST_START, ForecastResult


def run(series: pd.Series) -> ForecastResult:
    """Produce the seasonal-naive forecast for the test window."""
    start = time.perf_counter()

    test_index = series.loc[TEST_START:TEST_END].index
    lagged_index = test_index - pd.Timedelta(hours=HORIZON)
    if lagged_index.max() >= TEST_START:
        raise ValueError("naive baseline would read from the test window")

    point = pd.Series(series.loc[lagged_index].to_numpy(), index=test_index)

    return ForecastResult(
        name="naive",
        point=point,
        quantiles={},  # deliberately none: the baseline is point-only
        hyperparameters={"seasonal_lag_hours": HORIZON},
        runtime_seconds=time.perf_counter() - start,
    )
