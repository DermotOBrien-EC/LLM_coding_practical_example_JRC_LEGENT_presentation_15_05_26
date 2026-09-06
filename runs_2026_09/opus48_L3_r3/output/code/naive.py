"""Seasonal-naive baseline.

The idea is as simple as it gets: to guess the load at some hour, use the load
at the same hour exactly one week (168 hours) earlier. Electricity demand has a
strong weekly rhythm (workdays look alike, weekends look alike), so 'last week,
same hour' is a surprisingly hard baseline to beat. Everything fancier in this
study has to earn its keep against this line.

It has no hyperparameters, so there is nothing to tune on the validation set.
It also produces no prediction intervals.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

import common as c


def run(series: pd.Series) -> c.ModelResult:
    """Produce the 168-hour seasonal-naive forecast for the test week."""
    start = time.perf_counter()

    # For each test hour t, look up the observed load at t - 168 hours. Those
    # source hours (2019-12-25 to 2019-12-31) all sit inside the data we are
    # allowed to use, so no test-window value is ever peeked at.
    test_index = c.test_series(series).index
    source_index = test_index - pd.Timedelta(hours=c.HORIZON)
    point = series.loc[source_index].to_numpy(dtype=float)

    runtime = time.perf_counter() - start
    return c.ModelResult(
        name="naive",
        point=point,
        runtime_seconds=runtime,
        hyperparameters={"lag_hours": c.HORIZON},
        quantiles=None,
    )
