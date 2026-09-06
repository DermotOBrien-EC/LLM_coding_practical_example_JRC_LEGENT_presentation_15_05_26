"""Seasonal-naive baseline.

The idea is deliberately dumb: to guess the load at some hour, just reuse the
load observed at the same hour exactly one week (168 hours) earlier. Electricity
demand has a very strong weekly rhythm (workdays look alike, Saturdays look
alike, Sundays look alike), so "same hour last week" is a surprisingly hard
baseline to beat. Every other model has to earn its keep against this.

There is nothing to fit and nothing to tune, so there is no validation step.
The forecast for the test week is simply the observed load from the week before.
"""

from __future__ import annotations

import time

import pandas as pd

import common as c


def run(splits: c.Splits) -> c.ModelResult:
    start = time.time()
    # For each test hour t, look up the load at t - 168h in the full series.
    # For the test week (2020-01-01..07) that reference week is 2019-12-25..31,
    # which is entirely inside the data we are allowed to use.
    lag = pd.Timedelta(hours=168)
    reference_index = splits.test.index - lag
    point = pd.Series(
        splits.full.loc[reference_index].to_numpy(),
        index=splits.test.index,
        name="naive",
    )
    return c.ModelResult(
        name="naive",
        point=point,
        runtime_s=time.time() - start,
        hyperparameters={"lag_hours": 168},
        quantiles={},  # the naive baseline produces no prediction intervals
    )
