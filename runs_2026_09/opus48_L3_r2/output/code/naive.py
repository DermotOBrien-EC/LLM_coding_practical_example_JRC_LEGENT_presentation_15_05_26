"""Seasonal-naive baseline.

The idea is as simple as forecasting gets: whatever the load was at this same
hour one week ago, that is the forecast for now. Electricity demand has a very
strong weekly rhythm (weekday mornings look like other weekday mornings,
Sundays look like Sundays), so "same hour last week" is a genuinely hard
baseline to beat. Every other model has to earn its keep against this.

There is nothing to tune, so there is no validation step. The forecast for the
test week uses the observed load from the week before, which lives entirely in
the training data (Dec 25-31 2019).
"""

from __future__ import annotations

import time

import numpy as np

import common as C


def run(bundle: C.DataBundle) -> C.ForecastResult:
    start = time.time()

    # For each test hour t, look up the observed load at t - 168 hours.
    lag = 168
    forecast = np.empty(len(bundle.test), dtype=float)
    for i, ts in enumerate(bundle.test_index):
        source_ts = ts - np.timedelta64(lag, "h")
        forecast[i] = bundle.full.loc[source_ts]

    runtime = time.time() - start
    return C.ForecastResult(
        name="naive",
        point=forecast,
        runtime_seconds=runtime,
        hyperparameters={"lag_hours": lag},
        val_mape=None,  # no tuning for a parameter-free baseline
        quantiles={},  # the naive baseline does not produce intervals
    )


if __name__ == "__main__":
    b = C.build_bundle()
    res = run(b)
    print("naive test MAPE:", C.mape(b.test.to_numpy(), res.point))
