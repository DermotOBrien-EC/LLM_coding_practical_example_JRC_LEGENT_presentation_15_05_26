"""Seasonal naive baseline: today looks like the same hour last week.

The forecast for hour t is simply the load observed at hour t minus 168
hours. It has no parameters to tune, so there is nothing to select on the
validation window. It is here as the yardstick: any model that cannot beat
"repeat last week" is not earning its complexity.

It produces no prediction intervals, which is exactly the point. Getting
uncertainty out of it would require assumptions the method itself does not
make.
"""

from __future__ import annotations

import time

import pandas as pd

import common as c


def seasonal_naive_forecast(history: pd.Series, index: pd.DatetimeIndex, lag_hours: int = 168) -> pd.Series:
    """Copy the value from `lag_hours` earlier for every timestamp in `index`.

    `history` must already contain those earlier timestamps.
    """
    source_index = index - pd.Timedelta(hours=lag_hours)
    missing = source_index.difference(history.index)
    if len(missing) > 0:
        raise ValueError(f"History is missing {len(missing)} timestamps needed for the naive lag.")
    return pd.Series(history.loc[source_index].to_numpy(), index=index, name="naive")


def validation_mape(series: pd.Series) -> float:
    """Score the baseline on the same rolling 168-hour blocks the other models use."""
    known = series.loc[: c.VAL_END]  # never look past the validation window
    actual = known.loc[c.VAL_START : c.VAL_END]
    predicted = seasonal_naive_forecast(known, actual.index)
    return c.mape(actual.to_numpy(), predicted.to_numpy())


def run(series: pd.Series) -> c.ForecastResult:
    """Produce the baseline's forecast for the held-out test week."""
    start = time.time()

    history = c.train_plus_val(series)  # everything before 2020-01-01
    test = series.loc[c.TEST_START : c.TEST_END]
    forecast = seasonal_naive_forecast(history, test.index)

    return c.ForecastResult(
        name="naive",
        point_forecast=forecast,
        runtime_seconds=time.time() - start,
        hyperparameters={"lag_hours": 168, "note": "no parameters to select"},
        quantile_forecast=None,
        validation_mape_pct=validation_mape(series),
    )


if __name__ == "__main__":
    result = run(c.load_load_series())
    result.save()
    print(result.name, result.validation_mape_pct, result.runtime_seconds)
