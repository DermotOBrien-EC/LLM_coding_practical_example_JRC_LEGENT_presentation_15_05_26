"""Seasonal-naive baseline: next week looks exactly like last week.

The forecast for hour t is the observed load at hour t - 168 h. There is
nothing to fit and nothing to tune, so the validation step is skipped.
Every other model has to beat this to earn its keep.
"""

from __future__ import annotations

import pandas as pd

from common import HORIZON, ForecastResult, Stopwatch, block_mape, validation_blocks


def seasonal_naive(history: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Copy the value from exactly one week earlier for every hour in `index`."""
    source = index - pd.Timedelta(hours=HORIZON)
    return pd.Series(history.reindex(source).to_numpy(), index=index)


def run(train: pd.Series, val: pd.Series, test_index: pd.DatetimeIndex) -> ForecastResult:
    with Stopwatch() as sw:
        history = pd.concat([train, val])
        # The validation score is reported for context only; there is
        # nothing to select.
        val_forecasts = [seasonal_naive(history, block) for block in validation_blocks(val.index)]
        point = seasonal_naive(history, test_index)
    return ForecastResult(
        name="naive",
        point=point,
        quantiles=None,
        runtime_seconds=sw.seconds,
        hyperparameters={"lag_hours": HORIZON},
        validation={"val_mape_pct": block_mape(val, val_forecasts)},
    )
