"""Seasonal-naive baseline: yhat[t] = load[t - 168 h] (one week earlier).

This is the anchor every other model has to beat. It has no free parameters.
"""

from __future__ import annotations

import pandas as pd

from common import Splits, align


def run(splits: Splits) -> dict[str, object]:
    """Forecast the test window using the value from exactly one week before.

    The predecessor week is Dec 25-31 2019, which sits inside the refit set,
    so no lookup goes outside history.
    """
    history = splits.trainval
    test_index = splits.test.index
    forecast_vals = [history.loc[ts - pd.Timedelta(hours=168)] for ts in test_index]
    forecast = align(forecast_vals, test_index)
    return {
        "name": "naive",
        "forecast": forecast,
        "hyperparameters": {"lag_hours": 168},
        # No intervals for the baseline (spec says naive skips intervals).
        "lower_80": None,
        "upper_80": None,
        "lower_95": None,
        "upper_95": None,
        "quantiles": None,
    }
