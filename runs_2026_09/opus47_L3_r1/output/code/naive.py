"""Seasonal-naive baseline: forecast(t) = observed(t - 168h).

One week's lag captures both the daily cycle and the weekday pattern in
electricity demand. Every other model has to beat this to earn its keep.
"""
from __future__ import annotations

import pandas as pd

from common import Splits


def forecast(splits: Splits) -> pd.Series:
    """Return the 168-hour test forecast, indexed like splits.test."""
    # The forecast for each test hour is the observation 168 hours earlier,
    # which lies entirely inside the train+val window (last such lag is
    # 2019-12-31 23:00, still in the fit set).
    pred = splits.series.shift(168).reindex(splits.test.index)
    if pred.isna().any():
        raise RuntimeError("Naive forecast produced NaN, unexpected.")
    pred.name = "naive"
    return pred


def hyperparameters() -> dict[str, object]:
    return {"lag_hours": 168}
