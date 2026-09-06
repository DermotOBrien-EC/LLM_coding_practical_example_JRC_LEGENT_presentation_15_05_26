from __future__ import annotations

import time

import pandas as pd

from common import DataSplits, ForecastResult


def run(splits: DataSplits) -> ForecastResult:
    started = time.perf_counter()
    source_index = splits.test.index - pd.Timedelta(hours=168)
    forecast = pd.Series(
        splits.full.reindex(source_index).to_numpy(dtype=float),
        index=splits.test.index,
        name="forecast",
    )
    if forecast.isna().any():
        raise ValueError("The seasonal-naive forecast is missing one or more weekly lags.")
    return ForecastResult(
        name="naive",
        forecast=forecast,
        runtime_seconds=time.perf_counter() - started,
        hyperparameters={"lag_hours": 168, "validation_required": False},
    )
