from __future__ import annotations

import time

import pandas as pd

from common import ForecastResult, LOAD_COL, MODEL_LABELS, TEST_START, TEST_END, TIME_COL, series_from_frame


def _seasonal_naive(source: pd.DataFrame, target_times: pd.Series) -> pd.Series:
    source_series = series_from_frame(source)
    values: list[float] = []
    for ts in target_times:
        values.append(float(source_series.loc[ts - pd.Timedelta(hours=168)]))
    return pd.Series(values, index=target_times, name="forecast")


def run(df: pd.DataFrame) -> ForecastResult:
    start = time.perf_counter()
    test_times = df.loc[(df[TIME_COL] >= TEST_START) & (df[TIME_COL] <= TEST_END), TIME_COL]
    forecast = _seasonal_naive(df[[TIME_COL, LOAD_COL]], test_times)
    return ForecastResult(
        name="naive",
        display_name=MODEL_LABELS["naive"],
        forecast=forecast,
        runtime_seconds=time.perf_counter() - start,
        hyperparameters={"lag_hours": 168},
    )
