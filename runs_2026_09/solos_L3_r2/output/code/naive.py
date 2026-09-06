from __future__ import annotations

from time import perf_counter


from code.common import DataSplits, ForecastResult


def run(splits: DataSplits) -> ForecastResult:
    started = perf_counter()
    forecast = splits.full.shift(168).reindex(splits.test.index)
    if forecast.isna().any():
        raise ValueError("Seasonal-naive forecast is missing one or more weekly lags")
    return ForecastResult(
        name="naive",
        forecast=forecast.rename("naive"),
        runtime_seconds=perf_counter() - started,
        hyperparameters={"seasonal_lag_hours": 168},
    )
