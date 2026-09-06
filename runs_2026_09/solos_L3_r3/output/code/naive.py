from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import ForecastResult


def run_naive(train_validation: pd.Series, test_index: pd.DatetimeIndex) -> ForecastResult:
    started = time.perf_counter()
    lagged_index = test_index - pd.Timedelta(hours=168)
    point = train_validation.reindex(lagged_index).to_numpy(dtype=float)
    if np.isnan(point).any():
        raise ValueError("Seasonal-naive forecast is missing one or more 168-hour lags")
    return ForecastResult(
        name="naive",
        point=point,
        runtime_seconds=time.perf_counter() - started,
        hyperparameters={"lag_hours": 168},
    )
