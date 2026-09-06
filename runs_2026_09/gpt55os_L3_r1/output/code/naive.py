from __future__ import annotations

import numpy as np
import pandas as pd

from common import ForecastResult, elapsed_seconds, timer


def forecast(train_validation: pd.Series, test: pd.Series) -> ForecastResult:
    start = timer()
    forecast_values: list[float] = []
    for stamp in test.index:
        lag_stamp = stamp - pd.Timedelta(hours=168)
        forecast_values.append(float(train_validation.loc[lag_stamp]))
    return ForecastResult(
        name="naive",
        point=np.asarray(forecast_values, dtype=float),
        runtime_seconds=elapsed_seconds(start),
        hyperparameters={"lag_hours": 168},
    )
