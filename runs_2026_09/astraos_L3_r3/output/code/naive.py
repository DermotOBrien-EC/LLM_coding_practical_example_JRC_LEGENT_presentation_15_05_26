from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd

from .common import HORIZON, ForecastResult, mape, validation_blocks


def run(pretest: pd.Series) -> ForecastResult:
    start = perf_counter()
    actual, prediction = [], []
    for history, target in validation_blocks(pretest):
        actual.extend(target.values)
        prediction.extend(history.iloc[-HORIZON:].values[: len(target)])
    return ForecastResult(
        name="naive",
        point=pretest.iloc[-HORIZON:].to_numpy().copy(),
        quantiles=None,
        runtime_seconds=perf_counter() - start,
        hyperparameters={"seasonal_lag_hours": HORIZON},
        validation=[
            {
                "mape_validation_pct": mape(np.array(actual), np.array(prediction)),
                "configuration": "fixed baseline; no selection",
            }
        ],
    )
