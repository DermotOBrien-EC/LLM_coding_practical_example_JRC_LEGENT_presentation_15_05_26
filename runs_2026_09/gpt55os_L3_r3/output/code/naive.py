from __future__ import annotations

import pandas as pd

from common import ModelOutput, TEST_END, TEST_START


def run(series: pd.Series) -> ModelOutput:
    test_index = pd.date_range(TEST_START, TEST_END, freq="h")
    forecast = series.reindex(test_index - pd.Timedelta(hours=168)).copy()
    forecast.index = test_index
    forecast.name = "forecast"
    return ModelOutput(
        name="naive",
        forecast=forecast,
        runtime_seconds=0.0,
        hyperparameters={"lag_hours": 168},
    )
