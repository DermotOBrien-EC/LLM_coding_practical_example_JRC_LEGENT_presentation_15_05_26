from __future__ import annotations

from time import perf_counter
import pandas as pd
from .common import ModelResult


def run(pretest: pd.Series) -> ModelResult:
    start = perf_counter()
    point = pretest.iloc[-168:].to_numpy().copy()
    return ModelResult(
        "naive", point, None, perf_counter() - start, {"seasonal_lag_hours": 168}, []
    )
