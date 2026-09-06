from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import ModelResult, mape, validation_blocks


def run(train: pd.Series, pretest: pd.Series, future: pd.DatetimeIndex) -> ModelResult:
    start = time.perf_counter()
    validation = np.concatenate([history.iloc[-168:].to_numpy()[:len(target)]
                                 for history, target in validation_blocks(pretest, len(train))])
    point = pretest.loc[future - pd.Timedelta(hours=168)].to_numpy()
    score = mape(pretest.iloc[len(train):].to_numpy(), validation)
    return ModelResult("naive", point, None, validation, {"lag_hours": 168},
                       [{"validation_mape_pct": score}], time.perf_counter() - start)
