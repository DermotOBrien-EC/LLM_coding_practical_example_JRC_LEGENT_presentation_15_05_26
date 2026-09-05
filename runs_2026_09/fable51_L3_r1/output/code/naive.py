"""Seasonal-naive baseline: forecast = the load observed one week earlier.

There is nothing to fit. The forecast for hour t is simply the value at
t - 168 h, so the forecast for the whole test week is the last week of
December 2019 copied forward. It has no prediction intervals.
"""

from __future__ import annotations

import time

import pandas as pd

import common


def seasonal_naive(history: pd.Series, origin: pd.Timestamp, horizon: int = common.HORIZON) -> pd.Series:
    """Copy the last observed week forward, once per horizon block."""
    idx = common.horizon_index(origin, horizon)
    values = [history.loc[t - pd.Timedelta(hours=168)] if (t - pd.Timedelta(hours=168)) in history.index else float("nan") for t in idx]
    fc = pd.Series(values, index=idx, dtype=float)
    # Beyond one week ahead the lagged value would fall inside the horizon
    # itself; there it is filled by repeating the same week again.
    if fc.isna().any():
        week = fc.iloc[:168].to_numpy()
        for i in range(168, horizon):
            fc.iloc[i] = week[i % 168]
    return fc


def run(s: pd.Series) -> common.ModelResult:
    t0 = time.perf_counter()
    val_mape, val_table = common.rolling_origin_mape(s, seasonal_naive)
    history = common.history_before(s, common.TEST_START)
    point = seasonal_naive(history, common.TEST_START)
    return common.ModelResult(
        name="naive",
        point=point,
        quantiles=None,
        hyperparameters={"lag_hours": 168},
        runtime_seconds=time.perf_counter() - t0,
        validation_mape_pct=val_mape,
        validation_table=val_table,
        notes="No parameters. The test-week forecast is the observed load of 25 to 31 December 2019.",
    )


if __name__ == "__main__":
    common.silence_warnings()
    result = run(common.load_series())
    result.save()
    print(f"naive: validation MAPE {result.validation_mape_pct:.2f} %, test MAPE {common.mape(common.test_slice(common.load_series()), result.point):.2f} %")
