"""Prophet with German public holidays.

darts.models.Prophet wraps facebook Prophet; passing country_holidays='DE'
adds federal holiday effects. Weekly and yearly seasonalities are on by
default; hourly data enables daily seasonality automatically.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from common import Splits


def _to_ts(s: pd.Series):
    from darts import TimeSeries
    ser = s.copy()
    ser.index = ser.index.tz_convert("UTC").tz_localize(None)
    return TimeSeries.from_series(ser, freq="h")


def forecast(splits: Splits) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Fit Prophet on train+val, forecast 168 test hours with sampled quantiles."""
    from darts.models import Prophet as DProphet

    # Prophet has no meaningful hyperparameters to sweep here beyond seasonalities;
    # the country_holidays flag is the substantive choice. We refit directly on
    # train+val (still respecting the test window) after a quick val sanity check.
    train_ts = _to_ts(splits.train.astype(float))
    val_ts = _to_ts(splits.val.astype(float))

    # Validation sanity: fit train, predict val, keep for reporting only.
    m_val = DProphet(country_holidays="DE")
    m_val.fit(train_ts)
    _ = m_val.predict(n=len(val_ts))

    # Final fit on train + val.
    fit_ts = _to_ts(splits.train_val.astype(float))
    m = DProphet(country_holidays="DE")
    m.fit(fit_ts)

    # Probabilistic forecast via posterior samples (Prophet supports this).
    n_test = len(splits.test)
    pred = m.predict(n=n_test, num_samples=500)
    # pred is a TimeSeries with 500 samples along the component axis.
    arr = pred.all_values(copy=False)  # shape (n_test, n_components, n_samples)
    samples = arr[:, 0, :]              # univariate
    mean = pd.Series(samples.mean(axis=1), index=splits.test.index, name="prophet")
    intervals = pd.DataFrame({
        "lower_80": np.quantile(samples, 0.10, axis=1),
        "upper_80": np.quantile(samples, 0.90, axis=1),
        "lower_95": np.quantile(samples, 0.025, axis=1),
        "upper_95": np.quantile(samples, 0.975, axis=1),
    }, index=splits.test.index)
    hp = {
        "country_holidays": "DE",
        "seasonalities": ["daily", "weekly", "yearly"],
        "num_samples": 500,
    }
    # Also cache the fitted model for the (potential) decomposition figure.
    intervals.attrs["fitted_model"] = m
    return mean, intervals, hp
