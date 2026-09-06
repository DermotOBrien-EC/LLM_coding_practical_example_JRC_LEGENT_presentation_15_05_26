"""Prophet with German public holidays via darts.

Daily, weekly, and yearly seasonality; the German-holiday effect is what
this model is meant to buy on the Jan 1 observation.

Prophet does not have hyperparameters we need to sweep here beyond the
seasonality choice, but we still validate one control (whether to enable
`daily_seasonality`) on the Oct-Dec 2019 window so a single deliberate
choice is on record.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.models import Prophet as DartsProphet

from common import Splits, align, mape


warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("prophet").setLevel(logging.CRITICAL)
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)


def _to_ts(series: pd.Series) -> TimeSeries:
    """Wrap a pandas hourly series as a darts TimeSeries with UTC-naive index.

    Prophet expects timezone-naive timestamps; darts inherits that.
    """
    s = series.copy()
    s.index = s.index.tz_convert("UTC").tz_localize(None)
    return TimeSeries.from_series(s, freq="h")


def _fit_and_forecast(
    fit_on: pd.Series,
    horizon: int,
    daily_seasonality: bool,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit Prophet on `fit_on`, return (point forecast, sample matrix)."""
    model = DartsProphet(
        country_holidays="DE",
        daily_seasonality=daily_seasonality,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    model.fit(_to_ts(fit_on))
    samples_ts = model.predict(n=horizon, num_samples=num_samples)
    all_vals = samples_ts.all_values(copy=False)  # (horizon, 1, num_samples)
    samples = all_vals[:, 0, :]
    point = samples.mean(axis=1)
    return point, samples


def run(splits: Splits) -> dict[str, object]:
    """Pick the daily-seasonality flag on the validation window, then refit."""
    val_scores: dict[bool, float] = {}
    val_horizon = min(168, len(splits.val))
    for daily in (True, False):
        try:
            point, _ = _fit_and_forecast(
                splits.train, val_horizon, daily_seasonality=daily, num_samples=1
            )
            val_scores[daily] = mape(splits.val.iloc[:val_horizon].values, point)
        except Exception as exc:  # noqa: BLE001
            val_scores[daily] = float("inf")
            print(f"[prophet] daily={daily} failed: {exc}")

    best_daily = min(val_scores, key=val_scores.get)

    point, samples = _fit_and_forecast(
        splits.trainval,
        horizon=len(splits.test),
        daily_seasonality=best_daily,
        num_samples=500,
    )

    forecast = align(point, splits.test.index)
    lower_80 = align(np.quantile(samples, 0.10, axis=1), splits.test.index)
    upper_80 = align(np.quantile(samples, 0.90, axis=1), splits.test.index)
    lower_95 = align(np.quantile(samples, 0.025, axis=1), splits.test.index)
    upper_95 = align(np.quantile(samples, 0.975, axis=1), splits.test.index)
    q10 = align(np.quantile(samples, 0.10, axis=1), splits.test.index)
    q50 = align(np.quantile(samples, 0.50, axis=1), splits.test.index)
    q90 = align(np.quantile(samples, 0.90, axis=1), splits.test.index)

    return {
        "name": "prophet",
        "forecast": forecast,
        "lower_80": lower_80,
        "upper_80": upper_80,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "quantiles": {"0.1": q10, "0.5": q50, "0.9": q90},
        "samples": samples,
        "fitted_model": None,  # kept out to avoid pickling issues
        "hyperparameters": {
            "country_holidays": "DE",
            "daily_seasonality": best_daily,
            "weekly_seasonality": True,
            "yearly_seasonality": True,
            "num_samples": 500,
            "validation_mape_pct": val_scores[best_daily],
        },
    }
