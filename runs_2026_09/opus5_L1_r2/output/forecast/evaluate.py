"""Metrics and rolling-origin backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast.models import HORIZON, Forecaster


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Point-forecast accuracy for one aligned actual/prediction pair."""
    a = actual.to_numpy(dtype=float)
    p = predicted.reindex(actual.index).to_numpy(dtype=float)
    err = p - a
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAPE": float(np.mean(np.abs(err / a)) * 100.0),
        "sMAPE": float(np.mean(2 * np.abs(err) / (np.abs(a) + np.abs(p))) * 100.0),
        "BIAS": float(np.mean(err)),
        "PEAK_ERR": float(np.max(p) - np.max(a)),
    }


@dataclass(frozen=True)
class Window:
    """One backtest fold: train strictly before ``start``, score ``start``..168h."""

    start: pd.Timestamp

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, periods=HORIZON, freq="h", tz="UTC")


def january_windows(years: list[int]) -> list[Window]:
    """First-week-of-January folds, one per year, in UTC."""
    return [Window(pd.Timestamp(f"{y}-01-01 00:00", tz="UTC")) for y in years]


def backtest(
    series: pd.Series,
    build: dict[str, "type[Forecaster] | object"],
    windows: list[Window],
) -> pd.DataFrame:
    """Refit every model on each fold's history and score it on that fold."""
    rows: list[dict[str, object]] = []
    for window in windows:
        history = series.loc[: window.start - pd.Timedelta(hours=1)]
        actual = series.reindex(window.index)
        if actual.isna().any():
            raise ValueError(f"incomplete actuals for window {window.start}")
        for label, factory in build.items():
            model = factory() if callable(factory) else factory
            model.fit(history)
            prediction = model.predict(window.index)
            if not np.isfinite(prediction.to_numpy()).all():
                raise ValueError(f"{label} produced non-finite values at {window.start}")
            rows.append(
                {"window": window.start.date(), "model": label, **metrics(actual, prediction)}
            )
    return pd.DataFrame(rows)
