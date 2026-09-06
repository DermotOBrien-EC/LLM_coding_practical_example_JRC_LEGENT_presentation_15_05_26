"""Structural check: forecasts must not read data at or after the origin."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecast.data import load_series
from forecast.models import (
    HORIZON,
    HolidayAdjustedNaive,
    LightGBMModel,
    LinearProfileModel,
    SeasonalNaive,
    YearAgoNaive,
)

CSV = Path(__file__).resolve().parents[1] / "opsd_de_load.csv"
ORIGIN = pd.Timestamp("2020-01-01 00:00", tz="UTC")


@pytest.fixture(scope="module")
def series() -> pd.Series:
    return load_series(CSV)


def _factories() -> dict[str, object]:
    return {
        "seasonal_naive": SeasonalNaive,
        "year_ago": YearAgoNaive,
        "holiday_adjusted": HolidayAdjustedNaive,
        "linear": LinearProfileModel,
        "lightgbm": lambda: LightGBMModel(n_estimators=120),
    }


@pytest.mark.parametrize("label", list(_factories()))
def test_forecast_invariant_to_future_values(series: pd.Series, label: str) -> None:
    """Corrupting every observation from the origin on must not move the forecast.

    This is the real guarantee we need. A lag or rolling window that reached
    past the origin would change the prediction when the future is scrambled.
    """
    factory = _factories()[label]
    index = pd.date_range(ORIGIN, periods=HORIZON, freq="h", tz="UTC")
    history = series.loc[: ORIGIN - pd.Timedelta(hours=1)]

    clean = factory().fit(history).predict(index)

    corrupted = series.copy()
    future = corrupted.index >= ORIGIN
    assert future.sum() > 0
    rng = np.random.default_rng(0)
    corrupted[future] = rng.uniform(1e3, 1e5, size=int(future.sum()))
    # Fit on the same history, but hand the model the corrupted full series so
    # any lag reaching forward would be visible.
    model = factory()
    model.fit(history)
    if hasattr(model, "_history"):
        model._history = pd.concat([history, corrupted[future]])
    dirty = model.predict(index)

    # Guard against a vacuous pass: assert_allclose treats NaN as equal to NaN,
    # so an all-NaN model would satisfy the comparison without proving anything.
    assert np.isfinite(clean.to_numpy()).all(), f"{label} produced non-finite forecasts"
    assert clean.std() > 0, f"{label} produced a constant forecast"
    np.testing.assert_allclose(clean.to_numpy(), dirty.to_numpy(), rtol=1e-12)


def test_lag_features_never_reach_origin(series: pd.Series) -> None:
    """The shortest lag used equals the horizon, so its newest input is t-168h."""
    from forecast.models import _lag_features

    index = pd.date_range(ORIGIN, periods=HORIZON, freq="h", tz="UTC")
    feats = _lag_features(series, index)
    # lag_168 at the final target hour must be exactly the last training hour.
    assert (
        feats["lag_168"].iloc[-1] == series.loc[ORIGIN + pd.Timedelta(hours=HORIZON - 1 - HORIZON)]
    )
    with pytest.raises(ValueError):
        _lag_features(series, index, min_lag=167)


def test_source_data_is_complete(series: pd.Series) -> None:
    assert len(series) == 50400
    assert not series.isna().any()
