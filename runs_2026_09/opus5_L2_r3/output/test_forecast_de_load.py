"""Tests for the load forecaster, aimed at the leakage boundary."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecast_de_load import (
    HORIZON,
    HolidayCalendar,
    calendar_features,
    design_matrix,
    lag_features,
    metrics,
    seasonal_naive,
    window_index,
)


@pytest.fixture(scope="module")
def cal() -> HolidayCalendar:
    return HolidayCalendar()


@pytest.fixture(scope="module")
def series() -> pd.Series:
    idx = pd.date_range("2017-01-01", periods=30_000, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    hours = np.arange(len(idx))
    values = (
        50_000.0
        + 8_000.0 * np.sin(2 * np.pi * hours / 24.0)
        + 3_000.0 * np.sin(2 * np.pi * hours / 168.0)
        + rng.normal(0.0, 500.0, len(idx))
    )
    return pd.Series(values, index=idx, name="load_mw")


def test_short_lags_are_rejected(series: pd.Series, cal: HolidayCalendar) -> None:
    target = series.index[-HORIZON:]
    history = series.loc[series.index < target[0]]
    with pytest.raises(ValueError, match="would leak"):
        lag_features(history, target, cal, lags=(24, 168))


def test_features_are_immune_to_future_values(series: pd.Series, cal: HolidayCalendar) -> None:
    """Poisoning every actual inside the window must not move one feature value.

    Because every lag is >= HORIZON, no feature can reach a timestamp at or
    after the forecast origin, even if a careless caller hands the function a
    series that contains the window.
    """
    target = series.index[-HORIZON:]
    history = series.loc[series.index < target[0]]
    poisoned = series.copy()
    poisoned.loc[target] = 1e9

    clean = design_matrix(history, target, cal)
    with_future = design_matrix(poisoned, target, cal)
    pd.testing.assert_frame_equal(clean, with_future)


def test_the_leakage_check_can_actually_fail(series: pd.Series, cal: HolidayCalendar) -> None:
    """Red-state proof: a sub-horizon lag *would* pick the poison up.

    Without this, the test above could pass by asserting nothing.
    """
    target = series.index[-HORIZON:]
    poisoned = series.copy()
    poisoned.loc[target] = 1e9

    def shortcut(source: pd.Series) -> np.ndarray:
        return source.shift(24).reindex(target).to_numpy()

    clean_24h = shortcut(series)
    leaked_24h = shortcut(poisoned)
    assert not np.allclose(clean_24h[24:], leaked_24h[24:]), (
        "a 24h lag must expose in-window values, otherwise the immunity test above is vacuous"
    )


def test_lag_features_reach_only_into_history(series: pd.Series, cal: HolidayCalendar) -> None:
    target = series.index[-HORIZON:]
    history = series.loc[series.index < target[0]]
    frame = lag_features(history, target, cal, lags=(168, 336))

    expected_168 = series.reindex(target - pd.Timedelta(hours=168)).to_numpy()
    np.testing.assert_allclose(frame["lag_168h"].to_numpy(), expected_168)
    # The final target hour's 168h lag is the last hour of history, not later.
    assert (target - pd.Timedelta(hours=168)).max() == history.index[-1]


def test_calendar_features_need_no_load_at_all(cal: HolidayCalendar) -> None:
    idx = pd.date_range("2031-05-01", periods=48, freq="h", tz="UTC")
    frame = calendar_features(idx, cal)
    assert frame.notna().all().all()
    assert len(frame) == 48


def test_new_year_is_flagged_as_a_national_holiday(cal: HolidayCalendar) -> None:
    idx = window_index(pd.Timestamp("2020-01-01"))
    frame = calendar_features(idx, cal)
    local_day = idx.tz_convert("Europe/Berlin").day

    assert frame.loc[local_day == 1, "is_national_holiday"].eq(1.0).all()
    assert frame.loc[local_day == 1, "holiday_pop_share"].eq(1.0).all()
    # Epiphany, 6 January: a holiday in BY, BW and ST only.
    share_jan6 = frame.loc[local_day == 6, "holiday_pop_share"].unique()
    assert len(share_jan6) == 1
    assert 0.2 < share_jan6[0] < 0.35
    assert frame.loc[local_day == 6, "is_national_holiday"].eq(0.0).all()


def test_window_index_is_168_local_hours(cal: HolidayCalendar) -> None:
    del cal
    idx = window_index(pd.Timestamp("2020-01-01"))
    local = idx.tz_convert("Europe/Berlin")
    assert len(idx) == HORIZON
    assert str(local[0]) == "2020-01-01 00:00:00+01:00"
    assert str(local[-1]) == "2020-01-07 23:00:00+01:00"


def test_seasonal_naive_is_last_weeks_value(series: pd.Series, cal: HolidayCalendar) -> None:
    target = series.index[-HORIZON:]
    history = series.loc[series.index < target[0]]
    pred = seasonal_naive(history, target, cal)
    np.testing.assert_allclose(pred.to_numpy(), history.to_numpy()[-HORIZON:])


def test_metrics_on_a_hand_case() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="h", tz="UTC")
    actual = pd.Series([100.0, 100.0, 200.0, 200.0], index=idx)
    pred = pd.Series([110.0, 90.0, 220.0, 200.0], index=idx)
    got = metrics(actual, pred)

    assert got["MAE_MW"] == pytest.approx((10 + 10 + 20 + 0) / 4)
    assert got["MAPE_pct"] == pytest.approx((10 + 10 + 10 + 0) / 4)
    assert got["RMSE_MW"] == pytest.approx(np.sqrt((100 + 100 + 400 + 0) / 4))
    assert got["bias_MW"] == pytest.approx((10 - 10 + 20 + 0) / 4)
    assert got["max_abs_err_MW"] == pytest.approx(20.0)
    assert got["peak_err_pct"] == pytest.approx((220 - 200) / 200 * 100)
