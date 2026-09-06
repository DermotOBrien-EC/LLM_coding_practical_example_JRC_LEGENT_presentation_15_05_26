from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast_load import (
    FORECAST_START,
    HORIZON,
    ModelSpec,
    build_features,
    fit_predict,
    forecast_index,
    load_series,
    score,
    select_model,
)


def synthetic_series() -> pd.Series:
    index = pd.date_range("2019-08-01", "2020-01-10", freq="h", tz="UTC")
    values = 50000 + 7000 * np.sin(np.arange(len(index)) * 2 * np.pi / 24)
    return pd.Series(values, index=index, name="load_mw")


def test_requested_window_is_exactly_168_utc_hours() -> None:
    index = forecast_index(FORECAST_START)
    assert len(index) == HORIZON == 168
    assert index[0] == pd.Timestamp("2020-01-01T00:00:00Z")
    assert index[-1] == pd.Timestamp("2020-01-07T23:00:00Z")


@pytest.mark.parametrize("use_lags", [False, True])
def test_future_actuals_cannot_change_forecast(use_lags: bool) -> None:
    data = synthetic_series()
    spec = ModelSpec("test", use_lags=use_lags, max_leaf_nodes=7, max_iter=4)
    original = fit_predict(data, FORECAST_START, spec)
    altered = data.copy()
    altered.loc[altered.index >= FORECAST_START] = 9999999.0
    changed = fit_predict(altered, FORECAST_START, spec)
    truncated = fit_predict(data.loc[data.index < FORECAST_START], FORECAST_START, spec)
    np.testing.assert_array_equal(original, changed)
    np.testing.assert_array_equal(original, truncated)
    assert np.isfinite(original).all()
    assert (original > 0).all()


def test_weekly_lag_never_reaches_into_forecast_week() -> None:
    data = synthetic_series()
    history = data.loc[data.index < FORECAST_START]
    index = forecast_index(FORECAST_START)
    features = build_features(index, history, use_lags=True)
    expected = history.reindex(index - pd.Timedelta(hours=168)).to_numpy()
    np.testing.assert_array_equal(features["load_lag_168h"], expected)
    assert features["load_lag_168h"].iloc[-1] == history.iloc[-1]


def test_calendar_uses_german_local_time_and_new_year_holiday() -> None:
    index = pd.DatetimeIndex(["2019-12-31T23:00:00Z", "2020-01-01T23:00:00Z"])
    features = build_features(index, synthetic_series(), use_lags=False)
    assert features["hour"].tolist() == [0, 0]
    assert features["national_holiday"].tolist() == [1, 0]
    assert features["day_of_week"].tolist() == [2, 3]


def test_error_metrics_use_mw_and_percent() -> None:
    metrics = score(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert metrics["mae_mw"] == pytest.approx(15.0)
    assert metrics["rmse_mw"] == pytest.approx(np.sqrt(250.0))
    assert metrics["mape_percent"] == pytest.approx(10.0)
    assert metrics["bias_mw"] == pytest.approx(-5.0)
    assert metrics["wape_percent"] == pytest.approx(10.0)


def test_metrics_refuse_missing_or_mismatched_observations() -> None:
    with pytest.raises(ValueError):
        score(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        score(np.array([1.0, 2.0]), np.array([1.0]))


@pytest.mark.parametrize("defect", ["duplicate", "gap", "missing", "negative"])
def test_loader_rejects_invalid_data(tmp_path: Path, defect: str) -> None:
    frame = pd.DataFrame(
        {
            "utc_timestamp": pd.date_range("2019-01-01", periods=4, freq="h", tz="UTC"),
            "DE_load_actual_entsoe_transparency": [41000.0, 42000.0, 43000.0, 44000.0],
        }
    )
    if defect == "duplicate":
        frame.loc[1, "utc_timestamp"] = frame.loc[0, "utc_timestamp"]
    elif defect == "gap":
        frame = frame.drop(index=1)
    elif defect == "missing":
        frame.loc[1, "DE_load_actual_entsoe_transparency"] = np.nan
    else:
        frame.loc[1, "DE_load_actual_entsoe_transparency"] = -1.0
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_series(path)


def test_validation_refuses_target_week() -> None:
    data = synthetic_series()
    with pytest.raises(ValueError, match="before"):
        select_model(data, origins=[FORECAST_START])
