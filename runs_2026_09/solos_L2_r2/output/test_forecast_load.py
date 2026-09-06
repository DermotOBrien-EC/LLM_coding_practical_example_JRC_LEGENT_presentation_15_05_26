from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from forecast_load import (
    TARGET_START,
    build_calendar_features,
    build_forecast_table,
    calculate_metrics,
    fit_and_forecast,
    has_complete_forecast_data,
    load_hourly_load,
    make_forecast_index,
    make_target_index,
    split_training_and_target,
)


def test_target_index_matches_the_literal_requested_window() -> None:
    expected = pd.date_range("2020-01-01", periods=168, freq="h", tz="UTC")

    assert make_target_index().equals(expected)


def test_split_excludes_target_values_from_training() -> None:
    index = pd.date_range("2019-12-30", "2020-01-08", freq="h", tz="UTC", inclusive="left")
    load = pd.Series(np.arange(len(index), dtype=float), index=index, name="load_mw")

    training, actual = split_training_and_target(load)

    assert training.index.max() < TARGET_START
    assert actual.index.equals(pd.date_range("2020-01-01", periods=168, freq="h", tz="UTC"))
    assert len(actual) == 168


def test_calendar_features_use_german_local_calendar() -> None:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2020-01-02 00:00:00", tz="UTC"),
        ]
    )

    features = build_calendar_features(index)

    assert features.loc[index[0], "local_hour"] == 1
    assert features.loc[index[0], "is_holiday"] == 1
    assert features.loc[index[0], "is_new_year"] == 1
    assert features.loc[index[1], "is_holiday"] == 0


def test_metrics_match_known_values() -> None:
    index = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
    actual = pd.Series([100.0, 200.0], index=index)
    forecast = pd.Series([90.0, 220.0], index=index)

    metrics = calculate_metrics(actual, forecast)

    assert metrics.mae_mw == pytest.approx(15.0)
    assert metrics.rmse_mw == pytest.approx(np.sqrt(250.0))
    assert metrics.mape_percent == pytest.approx(10.0)
    assert metrics.r_squared == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("forecast_values", "expected_r_squared"),
    [([10.0, 10.0], 1.0), ([9.0, 11.0], 0.0)],
)
def test_metrics_keep_r_squared_finite_for_constant_actuals(
    forecast_values: list[float], expected_r_squared: float
) -> None:
    index = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
    actual = pd.Series([10.0, 10.0], index=index)
    forecast = pd.Series(forecast_values, index=index)

    metrics = calculate_metrics(actual, forecast)

    assert metrics.r_squared == expected_r_squared


def test_timestamp_mismatches_are_rejected() -> None:
    actual_index = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
    forecast_index = actual_index + pd.Timedelta(hours=1)
    actual = pd.Series([10.0, 20.0], index=actual_index)
    forecast = pd.Series([10.0, 20.0], index=forecast_index)

    with pytest.raises(ValueError, match="identical timestamp indexes"):
        calculate_metrics(actual, forecast)
    with pytest.raises(ValueError, match="identical timestamp indexes"):
        build_forecast_table(actual, forecast)


def test_fit_rejects_a_reversed_forecast_index_before_training() -> None:
    reversed_index = make_target_index()[::-1]
    empty_load = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))

    with pytest.raises(ValueError, match="sorted"):
        fit_and_forecast(empty_load, reversed_index)


def test_backtest_eligibility_requires_full_training_history() -> None:
    index = pd.date_range("2016-07-01", "2020-01-08", freq="h", tz="UTC", inclusive="left")
    load = pd.Series(np.ones(len(index)), index=index)

    assert not has_complete_forecast_data(
        load, make_forecast_index(pd.Timestamp("2017-01-01", tz="UTC"))
    )
    assert has_complete_forecast_data(
        load, make_forecast_index(pd.Timestamp("2019-01-01", tz="UTC"))
    )


def test_loader_normalizes_offsets_and_permits_gaps_outside_required_windows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "load.csv"
    path.write_text(
        "utc_timestamp,DE_load_actual_entsoe_transparency\n"
        "2015-01-01T00:00:00+00:00,9\n"
        "2020-01-01T00:00:00+00:00,10\n"
        "2020-01-01T02:00:00+01:00,11\n"
        "2020-09-01T00:00:00+00:00,12\n"
        "2020-09-03T00:00:00+00:00,13\n",
        encoding="utf-8",
    )

    load = load_hourly_load(path)

    expected_index = pd.DatetimeIndex(
        [
            pd.Timestamp("2015-01-01T00:00:00", tz="UTC"),
            pd.Timestamp("2020-01-01T00:00:00", tz="UTC"),
            pd.Timestamp("2020-01-01T01:00:00", tz="UTC"),
        ]
    )
    assert load.index.equals(expected_index)
    assert load.tolist() == [9.0, 10.0, 11.0]
