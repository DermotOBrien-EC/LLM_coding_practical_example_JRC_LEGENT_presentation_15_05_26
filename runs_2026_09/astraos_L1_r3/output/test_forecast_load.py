from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast_load import (
    CUTOFF,
    HORIZON,
    LAGS,
    LOAD_COLUMN,
    build_features,
    calendar_features,
    forecast_candidates,
    load_history,
    make_forecast_frame,
    select_model,
)


def synthetic_history() -> pd.Series:
    index = pd.date_range(pd.Timestamp("2015-01-01", tz="UTC"), CUTOFF, freq="h", inclusive="left")
    return pd.Series(50000.0 + np.arange(len(index)), index=index, name=LOAD_COLUMN)


def test_forecast_consumer_contract() -> None:
    frame = make_forecast_frame(np.full(HORIZON, 50000.0))
    times = pd.DatetimeIndex(pd.to_datetime(frame["utc_timestamp"], utc=True))
    assert len(frame) == 168
    assert list(frame.columns) == ["utc_timestamp", "forecast_load_mw"]
    assert times.equals(pd.date_range("2020-01-01", periods=168, freq="h", tz="UTC"))
    assert np.isfinite(frame["forecast_load_mw"]).all()


@pytest.mark.parametrize("values", [np.zeros(168), np.full(168, np.nan), np.ones(167)])
def test_forecast_rejects_invalid_output(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        make_forecast_frame(values)


def test_reader_discards_future_before_value_validation(tmp_path: Path) -> None:
    path = tmp_path / "load.csv"
    path.write_text(
        f"utc_timestamp,{LOAD_COLUMN}\n"
        "2019-12-31 22:00:00+00:00,42000\n"
        "2019-12-31 23:00:00+00:00,41000\n"
        "2020-01-01 00:00:00+00:00,THIS_MUST_NOT_BE_USED\n"
    )
    history = load_history(path)
    assert len(history) == 2
    assert history.index.max() == CUTOFF - pd.Timedelta(hours=1)
    assert history.tolist() == [42000.0, 41000.0]


@pytest.mark.parametrize(
    "timestamps",
    [
        ["2019-12-31 21:00", "2019-12-31 23:00"],
        ["2019-12-31 23:00", "2019-12-31 23:00"],
    ],
)
def test_reader_rejects_gaps_and_duplicates(tmp_path: Path, timestamps: list[str]) -> None:
    path = tmp_path / "load.csv"
    pd.DataFrame({"utc_timestamp": timestamps, LOAD_COLUMN: [40000, 42000]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError):
        load_history(path)


def test_local_holidays_and_new_year_wrap() -> None:
    index = pd.DatetimeIndex(
        [
            "2019-12-28 12:00Z",
            "2019-12-31 23:00Z",
            "2020-01-03 12:00Z",
            "2020-01-05 23:00Z",
            "2020-01-01 23:00Z",
        ]
    )
    features = calendar_features(index)
    assert features["new_year_distance"].iloc[:3].tolist() == [-4, 0, 2]
    assert features["christmas_distance"].iloc[:3].tolist() == [3, 7, 9]
    assert features["holiday_kind"].iloc[1] == 2
    assert features["hour"].iloc[1] == 0
    assert features["national_holiday"].iloc[1] == 1
    assert features["national_holiday"].iloc[4] == 0
    assert features["epiphany"].iloc[3] == 1


def test_week_ahead_lags_are_observed_and_correct() -> None:
    history = synthetic_history()
    target = pd.date_range(CUTOFF, periods=HORIZON, freq="h")
    features = build_features(target, history, with_lags=True)
    for lag in LAGS:
        source = target - pd.Timedelta(hours=lag)
        assert source.max() < CUTOFF
        np.testing.assert_array_equal(features[f"load_lag_{lag}"], history.loc[source])
    assert features["load_lag_168"].iloc[-1] == history.iloc[-1]


def test_candidate_predictions_ignore_appended_future_observations() -> None:
    history = synthetic_history()
    target = pd.date_range(CUTOFF, periods=HORIZON, freq="h")
    contaminated = pd.concat([history, pd.Series(1e12, index=target)])
    first = forecast_candidates(history, CUTOFF, n_estimators=4)
    second = forecast_candidates(contaminated, CUTOFF, n_estimators=4)
    assert set(first) == set(second)
    for name in first:
        assert len(first[name]) == HORIZON
        np.testing.assert_array_equal(first[name], second[name])


def test_selection_excludes_robustness_folds() -> None:
    metrics = pd.DataFrame(
        [
            {"model": "a", "fold_type": "new_year", "mae_mw": 1.0},
            {"model": "b", "fold_type": "new_year", "mae_mw": 2.0},
            {"model": "a", "fold_type": "robustness", "mae_mw": 1000.0},
            {"model": "b", "fold_type": "robustness", "mae_mw": 0.0},
        ]
    )
    assert select_model(metrics) == "a"
