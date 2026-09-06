from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast import (
    CUTOFF,
    build_tree_features,
    calendar_frame,
    forecast_index,
    load_history,
    make_forecast_table,
)


def test_consumer_receives_168_ordered_hourly_predictions() -> None:
    index = forecast_index(CUTOFF)
    table = make_forecast_table(index, np.full(168, 50000.0), 3000.0, 5000.0)
    assert len(table) == 168
    assert table.utc_timestamp.iloc[0] == pd.Timestamp("2020-01-01T00:00:00Z")
    assert table.utc_timestamp.iloc[-1] == pd.Timestamp("2020-01-07T23:00:00Z")
    assert table.utc_timestamp.is_unique
    assert (table.lower_95_mw <= table.lower_80_mw).all()
    assert (table.lower_80_mw <= table.forecast_mw).all()
    assert (table.forecast_mw <= table.upper_80_mw).all()
    assert (table.upper_80_mw <= table.upper_95_mw).all()
    assert table.local_timestamp.iloc[0].hour == 1


def test_loader_discards_future_targets_before_numeric_parsing(tmp_path: Path) -> None:
    source = tmp_path / "load.csv"
    source.write_text(
        "utc_timestamp,DE_load_actual_entsoe_transparency\n"
        "2019-12-31 22:00:00+00:00,45000\n"
        "2019-12-31 23:00:00+00:00,44000\n"
        "2020-01-01 00:00:00+00:00,DO_NOT_USE\n"
    )
    history = load_history(source, CUTOFF)
    assert len(history) == 2
    assert history.iloc[-1] == 44000
    assert history.index.max() < CUTOFF


def test_loader_rejects_missing_hours(tmp_path: Path) -> None:
    source = tmp_path / "load.csv"
    source.write_text(
        "utc_timestamp,DE_load_actual_entsoe_transparency\n"
        "2019-12-31 21:00:00+00:00,45000\n"
        "2019-12-31 23:00:00+00:00,44000\n"
    )
    with pytest.raises(ValueError, match="hourly"):
        load_history(source, CUTOFF)


def test_features_use_only_observations_available_at_origin() -> None:
    index = pd.date_range("2018-01-01", "2020-01-10", freq="h", tz="UTC")
    values = pd.Series(np.arange(len(index), dtype=float), index=index)
    target = forecast_index(CUTOFF)
    before = build_tree_features(target, values.loc[values.index < CUTOFF], CUTOFF)
    values.loc[values.index >= CUTOFF] = 1e12
    after = build_tree_features(target, values, CUTOFF)
    pd.testing.assert_frame_equal(before, after)
    assert before.lag_168.iloc[-1] == values.loc[CUTOFF - pd.Timedelta(hours=1)]
    assert before.lag_168.iloc[0] == values.loc[CUTOFF - pd.Timedelta(hours=168)]


def test_new_year_christmas_and_regional_holiday_features() -> None:
    index = pd.DatetimeIndex(
        ["2019-12-25T12:00Z", "2020-01-01T12:00Z", "2020-01-06T12:00Z"]
    )
    frame = calendar_frame(index)
    assert frame.holiday.tolist() == ["christmas", "new_year", "none"]
    assert frame.regional_epiphany.tolist() == [0, 0, 1]
    assert frame.effective_dow.tolist() == [6, 6, 0]


def test_calendar_uses_german_civil_time_through_dst() -> None:
    frame = calendar_frame(pd.DatetimeIndex(["2019-03-31T00:00Z", "2019-03-31T01:00Z"]))
    assert frame.hour.tolist() == [1, 3]


def test_prediction_table_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        make_forecast_table(forecast_index(CUTOFF), np.full(168, np.nan), 3000, 5000)
    with pytest.raises(ValueError):
        make_forecast_table(forecast_index(CUTOFF), np.full(167, 50000), 3000, 5000)
