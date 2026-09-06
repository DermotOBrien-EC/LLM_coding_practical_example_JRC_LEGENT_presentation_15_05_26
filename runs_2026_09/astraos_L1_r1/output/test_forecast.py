from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast_model import build_features, forecast_candidates, load_history, target_hours


def test_target_is_exactly_first_168_utc_hours_of_2020() -> None:
    index = target_hours("2020-01-01")
    assert len(index) == 168
    assert str(index[0]) == "2020-01-01 00:00:00+00:00"
    assert str(index[-1]) == "2020-01-07 23:00:00+00:00"
    assert (np.diff(index.asi8) == 3_600_000_000_000).all()


def test_loader_discards_future_values_before_validation(tmp_path: Path) -> None:
    path = tmp_path / "load.csv"
    path.write_text(
        "utc_timestamp,DE_load_actual_entsoe_transparency\n"
        "2019-12-31 22:00:00+00:00,50000\n"
        "2019-12-31 23:00:00+00:00,51000\n"
        "2020-01-01 00:00:00+00:00,not-an-observation\n"
    )
    history = load_history(path, "2020-01-01")
    assert history.tolist() == [50000.0, 51000.0]
    assert history.index.max() < target_hours("2020-01-01")[0]


def test_loader_rejects_missing_training_hour(tmp_path: Path) -> None:
    path = tmp_path / "load.csv"
    path.write_text(
        "utc_timestamp,DE_load_actual_entsoe_transparency\n"
        "2019-12-31 21:00:00+00:00,50000\n"
        "2019-12-31 23:00:00+00:00,51000\n"
    )
    with pytest.raises(ValueError, match="hourly"):
        load_history(path, "2020-01-01")


def test_features_use_german_local_holidays() -> None:
    index = pd.to_datetime(
        ["2019-12-31 23:00Z", "2020-01-01 23:00Z", "2020-01-06 12:00Z"], utc=True
    )
    features = build_features(index)
    assert features["hour"].tolist() == [0, 0, 13]
    assert features["new_year"].tolist() == [1, 0, 0]
    assert features["national_holiday"].tolist() == [1, 0, 0]
    assert features["epiphany"].tolist() == [0, 0, 1]


def test_forecaster_rejects_target_observations_in_history() -> None:
    history = pd.Series(
        50000.0, index=pd.date_range("2019-01-01", "2020-01-01", freq="h", tz="UTC")
    )
    with pytest.raises(ValueError, match="before"):
        forecast_candidates(history, target_hours("2020-01-01"))
