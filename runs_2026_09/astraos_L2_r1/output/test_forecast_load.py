from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from forecast_load import (
    FORECAST_START,
    HORIZON,
    ModelConfig,
    build_features,
    error_metrics,
    forecast_week,
    read_load,
    split_data,
    target_hours,
)


@pytest.fixture
def synthetic_load() -> pd.Series:
    index = pd.date_range("2018-01-01", periods=24 * 400, freq="h", tz="UTC")
    values = 50000 + 5000 * np.sin(np.arange(len(index)) * 2 * np.pi / 24)
    return pd.Series(values, index=index, name="load_mw")


def test_consumer_gets_168_finite_fixed_origin_predictions(
    synthetic_load: pd.Series,
) -> None:
    origin = synthetic_load.index[-1] + pd.Timedelta(hours=1)
    result = forecast_week(synthetic_load, origin, ModelConfig(15, max_iter=3))
    assert result.predictions.index.equals(target_hours(origin))
    assert len(result.predictions) == 168
    assert np.isfinite(result.predictions.to_numpy()).all()
    assert (result.predictions > 0).all()
    assert result.train_end < origin
    assert result.train_rows == len(synthetic_load) - 8760
    assert result.iterations == 3


def test_target_is_exactly_seven_complete_utc_days() -> None:
    index = target_hours(FORECAST_START)
    assert HORIZON == len(index) == 168
    assert index[0] == pd.Timestamp("2020-01-01T00:00:00Z")
    assert index[-1] == pd.Timestamp("2020-01-07T23:00:00Z")
    assert index[-1] + pd.Timedelta(hours=1) == pd.Timestamp("2020-01-08T00:00:00Z")


def test_real_data_split_excludes_holdout_and_later_observations() -> None:
    data = read_load(Path(__file__).with_name("opsd_de_load.csv"))
    history, actual = split_data(data)
    assert len(history) == 43824
    assert history.index.max() == pd.Timestamp("2019-12-31T23:00:00Z")
    assert actual.index.equals(target_hours(FORECAST_START))
    poisoned = data.copy()
    poisoned.loc[poisoned.index >= FORECAST_START] = 999999.0
    changed_history, changed_actual = split_data(poisoned)
    pd.testing.assert_series_equal(history, changed_history)
    assert (changed_actual == 999999).all()


def test_forecaster_refuses_training_on_target_week(synthetic_load: pd.Series) -> None:
    with pytest.raises(ValueError, match="before the forecast origin"):
        forecast_week(synthetic_load, synthetic_load.index[-10], ModelConfig(15, 3))


def test_last_hour_lag_168_is_last_available_history_hour(
    synthetic_load: pd.Series,
) -> None:
    origin = synthetic_load.index[-1] + pd.Timedelta(hours=1)
    index = target_hours(origin)
    features = build_features(synthetic_load, index)
    assert features.loc[index[-1], "lag_168"] == synthetic_load.iloc[-1]
    assert features.loc[index[0], "lag_168"] == synthetic_load.iloc[-168]
    assert features.loc[index[-1], "mean_168_lag_168"] == pytest.approx(
        synthetic_load.iloc[-168:].mean()
    )
    extended = pd.concat([synthetic_load, pd.Series(np.full(HORIZON, 999999.0), index=index)])
    pd.testing.assert_frame_equal(features, build_features(extended, index))


def test_feature_calendar_uses_berlin_time_and_partial_holidays(
    synthetic_load: pd.Series,
) -> None:
    index = pd.DatetimeIndex(
        ["2019-12-31T23:00:00Z", "2020-01-02T11:00:00Z", "2020-01-06T11:00:00Z"]
    )
    features = build_features(synthetic_load, index)
    assert features.iloc[0]["hour"] == 0
    assert features.iloc[0]["national_holiday"] == 1
    assert features.iloc[1]["holiday_previous_day"] == 1
    assert features.iloc[2]["epiphany"] == 1
    assert features.iloc[2]["national_holiday"] == 0


def test_feature_calendar_handles_dst_without_duplicate_utc_hours(
    synthetic_load: pd.Series,
) -> None:
    index = pd.date_range("2019-03-31T00:00:00Z", periods=3, freq="h")
    features = build_features(synthetic_load, index)
    assert features["hour"].tolist() == [1, 3, 4]
    assert features.index.equals(index)


def test_metrics_match_hand_calculation_and_signed_error() -> None:
    index = pd.date_range("2020-01-01", periods=2, freq="h", tz="UTC")
    actual = pd.Series([100.0, 200.0], index=index)
    forecast = pd.Series([110.0, 180.0], index=index)
    metrics = error_metrics(actual, forecast)
    assert metrics.mae_mw == pytest.approx(15.0)
    assert metrics.rmse_mw == pytest.approx(np.sqrt(250.0))
    assert metrics.mape_percent == pytest.approx(10.0)
    assert metrics.mean_error_mw == pytest.approx(-5.0)
    assert metrics.wape_percent == pytest.approx(10.0)


def test_metrics_refuse_misaligned_series() -> None:
    index = target_hours(FORECAST_START)
    actual = pd.Series(50000.0, index=index)
    with pytest.raises(ValueError, match="align"):
        error_metrics(actual, actual.iloc[::-1])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, 0.0, -1.0])
def test_input_rejects_invalid_loads(tmp_path: Path, bad_value: float) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "utc_timestamp": ["2019-01-01T00:00:00Z", "2019-01-01T01:00:00Z"],
            "DE_load_actual_entsoe_transparency": [50000.0, bad_value],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="finite and positive"):
        read_load(path)


@pytest.mark.parametrize("hours", [[0, 0], [0, 2]])
def test_input_rejects_duplicate_or_missing_hours(tmp_path: Path, hours: list[int]) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "utc_timestamp": [f"2019-01-01T{h:02d}:00:00Z" for h in hours],
            "DE_load_actual_entsoe_transparency": [50000.0, 51000.0],
        }
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="hourly|duplicate"):
        read_load(path)


def test_missing_actual_hour_is_not_imputed() -> None:
    data = read_load(Path(__file__).with_name("opsd_de_load.csv"))
    with pytest.raises(ValueError, match="168 actual"):
        split_data(data.drop(FORECAST_START))


def test_end_to_end_predictions_and_selection_ignore_all_future_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    import forecast_load as module

    source = Path(__file__).with_name("opsd_de_load.csv")
    monkeypatch.setattr(module, "CANDIDATES", (ModelConfig(15, 3), ModelConfig(31, 3)))
    module.run(source, tmp_path / "original")
    changed = pd.read_csv(source)
    times = pd.to_datetime(changed["utc_timestamp"], utc=True)
    changed.loc[times >= FORECAST_START, module.LOAD_COLUMN] = 999999.0
    poisoned_path = tmp_path / "poisoned.csv"
    changed.to_csv(poisoned_path, index=False)
    module.run(poisoned_path, tmp_path / "poisoned")
    original = pd.read_csv(tmp_path / "original" / "forecast.csv")
    poisoned = pd.read_csv(tmp_path / "poisoned" / "forecast.csv")
    assert len(original) == 168
    pd.testing.assert_series_equal(original["forecast_mw"], poisoned["forecast_mw"])
    original_validation = pd.read_csv(tmp_path / "original" / "validation.csv")
    poisoned_validation = pd.read_csv(tmp_path / "poisoned" / "validation.csv")
    pd.testing.assert_frame_equal(original_validation, poisoned_validation)
    report = json.loads((tmp_path / "original" / "metrics.json").read_text())
    recomputed_mae = (original["forecast_mw"] - original["actual_mw"]).abs().mean()
    assert report["metrics"]["forecast"]["mae_mw"] == pytest.approx(recomputed_mae)
    assert (tmp_path / "original" / "forecast_vs_actual.png").stat().st_size > 10000
