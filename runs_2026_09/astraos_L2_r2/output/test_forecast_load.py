from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from forecast_load import (
    HORIZON_HOURS,
    LAGS,
    ModelSpec,
    calendar_features,
    fit_forecast,
    forecast_index,
    lag_features,
    load_data,
    score,
    select_and_forecast,
)


@pytest.fixture
def synthetic_data() -> pd.Series:
    index = pd.date_range("2016-01-01", "2020-01-10 23:00", freq="h", tz="UTC")
    values = 50000 + 5000 * np.sin(2 * np.pi * index.hour / 24) + index.dayofweek * 300
    return pd.Series(np.asarray(values, dtype=float), index=index, name="load_mw")


def test_exact_requested_window() -> None:
    index = forecast_index()
    assert len(index) == HORIZON_HOURS == 168
    assert str(index[0]) == "2020-01-01 00:00:00+00:00"
    assert str(index[-1]) == "2020-01-07 23:00:00+00:00"
    assert (index[1:] - index[:-1] == pd.Timedelta(hours=1)).all()


def test_metric_definitions() -> None:
    metrics = score(np.array([100.0, 200.0]), np.array([110.0, 180.0]))
    assert metrics.mae_mw == pytest.approx(15)
    assert metrics.rmse_mw == pytest.approx(np.sqrt(250))
    assert metrics.mape_pct == pytest.approx(10)
    assert metrics.wape_pct == pytest.approx(10)
    assert metrics.bias_mw == pytest.approx(-5)


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [([0.0], [1.0]), ([np.nan], [1.0]), ([1.0], [np.inf]), ([1.0, 2.0], [1.0])],
)
def test_metrics_reject_invalid_inputs(actual: list[float], predicted: list[float]) -> None:
    with pytest.raises(ValueError):
        score(np.array(actual), np.array(predicted))


def test_local_calendar_uses_german_time() -> None:
    index = pd.DatetimeIndex(["2019-12-31 23:00:00+00:00", "2020-01-06 12:00:00+00:00"])
    features = calendar_features(index)
    assert features.iloc[0]["hour"] == 0
    assert features.iloc[0]["national_holiday"] == 1
    assert features.iloc[1]["epiphany"] == 1
    assert features.iloc[1]["national_holiday"] == 0


def test_all_forecast_lags_are_available_before_origin(synthetic_data: pd.Series) -> None:
    future = forecast_index()
    history = synthetic_data.loc[synthetic_data.index < future[0]]
    features = lag_features(history, future)
    assert min(LAGS) >= len(future)
    assert features.notna().all().all()
    assert features.iloc[-1]["lag_168h"] == history.iloc[-1]
    for lag in LAGS:
        expected = history.reindex(future - pd.Timedelta(hours=lag)).to_numpy()
        np.testing.assert_array_equal(features[f"lag_{lag}h"], expected)


def test_forecast_rejects_future_values_and_origin_gap(synthetic_data: pd.Series) -> None:
    future = forecast_index()
    spec = ModelSpec("calendar", use_lags=False, max_iter=2)
    with pytest.raises(ValueError, match="before"):
        fit_forecast(synthetic_data, future, spec)
    history = synthetic_data.loc[synthetic_data.index < future[0]]
    with pytest.raises(ValueError, match="immediately"):
        fit_forecast(history.iloc[:-1], future, spec)


def test_holdout_values_cannot_change_selection_or_predictions(synthetic_data: pd.Series) -> None:
    specs = (
        ModelSpec("calendar", use_lags=False, max_iter=2),
        ModelSpec("calendar_lags", use_lags=True, max_iter=2),
    )
    folds = ("2019-01-01",)
    original = select_and_forecast(synthetic_data, specs=specs, validation_starts=folds)
    changed = synthetic_data.copy()
    changed.loc[changed.index >= forecast_index()[0]] *= 10
    altered = select_and_forecast(changed, specs=specs, validation_starts=folds)
    pd.testing.assert_series_equal(original.forecast, altered.forecast)
    pd.testing.assert_frame_equal(original.validation, altered.validation)
    assert original.selected_model == altered.selected_model
    assert original.training_end < forecast_index()[0]
    assert original.forecast.notna().all()
    assert len(original.forecast) == 168
    assert original.validation.groupby("origin")["training_rows"].nunique().max() == 1


def test_validation_cannot_touch_test_week(synthetic_data: pd.Series) -> None:
    with pytest.raises(ValueError, match="validation"):
        select_and_forecast(synthetic_data, validation_starts=("2019-12-31",))


@pytest.mark.parametrize("problem", ["duplicate", "gap", "missing", "negative", "infinite"])
def test_loader_rejects_invalid_hourly_data(tmp_path: Path, problem: str) -> None:
    frame = pd.DataFrame(
        {
            "utc_timestamp": pd.date_range("2019-01-01", periods=4, freq="h", tz="UTC"),
            "DE_load_actual_entsoe_transparency": [40000.0, 41000.0, 42000.0, 43000.0],
        }
    )
    if problem == "duplicate":
        frame.loc[1, "utc_timestamp"] = frame.loc[0, "utc_timestamp"]
    elif problem == "gap":
        frame = frame.drop(index=1)
    elif problem == "missing":
        frame.loc[1, "DE_load_actual_entsoe_transparency"] = np.nan
    elif problem == "negative":
        frame.loc[1, "DE_load_actual_entsoe_transparency"] = -1
    else:
        frame.loc[1, "DE_load_actual_entsoe_transparency"] = np.inf
    path = tmp_path / "load.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_data(path)


def test_cli_writes_complete_consistent_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parent
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "forecast_load.py"),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    assert "168 hours" in completed.stdout
    with (tmp_path / "forecast.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    with (root / "opsd_de_load.csv").open() as stream:
        source = {
            datetime.fromisoformat(row["utc_timestamp"]): float(
                row["DE_load_actual_entsoe_transparency"]
            )
            for row in csv.DictReader(stream)
        }
    assert len(rows) == 168
    origin = datetime(2020, 1, 1, tzinfo=timezone.utc)
    errors: list[float] = []
    actuals: list[float] = []
    for hour, row in enumerate(rows):
        timestamp = datetime.fromisoformat(row["utc_timestamp"])
        assert timestamp == origin + timedelta(hours=hour)
        actual = float(row["actual_mw"])
        predicted = float(row["forecast_mw"])
        assert actual == source[timestamp]
        assert float(row["baseline_previous_week_mw"]) == source[timestamp - timedelta(hours=168)]
        assert float(row["error_mw"]) == pytest.approx(predicted - actual)
        assert math.isfinite(predicted) and predicted > 0
        actuals.append(actual)
        errors.append(predicted - actual)
    report = json.loads((tmp_path / "metrics.json").read_text())
    independently_computed = {
        "mae_mw": sum(abs(error) for error in errors) / 168,
        "rmse_mw": math.sqrt(sum(error**2 for error in errors) / 168),
        "mape_pct": 100 * sum(abs(error) / actual for error, actual in zip(errors, actuals)) / 168,
        "wape_pct": 100 * sum(abs(error) for error in errors) / sum(actuals),
        "bias_mw": sum(errors) / 168,
    }
    for key, value in independently_computed.items():
        assert report["model_metrics"][key] == pytest.approx(value, abs=1e-7)
    assert (
        report["data_audit"]["sha256"]
        == hashlib.sha256((root / "opsd_de_load.csv").read_bytes()).hexdigest()
    )
    assert report["available_history_rows"] == 43824
    assert report["fitted_rows"] == 35064
    assert datetime.fromisoformat(report["fitted_end_utc"]) < origin
    validation_mae = report["validation_mean_mae_mw"]
    assert report["selected_model"] == min(validation_mae, key=validation_mae.get)
    for filename in ("forecast.png", "forecast.svg", "daily_metrics.csv", "validation.csv"):
        assert (tmp_path / filename).stat().st_size > 0
    assert (tmp_path / "forecast.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
