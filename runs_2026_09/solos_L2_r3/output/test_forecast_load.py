from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
import pytest
from pandas.testing import assert_frame_equal  # type: ignore[import-untyped]

import forecast_load
from forecast_load import (
    FORECAST_END,
    FORECAST_HOURS,
    FORECAST_LAST,
    FORECAST_START,
    LOAD_COLUMN,
    MIN_TRAINING_ROWS,
    OUTPUT_STEM,
    REQUIRED_DATA_END,
    REQUIRED_DATA_START,
    TIMESTAMP_COLUMN,
    build_calendar_features,
    build_feature_frame,
    build_forecast_table,
    calculate_metrics,
    complete_training_rows,
    forecast_index,
    format_date_range,
    load_hourly_series,
    plot_forecast,
    relative_mape_improvement,
    sha256_file,
    train_and_predict,
    write_outputs,
)


def write_load_csv(
    path: Path,
    timestamps: pd.DatetimeIndex,
    values: npt.NDArray[np.float64],
) -> None:
    pd.DataFrame(
        {
            TIMESTAMP_COLUMN: timestamps,
            LOAD_COLUMN: values,
        }
    ).to_csv(path, index=False)


def test_forecast_index_contains_exactly_requested_hours() -> None:
    index = forecast_index()

    assert len(index) == FORECAST_HOURS
    assert index[0] == FORECAST_START
    assert index[-1] == FORECAST_END - pd.Timedelta(hours=1)
    assert (index[1:] - index[:-1] == pd.Timedelta(hours=1)).all()


def test_forecast_index_rejects_naive_start() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        forecast_index(pd.Timestamp("2020-01-01"))


def test_forecast_index_converts_aware_start_to_utc() -> None:
    start = pd.Timestamp("2020-01-01 01:00:00", tz="Europe/Berlin")

    index = forecast_index(start)

    assert index[0] == FORECAST_START
    assert str(index.tz) == "UTC"


def test_forecast_index_rejects_aware_off_grid_start() -> None:
    with pytest.raises(ValueError, match="hour boundary"):
        forecast_index(pd.Timestamp("2020-01-01 01:30:00", tz="Europe/Berlin"))


def test_date_range_label_handles_month_boundary() -> None:
    start = pd.Timestamp("2020-01-28", tz="UTC")
    end = pd.Timestamp("2020-02-03", tz="UTC")

    assert format_date_range(start, end) == "28 January-3 February 2020"


def test_target_values_do_not_leak_into_target_features() -> None:
    history_start = FORECAST_START - pd.Timedelta(hours=1_000)
    index = pd.date_range(history_start, FORECAST_END, freq="h")
    load = pd.Series(np.linspace(40_000.0, 70_000.0, len(index)), index=index)
    changed = load.copy()
    changed.loc[FORECAST_START:FORECAST_END] = 999_999.0

    original_features = build_feature_frame(load)
    changed_features = build_feature_frame(changed)

    target = forecast_index()
    assert_frame_equal(
        original_features.loc[target],
        changed_features.loc[target],
    )


def test_training_rows_exclude_forecast_boundary() -> None:
    index = pd.date_range(
        FORECAST_START - pd.Timedelta(hours=2),
        FORECAST_START + pd.Timedelta(hours=1),
        freq="h",
    )
    features = pd.DataFrame({"feature": 1.0}, index=index)

    rows = complete_training_rows(features, FORECAST_START)

    assert rows.loc[FORECAST_START - pd.Timedelta(hours=1)]
    assert not rows.loc[FORECAST_START]
    assert not rows.loc[FORECAST_START + pd.Timedelta(hours=1)]


def test_minimum_training_size_is_meaningful() -> None:
    assert MIN_TRAINING_ROWS >= 365 * 24


def test_train_and_predict_rejects_misaligned_indexes() -> None:
    load_index = pd.date_range("2018-01-01", periods=2, freq="h", tz="UTC")
    feature_index = load_index + pd.Timedelta(hours=1)
    load = pd.Series([50_000.0, 51_000.0], index=load_index)
    features = pd.DataFrame({"feature": [1.0, 2.0]}, index=feature_index)

    with pytest.raises(ValueError, match="indexes must match exactly"):
        train_and_predict(load, features, FORECAST_START)


def test_train_and_predict_rejects_non_positive_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NegativeModel:
        def fit(self, features: pd.DataFrame, target: pd.Series) -> NegativeModel:
            return self

        def predict(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
            return np.full(len(features), -1.0, dtype=np.float64)

    index = pd.date_range(
        FORECAST_START - pd.Timedelta(hours=MIN_TRAINING_ROWS),
        FORECAST_LAST,
        freq="h",
    )
    load = pd.Series(np.full(len(index), 50_000.0), index=index)
    features = pd.DataFrame({"feature": np.ones(len(index))}, index=index)
    monkeypatch.setattr(forecast_load, "make_model", NegativeModel)

    with pytest.raises(ValueError, match="strictly positive"):
        train_and_predict(load, features, FORECAST_START)


def test_reformation_day_2017_is_a_public_holiday() -> None:
    index = pd.DatetimeIndex([pd.Timestamp("2017-10-31 12:00:00", tz="UTC")])

    features = build_calendar_features(index)

    assert features.iloc[0]["is_public_holiday"] == 1.0


def test_metrics_use_forecast_minus_actual_for_bias() -> None:
    actual = np.array([100.0, 200.0, 400.0])
    predicted = np.array([110.0, 180.0, 440.0])

    metrics = calculate_metrics(actual, predicted)

    assert metrics.mae_mw == pytest.approx(70.0 / 3.0)
    assert metrics.rmse_mw == pytest.approx(np.sqrt(700.0))
    assert metrics.mape_percent == pytest.approx(10.0)
    assert metrics.mean_error_mw == pytest.approx(10.0)


def test_constant_actual_values_produce_valid_json_metrics() -> None:
    metrics = calculate_metrics(np.array([100.0, 100.0]), np.array([90.0, 110.0]))

    assert metrics.r_squared is None
    json.dumps(asdict(metrics), allow_nan=False)


def test_relative_improvement_is_undefined_for_perfect_baseline() -> None:
    assert relative_mape_improvement(model_mape=0.0, baseline_mape=0.0) is None


def test_loader_rejects_empty_dataset(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=[TIMESTAMP_COLUMN, LOAD_COLUMN]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="empty"):
        load_hourly_series(path)


@pytest.mark.parametrize("invalid_value", [np.inf, -1.0, 0.0])
def test_loader_rejects_invalid_load_values(
    tmp_path: Path,
    invalid_value: float,
) -> None:
    path = tmp_path / "invalid.csv"
    timestamps = pd.date_range("2015-01-01", periods=2, freq="h", tz="UTC")
    write_load_csv(path, timestamps, np.array([40_000.0, invalid_value]))

    with pytest.raises(ValueError, match="finite and strictly positive"):
        load_hourly_series(path)


def test_loader_reports_missing_load_values(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    timestamps = pd.date_range("2015-01-01", periods=2, freq="h", tz="UTC")
    write_load_csv(path, timestamps, np.array([40_000.0, np.nan]))

    with pytest.raises(ValueError, match="missing load values"):
        load_hourly_series(path)


def test_loader_reports_missing_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "missing-timestamp.csv"
    pd.DataFrame(
        {
            TIMESTAMP_COLUMN: ["2015-01-01T00:00:00Z", None],
            LOAD_COLUMN: [40_000.0, 41_000.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing timestamps"):
        load_hourly_series(path)


def test_loader_rejects_missing_hour(tmp_path: Path) -> None:
    path = tmp_path / "missing-hour.csv"
    timestamps = pd.DatetimeIndex(
        [
            pd.Timestamp("2015-01-01 00:00:00", tz="UTC"),
            pd.Timestamp("2015-01-01 02:00:00", tz="UTC"),
        ]
    )
    write_load_csv(path, timestamps, np.full(len(timestamps), 50_000.0))

    with pytest.raises(ValueError, match="continuous"):
        load_hourly_series(path)


def test_loader_requires_exact_final_forecast_hour(tmp_path: Path) -> None:
    path = tmp_path / "missing-final-hour.csv"
    timestamps = pd.date_range(
        REQUIRED_DATA_START,
        REQUIRED_DATA_END + pd.Timedelta(hours=1),
        freq="h",
    )
    timestamps = timestamps.delete(timestamps.get_loc(REQUIRED_DATA_END))
    write_load_csv(path, timestamps, np.full(len(timestamps), 50_000.0))

    with pytest.raises(ValueError, match="must contain 2020-01-07"):
        load_hourly_series(path)


def test_loader_rejects_off_grid_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "off-grid.csv"
    timestamps = pd.date_range("2015-01-01 00:30:00", periods=50_000, freq="h", tz="UTC")
    write_load_csv(path, timestamps, np.full(len(timestamps), 50_000.0))

    with pytest.raises(ValueError, match="hour boundaries"):
        load_hourly_series(path)


def test_plot_uses_production_table_and_restores_configuration(tmp_path: Path) -> None:
    index = forecast_index()
    actual = np.linspace(40_000.0, 60_000.0, len(index), dtype=np.float64)
    predicted = actual + 500.0
    table = build_forecast_table(index, actual, predicted)
    original_font = list(matplotlib.rcParams["font.family"])
    original_color = matplotlib.rcParams["text.color"]
    plot_path = tmp_path / "plot.png"

    plot_forecast(table, plot_path)

    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0
    assert matplotlib.rcParams["font.family"] == original_font
    assert matplotlib.rcParams["text.color"] == original_color
    assert list(table.columns) == [
        "actual_load_mw",
        "forecast_load_mw",
        "error_mw",
        "absolute_percentage_error_percent",
    ]
    assert table.index.name == TIMESTAMP_COLUMN


def test_write_outputs_runs_full_pipeline_and_records_artifact_digests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SeasonalModel:
        def fit(self, features: pd.DataFrame, target: pd.Series) -> SeasonalModel:
            return self

        def predict(self, features: pd.DataFrame) -> npt.NDArray[np.float64]:
            return np.asarray(
                features[f"lag_{forecast_load.WEEK_HOURS}"].to_numpy(),
                dtype=np.float64,
            )

    index = pd.date_range(REQUIRED_DATA_START, REQUIRED_DATA_END, freq="h")
    hours = np.arange(len(index), dtype=np.float64)
    values = (
        50_000.0
        + 6_000.0 * np.sin(2 * np.pi * hours / 24)
        + 3_000.0 * np.cos(2 * np.pi * hours / (7 * 24))
        + 0.05 * hours
    )
    load = pd.Series(values, index=index, name=LOAD_COLUMN)
    features = build_feature_frame(load)
    monkeypatch.setattr(forecast_load, "make_model", SeasonalModel)

    csv_path, metrics_path, plot_path, metrics = write_outputs(load, features, tmp_path)

    assert csv_path == tmp_path / f"{OUTPUT_STEM}.csv"
    assert metrics_path == tmp_path / f"{OUTPUT_STEM}_metrics.json"
    assert plot_path == tmp_path / f"{OUTPUT_STEM}.png"
    output_table = pd.read_csv(csv_path)
    assert len(output_table) == FORECAST_HOURS
    output_timestamps = pd.to_datetime(output_table[TIMESTAMP_COLUMN], utc=True)
    assert output_timestamps.iloc[0] == FORECAST_START
    assert output_timestamps.iloc[-1] == FORECAST_LAST
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert report["accuracy"]["mape_percent"] == pytest.approx(metrics.mape_percent)
    assert report["artifact_digests"] == {
        "forecast_csv_sha256": sha256_file(csv_path),
        "plot_png_sha256": sha256_file(plot_path),
    }
