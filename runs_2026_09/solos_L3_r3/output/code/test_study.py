from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common import ForecastResult, LOAD_COLUMN, evaluate_point_forecast, load_and_split_data
from forecast import _validate_result
from lightgbm_features import (
    FEATURE_NAMES,
    build_training_frame,
    make_feature_row,
    residual_quantile_forecast,
)


def test_data_contract() -> None:
    splits = load_and_split_data(Path(__file__).parents[1] / "opsd_de_load.csv")

    assert len(splits.full) == 50_400
    assert len(splits.train) == 41_616
    assert len(splits.validation) == 2_208
    assert len(splits.test) == 168
    assert splits.full[LOAD_COLUMN].isna().sum() == 0


def test_point_metrics() -> None:
    actual = np.array([100.0, 200.0])
    forecast = np.array([90.0, 220.0])

    metrics = evaluate_point_forecast(actual, forecast)

    assert metrics["mape_pct"] == 10.0
    assert metrics["mae_mw"] == 15.0
    assert np.isclose(metrics["rmse_mw"], np.sqrt(250.0))


def test_lag_and_rolling_features_are_strictly_past_only() -> None:
    index = pd.date_range("2015-01-01", periods=9_000, freq="h")
    series = pd.Series(np.arange(9_000, dtype=float), index=index, name=LOAD_COLUMN)

    frame = build_training_frame(series)
    timestamp = frame.index[0]
    row = frame.loc[timestamp]
    position = series.index.get_loc(timestamp)

    assert row["lag_24h"] == series.iloc[position - 24]
    assert row["lag_168h"] == series.iloc[position - 168]
    assert row["lag_8736h"] == series.iloc[position - 8_736]
    assert row["rolling_mean_24h"] == series.iloc[position - 24 : position].mean()
    assert row["rolling_mean_168h"] == series.iloc[position - 168 : position].mean()

    history = series.iloc[:position]
    serving_row = make_feature_row(history, timestamp)
    np.testing.assert_allclose(
        row.loc[list(FEATURE_NAMES)].to_numpy(dtype=float),
        np.asarray([serving_row[name] for name in FEATURE_NAMES], dtype=float),
    )


def test_future_feature_row_uses_only_supplied_history() -> None:
    index = pd.date_range("2019-01-01", periods=9_000, freq="h")
    history = pd.Series(np.arange(9_000, dtype=float), index=index, name=LOAD_COLUMN)
    timestamp = index[-1] + pd.Timedelta(hours=1)

    row = make_feature_row(history, timestamp)

    assert row["lag_24h"] == history.iloc[-24]
    assert row["lag_168h"] == history.iloc[-168]
    assert row["lag_8736h"] == history.iloc[-8_736]
    assert row["rolling_mean_24h"] == history.iloc[-24:].mean()


def test_residual_quantiles_use_validation_errors_by_hour() -> None:
    validation_index = pd.date_range("2019-10-01", periods=48, freq="h")
    validation_forecast = np.full(48, 100.0)
    residuals = np.concatenate([np.full(24, -10.0), np.full(24, 10.0)])
    validation_actual = pd.Series(
        validation_forecast + residuals,
        index=validation_index,
    )
    forecast_index = pd.date_range("2020-01-01", periods=24, freq="h")
    point = np.full(24, 200.0)

    quantiles = residual_quantile_forecast(
        point,
        validation_actual,
        validation_forecast,
        forecast_index,
    )

    assert set(quantiles) == {0.025, 0.1, 0.5, 0.9, 0.975}
    np.testing.assert_allclose(quantiles[0.5], point)
    assert np.all(quantiles[0.025] <= quantiles[0.1])
    assert np.all(quantiles[0.1] <= quantiles[0.5])
    assert np.all(quantiles[0.5] <= quantiles[0.9])
    assert np.all(quantiles[0.9] <= quantiles[0.975])


def test_non_naive_result_requires_complete_quantiles() -> None:
    result = ForecastResult(
        name="prophet",
        point=np.ones(2),
        runtime_seconds=0.0,
        hyperparameters={},
    )

    with pytest.raises(ValueError, match="incomplete quantile set"):
        _validate_result(result, "prophet", 2)
