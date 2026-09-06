from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import holidays
import matplotlib as mpl
import numpy as np
import pandas as pd

LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
TIME_COLUMN = "utc_timestamp"
TRAIN_END = pd.Timestamp("2019-09-30 23:00:00", tz="UTC")
VALIDATION_END = pd.Timestamp("2019-12-31 23:00:00", tz="UTC")
TEST_END = pd.Timestamp("2020-01-07 23:00:00", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
SEED = 2020

MODEL_ORDER = ["naive", "sarima", "prophet", "lightgbm", "nbeats", "patchtst"]
MODEL_LABELS = {
    "naive": "Seasonal naive",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "TSMixer",
}
MODEL_COLORS = {
    "naive": "#2a78d6",
    "sarima": "#eb6834",
    "prophet": "#1baf7a",
    "lightgbm": "#eda100",
    "nbeats": "#e87ba4",
    "patchtst": "#008300",
}


@dataclass(frozen=True)
class DataSplits:
    full: pd.Series
    train: pd.Series
    validation: pd.Series
    train_validation: pd.Series
    test: pd.Series


@dataclass
class ForecastResult:
    name: str
    forecast: pd.Series
    runtime_seconds: float
    hyperparameters: dict[str, Any]
    quantiles: dict[float, pd.Series] = field(default_factory=dict)
    validation_mape_pct: float | None = None
    feature_importance: pd.Series | None = None
    prophet_components: pd.DataFrame | None = None


def load_splits(csv_path: Path) -> DataSplits:
    frame = pd.read_csv(csv_path, parse_dates=[TIME_COLUMN])
    expected_columns = [TIME_COLUMN, LOAD_COLUMN]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"Expected columns {expected_columns}, found {list(frame.columns)}")
    if len(frame) != 50_400:
        raise ValueError(f"Expected 50,400 rows, found {len(frame):,}")
    if frame.isna().any().any():
        raise ValueError("The input contains missing values; no imputation is permitted")

    timestamps = pd.DatetimeIndex(frame[TIME_COLUMN])
    if timestamps.tz is None:
        raise ValueError("utc_timestamp must include a UTC offset")
    timestamps = timestamps.tz_convert("UTC")
    if timestamps.has_duplicates:
        raise ValueError("The input contains duplicate timestamps")
    expected_index = pd.date_range(timestamps[0], timestamps[-1], freq="h", tz="UTC")
    if not timestamps.equals(expected_index):
        missing = expected_index.difference(timestamps)
        raise ValueError(f"The input has gaps; first missing timestamps: {list(missing[:5])}")

    values = frame[LOAD_COLUMN].astype(float).to_numpy()
    full = pd.Series(values, index=timestamps, name="load_mw")
    if full.index[0] != pd.Timestamp("2015-01-01 00:00:00", tz="UTC"):
        raise ValueError(f"Unexpected first timestamp: {full.index[0]}")
    if full.index[-1] != pd.Timestamp("2020-09-30 23:00:00", tz="UTC"):
        raise ValueError(f"Unexpected last timestamp: {full.index[-1]}")

    train = full.loc[:TRAIN_END]
    validation = full.loc[TRAIN_END + pd.Timedelta(hours=1) : VALIDATION_END]
    train_validation = full.loc[:VALIDATION_END]
    test = full.loc[TEST_START:TEST_END]
    expected_sizes = (41_616, 2_208, 43_824, 168)
    actual_sizes = (len(train), len(validation), len(train_validation), len(test))
    if actual_sizes != expected_sizes:
        raise ValueError(f"Unexpected split sizes: {actual_sizes}, expected {expected_sizes}")
    return DataSplits(full, train, validation, train_validation, test)


def mape_pct(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted arrays must have the same shape")
    if np.any(actual_values == 0.0):
        raise ValueError("MAPE is undefined for zero actual values")
    return float(np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100.0)


def rmse_mw(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    residual = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(np.square(residual))))


def mae_mw(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    residual = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(residual)))


def score_result(result: ForecastResult, actual: pd.Series) -> dict[str, Any]:
    forecast = result.forecast.reindex(actual.index)
    if forecast.isna().any():
        raise ValueError(f"{result.name} forecast does not cover the complete test index")
    jan1 = actual.index.date == pd.Timestamp("2020-01-01").date()
    jan2_onward = ~jan1
    return {
        "name": result.name,
        "mape_test_pct": mape_pct(actual, forecast),
        "rmse_test_mw": rmse_mw(actual, forecast),
        "mae_test_mw": mae_mw(actual, forecast),
        "mape_jan1_pct": mape_pct(actual[jan1], forecast[jan1]),
        "mape_jan2_to_jan7_pct": mape_pct(actual[jan2_onward], forecast[jan2_onward]),
        "runtime_seconds": float(result.runtime_seconds),
        "hyperparameters": result.hyperparameters,
    }


def per_day_mape(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    aligned = predicted.reindex(actual.index)
    absolute_percentage_error = np.abs((actual - aligned) / actual) * 100.0
    return absolute_percentage_error.groupby(actual.index.date).mean()


def pinball_loss(
    actual: pd.Series | np.ndarray,
    quantile_forecast: pd.Series | np.ndarray,
    quantile: float,
) -> float:
    error = np.asarray(actual, dtype=float) - np.asarray(quantile_forecast, dtype=float)
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


def interval_coverage(
    actual: pd.Series | np.ndarray,
    lower: pd.Series | np.ndarray,
    upper: pd.Series | np.ndarray,
) -> float:
    actual_values = np.asarray(actual, dtype=float)
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    return float(np.mean((actual_values >= lower_values) & (actual_values <= upper_values)))


def german_holiday_dates(index: pd.DatetimeIndex) -> set[date]:
    years = sorted(set(index.year.tolist()))
    calendar = holidays.Germany(years=years)
    return set(calendar.keys())


def make_utc_series(values: np.ndarray, index: pd.DatetimeIndex, name: str) -> pd.Series:
    array = np.asarray(values, dtype=float).reshape(-1)
    if len(array) != len(index):
        raise ValueError(f"Expected {len(index)} values for {name}, found {len(array)}")
    return pd.Series(array, index=index, name=name)


def set_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#c3c2b7",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#e1e0d9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
