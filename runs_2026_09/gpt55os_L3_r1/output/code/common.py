from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import holidays
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
TIMESTAMP_COLUMN = "utc_timestamp"
TRAIN_START = pd.Timestamp("2015-01-01 00:00:00")
TRAIN_END = pd.Timestamp("2019-09-30 23:00:00")
VAL_START = pd.Timestamp("2019-10-01 00:00:00")
VAL_END = pd.Timestamp("2019-12-31 23:00:00")
TEST_START = pd.Timestamp("2020-01-01 00:00:00")
TEST_END = pd.Timestamp("2020-01-07 23:00:00")
SEED = 20260906

MODEL_NAMES = {
    "naive": "Seasonal naive",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "TSMixer",
}
MODEL_ORDER = ["naive", "sarima", "prophet", "lightgbm", "nbeats", "patchtst"]
MODEL_COLORS = {
    "naive": "#0072B2",
    "sarima": "#D55E00",
    "prophet": "#009E73",
    "lightgbm": "#CC79A7",
    "nbeats": "#56B4E9",
    "patchtst": "#E69F00",
}


@dataclass(slots=True)
class ForecastResult:
    name: str
    point: np.ndarray
    runtime_seconds: float
    hyperparameters: dict[str, Any]
    validation_mape_pct: float | None = None
    q025: np.ndarray | None = None
    q10: np.ndarray | None = None
    q25: np.ndarray | None = None
    q50: np.ndarray | None = None
    q75: np.ndarray | None = None
    q90: np.ndarray | None = None
    q975: np.ndarray | None = None
    feature_importance: pd.DataFrame | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DataSplits:
    full: pd.DataFrame
    train: pd.Series
    validation: pd.Series
    train_validation: pd.Series
    test: pd.Series


def timer() -> float:
    return time.perf_counter()


def elapsed_seconds(start: float) -> float:
    return float(time.perf_counter() - start)


def load_data(path: Path) -> DataSplits:
    df = pd.read_csv(path)
    required = {TIMESTAMP_COLUMN, LOAD_COLUMN}
    if set(df.columns) != required:
        raise ValueError(f"Expected columns {sorted(required)}, got {df.columns.tolist()}")
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], utc=True).dt.tz_convert(None)
    df = df.sort_values(TIMESTAMP_COLUMN).set_index(TIMESTAMP_COLUMN)
    df.index.name = "timestamp_utc"
    if len(df) != 50_400:
        raise ValueError(f"Expected 50400 rows, got {len(df)}")
    if df[LOAD_COLUMN].isna().any():
        raise ValueError("Load column contains NaN values")
    expected_index = pd.date_range(df.index.min(), df.index.max(), freq="h")
    if not df.index.equals(expected_index):
        raise ValueError("Hourly timestamp sequence has gaps or duplicates")
    train = df.loc[TRAIN_START:TRAIN_END, LOAD_COLUMN]
    validation = df.loc[VAL_START:VAL_END, LOAD_COLUMN]
    train_validation = df.loc[TRAIN_START:VAL_END, LOAD_COLUMN]
    test = df.loc[TEST_START:TEST_END, LOAD_COLUMN]
    if len(test) != 168:
        raise ValueError(f"Expected 168 test observations, got {len(test)}")
    return DataSplits(full=df, train=train, validation=validation, train_validation=train_validation, test=test)


def german_holiday_dates(years: range | list[int]) -> set[pd.Timestamp]:
    calendar = holidays.Germany(years=list(years))
    return {pd.Timestamp(day) for day in calendar.keys()}


def holiday_mask(index: pd.DatetimeIndex) -> np.ndarray:
    years = range(int(index.min().year) - 1, int(index.max().year) + 2)
    holiday_days = german_holiday_dates(years)
    normalized = pd.DatetimeIndex(index).normalize()
    return np.array([day in holiday_days for day in normalized], dtype=bool)


def mape_pct(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    forecast_arr = np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs((actual_arr - forecast_arr) / actual_arr)) * 100.0)


def mae_mw(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float))))


def rmse_mw(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(math.sqrt(np.mean((np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float)) ** 2)))


def model_metrics(result: ForecastResult, actual: pd.Series) -> dict[str, Any]:
    point = np.asarray(result.point, dtype=float)
    if len(point) != len(actual):
        raise ValueError(f"{result.name} forecast length {len(point)} does not match actual length {len(actual)}")
    jan1_mask = actual.index.normalize() == pd.Timestamp("2020-01-01")
    work_mask = ~jan1_mask
    return {
        "name": result.name,
        "mape_test_pct": mape_pct(actual.to_numpy(), point),
        "rmse_test_mw": rmse_mw(actual.to_numpy(), point),
        "mae_test_mw": mae_mw(actual.to_numpy(), point),
        "mape_jan1_pct": mape_pct(actual.to_numpy()[jan1_mask], point[jan1_mask]),
        "mape_jan2_to_jan7_pct": mape_pct(actual.to_numpy()[work_mask], point[work_mask]),
        "runtime_seconds": float(result.runtime_seconds),
        "hyperparameters": result.hyperparameters,
    }


def per_day_mape(actual: pd.Series, forecast: np.ndarray) -> pd.Series:
    rows: list[tuple[str, float]] = []
    forecast_arr = np.asarray(forecast, dtype=float)
    for day, group in actual.groupby(actual.index.normalize()):
        positions = actual.index.normalize() == day
        rows.append((day.strftime("%Y-%m-%d"), mape_pct(group.to_numpy(), forecast_arr[positions])))
    return pd.Series(dict(rows), name="mape_pct")


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    return float(np.mean((actual_arr >= np.asarray(lower, dtype=float)) & (actual_arr <= np.asarray(upper, dtype=float))))


def pinball_loss(actual: np.ndarray, forecast_quantile: np.ndarray, quantile: float) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(forecast_quantile, dtype=float)
    errors = actual_arr - pred_arr
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1.0) * errors)))


def normal_interval(mean: np.ndarray, se: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    mean_arr = np.asarray(mean, dtype=float)
    se_arr = np.asarray(se, dtype=float)
    return mean_arr - z * se_arr, mean_arr + z * se_arr


def set_figure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): convert(val) for key, val in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    path.write_text(json.dumps(convert(payload), indent=2) + "\n", encoding="utf-8")


def uname_string() -> str:
    return " ".join(platform.uname())


def format_metric(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def test_window_label(index: pd.DatetimeIndex) -> None:
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlim(index.min(), index.max())
