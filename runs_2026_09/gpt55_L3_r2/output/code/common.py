from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOAD_COL = "DE_load_actual_entsoe_transparency"
TIME_COL = "utc_timestamp"
TRAIN_START = pd.Timestamp("2015-01-01 00:00:00+00:00")
TRAIN_END = pd.Timestamp("2019-09-30 23:00:00+00:00")
VAL_START = pd.Timestamp("2019-10-01 00:00:00+00:00")
VAL_END = pd.Timestamp("2019-12-31 23:00:00+00:00")
TEST_START = pd.Timestamp("2020-01-01 00:00:00+00:00")
TEST_END = pd.Timestamp("2020-01-07 23:00:00+00:00")

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
    "naive": "#1f77b4",
    "sarima": "#ff7f0e",
    "prophet": "#2ca02c",
    "lightgbm": "#d62728",
    "nbeats": "#9467bd",
    "patchtst": "#8c564b",
}

@dataclass
class ForecastResult:
    name: str
    display_name: str
    forecast: pd.Series
    runtime_seconds: float
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    q10: pd.Series | None = None
    q50: pd.Series | None = None
    q90: pd.Series | None = None
    lower80: pd.Series | None = None
    upper80: pd.Series | None = None
    lower95: pd.Series | None = None
    upper95: pd.Series | None = None
    fitted_model: Any | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def load_data(path: Path = Path("opsd_de_load.csv")) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    validate_data(df)
    return df


def validate_data(df: pd.DataFrame) -> None:
    if len(df) != 50_400:
        raise ValueError(f"Expected 50,400 rows, found {len(df):,}")
    if df[LOAD_COL].isna().any():
        raise ValueError("Load column contains missing values")
    if df[TIME_COL].isna().any():
        raise ValueError("Timestamp column contains missing values")
    deltas = df[TIME_COL].diff().dropna()
    if not (deltas == pd.Timedelta(hours=1)).all():
        raise ValueError("Timestamp column is not exactly hourly")
    if df[TIME_COL].iloc[0] != pd.Timestamp("2015-01-01 00:00:00+00:00"):
        raise ValueError("Unexpected first timestamp")
    if df[TIME_COL].iloc[-1] != pd.Timestamp("2020-09-30 23:00:00+00:00"):
        raise ValueError("Unexpected final timestamp")


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[(df[TIME_COL] >= TRAIN_START) & (df[TIME_COL] <= TRAIN_END)].copy()
    val = df[(df[TIME_COL] >= VAL_START) & (df[TIME_COL] <= VAL_END)].copy()
    train_val = df[(df[TIME_COL] >= TRAIN_START) & (df[TIME_COL] <= VAL_END)].copy()
    test = df[(df[TIME_COL] >= TEST_START) & (df[TIME_COL] <= TEST_END)].copy()
    if (len(train), len(val), len(train_val), len(test)) != (41_616, 2_208, 43_824, 168):
        raise ValueError("Unexpected split sizes")
    return train, val, train_val, test


def series_from_frame(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df[LOAD_COL].to_numpy(dtype=float), index=df[TIME_COL], name=LOAD_COL)


def mape_pct(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs((a - p) / a)) * 100.0)


def mae_mw(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(a - p)))


def rmse_mw(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def pinball_loss(actual: pd.Series | np.ndarray, predicted_quantile: pd.Series | np.ndarray, q: float) -> float:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted_quantile, dtype=float)
    diff = a - p
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def align_series(values: pd.Series | np.ndarray, index: pd.Index, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        out = values.copy()
        out.index = index
        out.name = name
        return out.astype(float)
    return pd.Series(np.asarray(values, dtype=float), index=index, name=name)


def evaluate_result(result: ForecastResult, actual: pd.Series) -> dict[str, Any]:
    forecast = result.forecast.reindex(actual.index)
    jan1_mask = actual.index.normalize() == pd.Timestamp("2020-01-01 00:00:00+00:00")
    jan2_7_mask = ~jan1_mask
    return {
        "name": result.name,
        "mape_test_pct": mape_pct(actual, forecast),
        "rmse_test_mw": rmse_mw(actual, forecast),
        "mae_test_mw": mae_mw(actual, forecast),
        "mape_jan1_pct": mape_pct(actual[jan1_mask], forecast[jan1_mask]),
        "mape_jan2_to_jan7_pct": mape_pct(actual[jan2_7_mask], forecast[jan2_7_mask]),
        "runtime_seconds": float(result.runtime_seconds),
        "hyperparameters": result.hyperparameters,
    }


def per_day_mape(result: ForecastResult, actual: pd.Series) -> pd.Series:
    forecast = result.forecast.reindex(actual.index)
    rows: dict[str, float] = {}
    for day, actual_day in actual.groupby(actual.index.strftime("%Y-%m-%d")):
        rows[day] = mape_pct(actual_day, forecast.loc[actual_day.index])
    return pd.Series(rows, name=result.name)


def coverage(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    a = actual.to_numpy(dtype=float)
    lo = lower.reindex(actual.index).to_numpy(dtype=float)
    hi = upper.reindex(actual.index).to_numpy(dtype=float)
    return float(np.mean((a >= lo) & (a <= hi)) * 100.0)


def de_holiday_flags(index: pd.DatetimeIndex) -> pd.Series:
    years = sorted(set(index.year.tolist()))
    de_holidays = holidays.Germany(years=years)
    dates = [ts.date() for ts in index]
    return pd.Series([date in de_holidays for date in dates], index=index)


def apply_figure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
