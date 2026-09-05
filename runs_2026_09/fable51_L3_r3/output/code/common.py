"""Shared pieces for the load forecasting bake-off.

Everything the six model scripts and the orchestrator have in common lives
here: how the data is loaded and split, how forecasts are scored, and how
figures are styled so every model keeps the same colour in every plot.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import holidays
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# The study mandates a script called prophet.py, which would shadow the
# installed `prophet` package whenever this directory is first on sys.path
# (it is, for any script run from here). Moving the directory to the end
# lets installed packages win while our own modules stay importable.
_CODE_DIR = str(Path(__file__).resolve().parent)
if _CODE_DIR in sys.path:
    sys.path.remove(_CODE_DIR)
    sys.path.append(_CODE_DIR)
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

RUN_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = RUN_DIR / "opsd_de_load.csv"
FIG_DIR = RUN_DIR / "figures"
TARGET = "DE_load_actual_entsoe_transparency"

TRAIN_START = pd.Timestamp("2015-01-01 00:00", tz="UTC")
TRAIN_END = pd.Timestamp("2019-09-30 23:00", tz="UTC")
VAL_START = pd.Timestamp("2019-10-01 00:00", tz="UTC")
VAL_END = pd.Timestamp("2019-12-31 23:00", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01 00:00", tz="UTC")
TEST_END = pd.Timestamp("2020-01-07 23:00", tz="UTC")
EXPECTED_ROWS = 50_400

HORIZON = 168
SEED = 42
# The five quantiles every probabilistic model must return. 0.1/0.9 bound
# the 80 % interval, 0.025/0.975 bound the 95 % interval, 0.5 is the median.
QUANTILES: tuple[float, ...] = (0.025, 0.1, 0.5, 0.9, 0.975)

MODEL_NAMES: list[str] = ["naive", "sarima", "prophet", "lightgbm", "nbeats", "patchtst"]
MODEL_LABELS: dict[str, str] = {
    "naive": "Seasonal naive (t-168 h)",
    "sarima": "SARIMA",
    "prophet": "Prophet (DE holidays)",
    "lightgbm": "LightGBM (features)",
    "nbeats": "N-BEATS",
    "patchtst": "TSMixer (PatchTST slot)",
}
# One fixed tab10 colour per model, used in every figure.
_TAB10 = plt.get_cmap("tab10").colors
MODEL_COLORS: dict[str, tuple[float, float, float]] = {
    name: _TAB10[i] for i, name in enumerate(MODEL_NAMES)
}
OBSERVED_COLOR = "black"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_series() -> pd.Series:
    """Read the CSV and return the load as an hourly, UTC-indexed series.

    The function checks the guarantees the study relies on (row count, no
    missing values, no gaps) and fails loudly if any of them is broken.
    """
    df = pd.read_csv(DATA_PATH, parse_dates=["utc_timestamp"])
    series = df.set_index("utc_timestamp")[TARGET].astype("float64")
    series.index = pd.DatetimeIndex(series.index).tz_convert("UTC")
    series.index.name = "utc_timestamp"
    series.name = "load_mw"

    if len(series) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(series)}")
    if series.isna().any():
        raise ValueError("series contains NaN values")
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError("timestamps are not strictly increasing")
    full_index = pd.date_range(series.index[0], series.index[-1], freq="h", tz="UTC")
    if len(full_index) != len(series):
        raise ValueError("hourly index has gaps")
    return series.asfreq("h")


def split(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (train, validation, test) exactly as the study defines them."""
    train = series[TRAIN_START:TRAIN_END]
    val = series[VAL_START:VAL_END]
    test = series[TEST_START:TEST_END]
    if len(test) != HORIZON:
        raise ValueError(f"test window has {len(test)} rows, expected {HORIZON}")
    return train, val, test


def test_index() -> pd.DatetimeIndex:
    """The 168 hourly timestamps of the held-out test week."""
    return pd.date_range(TEST_START, TEST_END, freq="h", tz="UTC")


def validation_origins(val: pd.Series, horizon: int = HORIZON) -> list[pd.Timestamp]:
    """Forecast origins for rolling one-week-ahead scoring on the validation set.

    The validation window holds 13 whole weeks (2,184 of its 2,208 hours), so
    every model is scored on 13 non-overlapping 168-hour forecasts, each made
    from the hour just before the week starts. This mirrors how the test week
    is scored and keeps the selection metric comparable across models.
    """
    n_weeks = len(val) // horizon
    return [val.index[k * horizon] for k in range(n_weeks)]


# --------------------------------------------------------------------------
# Calendar features (derived from the timestamp only)
# --------------------------------------------------------------------------
def german_holidays(years: range | list[int]) -> holidays.HolidayBase:
    """German federal public holidays (no state-specific ones)."""
    return holidays.Germany(years=years)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour, weekday, month, weekend and holiday flags for each timestamp.

    Load follows the local clock, so the flags are computed in Europe/Berlin
    time: 23:00 UTC on 31 December is already New Year's Day in Germany.
    """
    local = index.tz_convert("Europe/Berlin")
    hol = german_holidays(range(local.year.min(), local.year.max() + 1))
    local_dates = local.date
    frame = pd.DataFrame(
        {
            "hour": local.hour,
            "day_of_week": local.dayofweek,
            "month": local.month,
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "is_public_holiday_de": np.array([d in hol for d in local_dates], dtype=int),
        },
        index=index,
    )
    return frame


# --------------------------------------------------------------------------
# Forecast container and scoring
# --------------------------------------------------------------------------
@dataclass
class ForecastResult:
    """Everything the orchestrator needs from one model."""

    name: str
    point: pd.Series
    quantiles: dict[float, pd.Series] | None
    runtime_seconds: float
    hyperparameters: dict[str, Any]
    validation_mape_pct: float | None
    extras: dict[str, Any] = field(default_factory=dict)


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs(actual - forecast) / np.abs(actual)) * 100.0)


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs(actual - forecast)))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actual values that fall inside [lower, upper]."""
    actual = np.asarray(actual, dtype=float)
    inside = (actual >= np.asarray(lower)) & (actual <= np.asarray(upper))
    return float(np.mean(inside))


def pinball_loss(actual: np.ndarray, quantile_forecast: np.ndarray, tau: float) -> float:
    """Average pinball loss, the proper score for a single quantile forecast."""
    diff = np.asarray(actual, dtype=float) - np.asarray(quantile_forecast, dtype=float)
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def per_day_mape(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    """MAPE for each UTC calendar day of the test week."""
    err = (actual - forecast).abs() / actual.abs() * 100.0
    return err.groupby(err.index.date).mean()


def evaluate_point(actual: pd.Series, result: ForecastResult) -> dict[str, Any]:
    """Point-forecast metrics on the test window, including the holiday split."""
    fc = result.point.reindex(actual.index)
    if fc.isna().any():
        raise ValueError(f"{result.name}: forecast does not cover the test window")
    jan1 = actual.index.date == TEST_START.date()
    daily = per_day_mape(actual, fc)
    return {
        "name": result.name,
        "mape_test_pct": mape(actual.values, fc.values),
        "rmse_test_mw": rmse(actual.values, fc.values),
        "mae_test_mw": mae(actual.values, fc.values),
        "mape_jan1_pct": mape(actual.values[jan1], fc.values[jan1]),
        "mape_jan2_to_jan7_pct": mape(actual.values[~jan1], fc.values[~jan1]),
        "per_day_mape_pct": {str(d): float(v) for d, v in daily.items()},
        "validation_mape_pct": result.validation_mape_pct,
        "runtime_seconds": result.runtime_seconds,
        "hyperparameters": result.hyperparameters,
    }


def evaluate_intervals(actual: pd.Series, result: ForecastResult) -> dict[str, float]:
    """Coverage and pinball losses for a model that returned quantiles."""
    if result.quantiles is None:
        raise ValueError(f"{result.name} has no prediction intervals")
    q = {tau: s.reindex(actual.index).values for tau, s in result.quantiles.items()}
    y = actual.values
    return {
        "coverage_80pct": coverage(y, q[0.1], q[0.9]),
        "coverage_95pct": coverage(y, q[0.025], q[0.975]),
        "pinball_loss_q10": pinball_loss(y, q[0.1], 0.1),
        "pinball_loss_q50": pinball_loss(y, q[0.5], 0.5),
        "pinball_loss_q90": pinball_loss(y, q[0.9], 0.9),
    }


def rolling_validation_mape(
    val: pd.Series, forecasts: list[pd.Series], origins: list[pd.Timestamp]
) -> float:
    """Pool the 13 one-week validation forecasts into one MAPE figure."""
    actual_parts = []
    forecast_parts = []
    for origin, fc in zip(origins, forecasts):
        window = val[origin : origin + pd.Timedelta(hours=len(fc) - 1)]
        actual_parts.append(window.values)
        forecast_parts.append(fc.reindex(window.index).values)
    return mape(np.concatenate(actual_parts), np.concatenate(forecast_parts))


class Timer:
    """Wall-clock timer used as a context manager."""

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.start


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "figure.dpi": 100,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path
