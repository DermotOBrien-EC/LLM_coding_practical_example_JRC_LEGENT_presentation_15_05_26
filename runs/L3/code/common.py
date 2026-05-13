"""Shared utilities for the forecasting bake-off.

Plain language: this module is the boring plumbing that every model script
needs — where the CSV lives, where the train/validation/test boundaries are,
how to score a forecast, which colour each model gets on every figure.
Keeping all of it here means the per-model scripts can stay short and the
orchestrator can trust that "naive forecast for the test window" means the
same thing across the whole study.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = ROOT / "opsd_de_load.csv"
FIG_DIR: Path = ROOT / "figures"
METRICS_JSON: Path = ROOT / "metrics.json"
METRICS_CSV: Path = ROOT / "metrics.csv"
TRANSCRIPT: Path = ROOT / "transcript.md"

LOAD_COL: str = "DE_load_actual_entsoe_transparency"

# ---------------------------------------------------------------------------
# Splits — fixed once for the whole study
# ---------------------------------------------------------------------------

# Train end (inclusive at hourly resolution).
TRAIN_END: pd.Timestamp = pd.Timestamp("2019-09-30 23:00")
# Validation runs from the hour after TRAIN_END through this point.
VAL_END: pd.Timestamp = pd.Timestamp("2019-12-31 23:00")
# Test window — held out, never touched until final scoring.
TEST_START: pd.Timestamp = pd.Timestamp("2020-01-01 00:00")
TEST_END: pd.Timestamp = pd.Timestamp("2020-01-07 23:00")
N_TEST: int = 168

RANDOM_SEED: int = 42


# ---------------------------------------------------------------------------
# Visual identity — each model keeps its colour across every figure.
# ---------------------------------------------------------------------------

# tab10 picks, chosen so the colours stay distinct in print and on screen.
MODEL_ORDER: list[str] = [
    "naive",
    "sarima",
    "prophet",
    "lightgbm",
    "nbeats",
    "patchtst",
]

MODEL_LABEL: dict[str, str] = {
    "naive": "Seasonal naive",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "TSMixer*",  # PatchTST substitute, asterisk flags it.
}

_TAB10 = mpl.colormaps["tab10"].colors
MODEL_COLOR: dict[str, tuple[float, float, float]] = {
    "naive": _TAB10[7],     # grey
    "sarima": _TAB10[0],    # blue
    "prophet": _TAB10[2],   # green
    "lightgbm": _TAB10[1],  # orange
    "nbeats": _TAB10[4],    # purple
    "patchtst": _TAB10[3],  # red
}


def set_figure_style() -> None:
    """Set the rcParams once so every figure looks the same."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "lines.linewidth": 1.4,
        }
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_load_series() -> pd.Series:
    """Return the German hourly load as a pandas Series indexed by UTC time.

    The CSV stores UTC timestamps in ISO-8601 with offset. We strip the
    timezone after parsing because some downstream libraries (Prophet,
    parts of darts) refuse tz-aware indexes; all timestamps in this study
    are UTC by construction, so the strip is lossless.
    """
    df = pd.read_csv(DATA_PATH, parse_dates=["utc_timestamp"])
    df["utc_timestamp"] = df["utc_timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    series = df.set_index("utc_timestamp")[LOAD_COL].astype(float)
    series.index.name = "timestamp"
    series.index.freq = "h"
    # Sanity: must be hourly, no NaN, no gaps. We don't repair — we shout.
    if series.isna().any():
        raise ValueError("Load series contains NaN — refusing to silently impute.")
    deltas = series.index.to_series().diff().dropna()
    if (deltas != pd.Timedelta("1h")).any():
        raise ValueError("Load series has missing hours.")
    if len(series) != 50_400:
        raise ValueError(f"Expected 50,400 hourly rows, got {len(series)}.")
    return series


def split_series(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (train, validation, train_plus_val, test) Series."""
    train = series.loc[: TRAIN_END]
    val = series.loc[TRAIN_END + pd.Timedelta("1h") : VAL_END]
    trainval = series.loc[: VAL_END]
    test = series.loc[TEST_START:TEST_END]
    if len(test) != N_TEST:
        raise ValueError(f"Test window has {len(test)} rows, expected {N_TEST}.")
    return train, val, trainval, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute percentage error, in percent.

    German load is far from zero everywhere in the test window, so the
    standard MAPE definition is safe — no protection against zero division
    is required.
    """
    return float(np.mean(np.abs((actual - pred) / actual)) * 100.0)


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def pinball_loss(actual: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Quantile (pinball) loss at level q in (0, 1).

    Reads as: when the forecast under-shoots, pay q times the error; when
    it over-shoots, pay (1 - q) times the error. q=0.5 reduces to half the
    MAE.
    """
    diff = actual - pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals inside [lower, upper] inclusive."""
    return float(np.mean((actual >= lower) & (actual <= upper)))


# ---------------------------------------------------------------------------
# Forecast container
# ---------------------------------------------------------------------------


@dataclass
class ModelForecast:
    """Everything one model produces for the 168-hour test window.

    `q10`, `q50`, `q90` are optional — the seasonal naive does not produce
    intervals, so it sets them to None.
    """

    name: str
    point: np.ndarray              # shape (168,)
    q10: np.ndarray | None = None
    q50: np.ndarray | None = None
    q90: np.ndarray | None = None
    lower80: np.ndarray | None = None
    upper80: np.ndarray | None = None
    lower95: np.ndarray | None = None
    upper95: np.ndarray | None = None
    runtime_seconds: float = 0.0
    hyperparameters: dict = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Holiday helper
# ---------------------------------------------------------------------------


def german_federal_holidays(years: Iterable[int]) -> set[pd.Timestamp]:
    """Set of dates that are public holidays in Germany at the federal level.

    The `holidays` package returns subdivision holidays in some calls; we use
    `holidays.Germany()` without a subdivision, which returns the union of
    federal-level holidays — Jan 1, Good Friday, Easter Monday, May 1, etc.
    Only Jan 1 falls inside the test window, but we keep this general so the
    LightGBM feature engineering doesn't have to special-case it.
    """
    import holidays as _hol  # local import keeps top-level imports light

    de = _hol.Germany(years=list(years))
    return {pd.Timestamp(d) for d in de.keys()}


def is_public_holiday_de(index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean array marking German federal public holidays.

    We compare the calendar date in UTC. For a study that mixes UTC
    timestamps with calendar-based holiday dates this is an approximation;
    the only test-window holiday (Jan 1, 2020) is fully contained in UTC
    Jan 1, so the approximation is exact for this exercise.
    """
    years = sorted({int(t.year) for t in index})
    holidays_set = german_federal_holidays(years)
    dates = pd.to_datetime(index.date)
    return np.array([d in holidays_set for d in dates])


# ---------------------------------------------------------------------------
# Per-model breakdown helpers used by the orchestrator
# ---------------------------------------------------------------------------


def jan1_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True for hours falling on 2020-01-01 (the only test-window holiday)."""
    return np.array([t.date() == pd.Timestamp("2020-01-01").date() for t in index])

