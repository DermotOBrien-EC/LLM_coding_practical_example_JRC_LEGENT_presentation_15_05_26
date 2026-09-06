"""Shared machinery for the German-load forecasting bake-off.

This module holds everything the six model scripts and the orchestrator need
to agree on: how the data is loaded and split, how a forecast is scored, and
how every figure is styled so a model keeps the same colour in every plot.

Design choices worth stating up front:

* All timestamps are UTC. The source file is timezone-aware ("+00:00"); we
  drop the timezone label after loading because several of the forecasting
  libraries choke on timezone-aware indices, and dropping a label that is
  constant (always UTC) changes nothing about the data.
* A forecast is represented as a plain pandas Series indexed by the 168 test
  hours. Probabilistic models additionally return a dictionary mapping a
  quantile level (e.g. 0.1) to such a Series.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import holidays
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# code/ lives one level below the run directory; everything is written there.
RUN_DIR: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = RUN_DIR / "opsd_de_load.csv"
FIGURE_DIR: Path = RUN_DIR / "figures"
LOAD_COLUMN: str = "DE_load_actual_entsoe_transparency"

# --------------------------------------------------------------------------
# Split boundaries (inclusive, hourly, UTC)
# --------------------------------------------------------------------------

TRAIN_END = pd.Timestamp("2019-09-30 23:00")
VAL_START = pd.Timestamp("2019-10-01 00:00")
VAL_END = pd.Timestamp("2019-12-31 23:00")   # also the Train+Val refit boundary
TEST_START = pd.Timestamp("2020-01-01 00:00")
TEST_END = pd.Timestamp("2020-01-07 23:00")

# Jan 1 2020 is a Wednesday and a German federal holiday (New Year's Day).
JAN1_END = pd.Timestamp("2020-01-01 23:00")
JAN2_START = pd.Timestamp("2020-01-02 00:00")

# --------------------------------------------------------------------------
# Model registry: fixed order, labels, and colours used across all figures.
# --------------------------------------------------------------------------

MODEL_ORDER: list[str] = ["naive", "sarima", "prophet", "lightgbm", "nbeats", "patchtst"]

MODEL_LABELS: dict[str, str] = {
    "naive": "Seasonal-naive",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "PatchTST (Transformer)",
}

# One fixed colour per model, drawn from the tab10 qualitative palette, so a
# model is the same colour in every figure.
_TAB10 = plt.get_cmap("tab10").colors
MODEL_COLORS: dict[str, tuple[float, float, float]] = {
    "naive": _TAB10[7],      # grey  - the baseline
    "sarima": _TAB10[0],     # blue
    "prophet": _TAB10[1],    # orange
    "lightgbm": _TAB10[2],   # green
    "nbeats": _TAB10[3],     # red
    "patchtst": _TAB10[4],   # purple
}

# Global random seed used everywhere a seed is accepted.
SEED: int = 42


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclasses.dataclass
class ModelResult:
    """Everything one model hands back to the orchestrator.

    point:        the 168-hour point forecast, indexed by the test hours.
    quantiles:    optional {quantile_level: Series}. Empty for the naive model.
    runtime_s:    wall-clock seconds the whole fit/select/refit/forecast took.
    hyperparameters: the configuration that was actually used, for the record.
    extra:        model-specific artefacts (e.g. LightGBM feature importances,
                  Prophet seasonal components) needed only for certain figures.
    """

    name: str
    point: pd.Series
    runtime_s: float
    hyperparameters: dict
    quantiles: dict[float, pd.Series] = dataclasses.field(default_factory=dict)
    extra: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Splits:
    """The four series a model needs, all sharing the same units (MW)."""

    train: pd.Series      # 2015-01-01 .. 2019-09-30
    val: pd.Series        # 2019-10-01 .. 2019-12-31
    test: pd.Series       # 2020-01-01 .. 2020-01-07  (held out)
    trainval: pd.Series   # 2015-01-01 .. 2019-12-31  (final refit data)
    full: pd.Series       # everything, for context plots and lag lookups


# --------------------------------------------------------------------------
# Data loading and verification
# --------------------------------------------------------------------------


def load_series() -> pd.Series:
    """Load the load series as a timezone-naive UTC hourly pandas Series."""
    frame = pd.read_csv(DATA_PATH)
    index = pd.to_datetime(frame["utc_timestamp"], utc=True).dt.tz_localize(None)
    series = pd.Series(frame[LOAD_COLUMN].to_numpy(dtype=float), index=index, name="load")
    series = series.sort_index()
    series.index.freq = pd.infer_freq(series.index)
    return series


def verify_series(series: pd.Series) -> dict[str, object]:
    """Check the data is what the prompt promises: 50,400 clean hourly rows.

    Returns a small report rather than raising, so the caller can log it.
    """
    full_range = pd.date_range(series.index[0], series.index[-1], freq="h")
    report = {
        "n_rows": int(len(series)),
        "n_nan": int(series.isna().sum()),
        "n_gaps": int(len(full_range) - len(series)),
        "start": str(series.index[0]),
        "end": str(series.index[-1]),
        "monotonic": bool(series.index.is_monotonic_increasing),
    }
    return report


def get_splits(series: pd.Series) -> Splits:
    """Cut the series into train / val / test / train+val, no overlaps."""
    return Splits(
        train=series.loc[:TRAIN_END].copy(),
        val=series.loc[VAL_START:VAL_END].copy(),
        test=series.loc[TEST_START:TEST_END].copy(),
        trainval=series.loc[:VAL_END].copy(),
        full=series.copy(),
    )


# --------------------------------------------------------------------------
# German federal holidays (allowed a-priori calendar feature)
# --------------------------------------------------------------------------


def german_holidays(years: range) -> holidays.HolidayBase:
    """German nationwide (federal) public holidays for the given years.

    holidays.Germany() with no state ("subdiv") argument returns exactly the
    nine holidays that hold across the whole country, which is what "German
    federal holidays" means. State-specific days (e.g. Epiphany) are excluded
    on purpose: the load series is national.
    """
    return holidays.Germany(years=years)


def is_public_holiday_de(index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean array: is each timestamp's calendar date a federal holiday?"""
    years = range(index.year.min(), index.year.max() + 1)
    cal = german_holidays(years)
    dates = index.normalize()
    return np.array([d.date() in cal for d in dates], dtype=bool)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, in percent. Load is always positive."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100.0)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def pinball_loss(actual: np.ndarray, quantile_pred: np.ndarray, q: float) -> float:
    """Average pinball (quantile) loss at level q. Lower is better."""
    actual = np.asarray(actual, dtype=float)
    quantile_pred = np.asarray(quantile_pred, dtype=float)
    diff = actual - quantile_pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals falling inside [lower, upper] (inclusive)."""
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((actual >= lower) & (actual <= upper)))


def holiday_breakdown(actual: pd.Series, predicted: pd.Series) -> tuple[float, float]:
    """MAPE on Jan 1 (holiday) and MAPE on Jan 2-7 (working/weekend days).

    This is the diagnostic the prompt cares about: models that ignore the
    holiday calendar should be much worse on Jan 1.
    """
    jan1 = actual.loc[:JAN1_END]
    jan2_7 = actual.loc[JAN2_START:]
    mape_jan1 = mape(jan1.to_numpy(), predicted.loc[:JAN1_END].to_numpy())
    mape_rest = mape(jan2_7.to_numpy(), predicted.loc[JAN2_START:].to_numpy())
    return mape_jan1, mape_rest


# --------------------------------------------------------------------------
# Figure styling
# --------------------------------------------------------------------------

TS_FIGSIZE = (11.0, 6.0)
SQ_FIGSIZE = (6.0, 6.0)


def set_style() -> None:
    """One consistent look for every figure: sans-serif, 300 dpi on save."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    })


def save_fig(fig: plt.Figure, filename: str) -> Path:
    """Save a figure into figures/ at 300 dpi and close it."""
    FIGURE_DIR.mkdir(exist_ok=True)
    path = FIGURE_DIR / filename
    fig.savefig(path)
    plt.close(fig)
    return path


# A convenient type alias for a model entry point.
ModelRunner = Callable[[Splits], ModelResult]
