"""Shared utilities for the six-model forecasting bake-off.

Loading, splitting, metrics, and figure style live here so each per-model
script imports one place. The goal is a single source of truth for the split
boundaries and metric definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import numpy as np
import pandas as pd


DATA_FILE = Path(__file__).resolve().parent.parent / "opsd_de_load.csv"
LOAD_COL = "DE_load_actual_entsoe_transparency"

# Fixed split boundaries. Test week is 168 h starting 2020-01-01 00:00 UTC.
TRAIN_START = pd.Timestamp("2015-01-01 00:00", tz="UTC")
TRAIN_END = pd.Timestamp("2019-09-30 23:00", tz="UTC")
VAL_START = pd.Timestamp("2019-10-01 00:00", tz="UTC")
VAL_END = pd.Timestamp("2019-12-31 23:00", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01 00:00", tz="UTC")
TEST_END = pd.Timestamp("2020-01-07 23:00", tz="UTC")

# One qualitative colour per model, shared across all figures so the reader
# recognises each model at a glance.
MODEL_ORDER: list[str] = [
    "naive",
    "sarima",
    "prophet",
    "lightgbm",
    "nbeats",
    "patchtst",
]
MODEL_LABELS: dict[str, str] = {
    "naive": "Seasonal-naive",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "TSMixer (PatchTST slot)",
}
_TAB10 = mpl.colormaps["tab10"].colors
MODEL_COLOURS: dict[str, tuple[float, float, float]] = {
    name: _TAB10[i] for i, name in enumerate(MODEL_ORDER)
}

RANDOM_SEED = 42


def apply_figure_style() -> None:
    """Set a shared matplotlib style. Sans-serif, 300 dpi, tight layout."""
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
        }
    )


@dataclass(frozen=True)
class Splits:
    """The three windows used for fitting, selection, and final scoring."""

    train: pd.Series
    val: pd.Series
    test: pd.Series
    trainval: pd.Series  # Train + Validation concatenated, used for refit.


def load_series() -> pd.Series:
    """Read the CSV and return an hourly, UTC-indexed float series (MW)."""
    df = pd.read_csv(DATA_FILE)
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
    series = (
        df.set_index("utc_timestamp")[LOAD_COL]
        .astype("float64")
        .sort_index()
    )
    series.index.freq = pd.infer_freq(series.index)
    return series


def make_splits() -> Splits:
    """Slice the series into train, validation, test, and refit windows."""
    series = load_series()
    train = series.loc[TRAIN_START:TRAIN_END]
    val = series.loc[VAL_START:VAL_END]
    test = series.loc[TEST_START:TEST_END]
    trainval = series.loc[TRAIN_START:VAL_END]
    assert len(test) == 168, f"test window is {len(test)} h, expected 168"
    return Splits(train=train, val=val, test=test, trainval=trainval)


def full_series() -> pd.Series:
    """The complete series 2015-01-01 to 2020-09-30 for context plots."""
    return load_series()


# ----------------------------- metrics ----------------------------------


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error in percent."""
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100.0)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean square error in MW."""
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error in MW."""
    actual = np.asarray(actual, dtype="float64")
    predicted = np.asarray(predicted, dtype="float64")
    return float(np.mean(np.abs(actual - predicted)))


def pinball(actual: np.ndarray, predicted_q: np.ndarray, q: float) -> float:
    """Pinball loss at quantile q (in MW; lower is better)."""
    actual = np.asarray(actual, dtype="float64")
    predicted_q = np.asarray(predicted_q, dtype="float64")
    diff = actual - predicted_q
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of test hours inside [lower, upper]."""
    actual = np.asarray(actual, dtype="float64")
    lower = np.asarray(lower, dtype="float64")
    upper = np.asarray(upper, dtype="float64")
    return float(np.mean((actual >= lower) & (actual <= upper)))


def per_day_mape(
    actual: pd.Series, predicted: pd.Series
) -> dict[str, float]:
    """MAPE grouped by calendar day of the test week."""
    df = pd.DataFrame({"y": actual.values, "yhat": predicted.values}, index=actual.index)
    out: dict[str, float] = {}
    for day, block in df.groupby(df.index.date):
        out[str(day)] = mape(block["y"].values, block["yhat"].values)
    return out


def align(pred: Iterable[float], index: pd.DatetimeIndex) -> pd.Series:
    """Wrap a raw array/list of predictions as a pandas Series on `index`."""
    arr = np.asarray(list(pred), dtype="float64")
    assert arr.shape == (len(index),), (
        f"prediction length {arr.shape} vs index {len(index)}"
    )
    return pd.Series(arr, index=index, name="forecast")
