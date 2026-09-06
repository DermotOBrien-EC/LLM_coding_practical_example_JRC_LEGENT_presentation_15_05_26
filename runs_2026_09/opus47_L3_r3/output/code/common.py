"""Shared plumbing for the forecasting bake-off.

Centralises the data loader, the train / validation / test split, evaluation
metrics, the palette that every figure reuses so a given model keeps its
colour, and a couple of matplotlib defaults.

Every model script imports from here; the orchestrator (forecast.py) also
imports the palette and metric helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_DIR: Path = Path(__file__).resolve().parent.parent
CSV_PATH: Path = RUN_DIR / "opsd_de_load.csv"
FIG_DIR: Path = RUN_DIR / "figures"

RANDOM_SEED: int = 42

TRAIN_END: pd.Timestamp = pd.Timestamp("2019-09-30 23:00", tz="UTC")
VAL_START: pd.Timestamp = pd.Timestamp("2019-10-01 00:00", tz="UTC")
VAL_END: pd.Timestamp = pd.Timestamp("2019-12-31 23:00", tz="UTC")
TEST_START: pd.Timestamp = pd.Timestamp("2020-01-01 00:00", tz="UTC")
TEST_END: pd.Timestamp = pd.Timestamp("2020-01-07 23:00", tz="UTC")

MODEL_ORDER: list[str] = [
    "naive",
    "sarima",
    "prophet",
    "lightgbm",
    "nbeats",
    "patchtst",
]

MODEL_DISPLAY: dict[str, str] = {
    "naive": "Seasonal-naive (168 h)",
    "sarima": "SARIMA",
    "prophet": "Prophet (DE holidays)",
    "lightgbm": "LightGBM (features)",
    "nbeats": "N-BEATS",
    "patchtst": "TSMixer",
}

# tab10 in a fixed order so each model owns one colour across every figure.
_TAB10 = plt.get_cmap("tab10").colors
MODEL_COLOR: dict[str, tuple[float, float, float]] = {
    name: _TAB10[i] for i, name in enumerate(MODEL_ORDER)
}


@dataclass
class Split:
    """The three chronological slices used everywhere."""

    train: pd.Series
    val: pd.Series
    test: pd.Series

    @property
    def train_plus_val(self) -> pd.Series:
        return pd.concat([self.train, self.val])

    @property
    def full(self) -> pd.Series:
        return pd.concat([self.train, self.val, self.test])


def load_series() -> pd.Series:
    """Load the OPSD hourly load and return a tz-aware pandas Series in MW."""
    df = pd.read_csv(CSV_PATH, parse_dates=["utc_timestamp"])
    df = df.sort_values("utc_timestamp").reset_index(drop=True)
    s = df.set_index("utc_timestamp")["DE_load_actual_entsoe_transparency"].astype(float)
    s.index = s.index.tz_convert("UTC") if s.index.tz is not None else s.index.tz_localize("UTC")
    s.name = "load_mw"
    # Cheap invariants: the AGENTS.md file promises 50,400 clean hourly rows.
    if len(s) != 50_400:
        raise AssertionError(f"expected 50400 rows, got {len(s)}")
    if s.isna().any():
        raise AssertionError("unexpected NaN in load series")
    spacing = s.index.to_series().diff().dropna().unique()
    if not (len(spacing) == 1 and spacing[0] == pd.Timedelta(hours=1)):
        raise AssertionError(f"expected strict 1 h spacing, got {spacing}")
    return s


def get_split() -> Split:
    """Return the fixed Train / Validation / Test partition."""
    s = load_series()
    train = s.loc[: TRAIN_END]
    val = s.loc[VAL_START:VAL_END]
    test = s.loc[TEST_START:TEST_END]
    if len(test) != 168:
        raise AssertionError(f"expected 168 test hours, got {len(test)}")
    return Split(train=train, val=val, test=test)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error in percent. Load is always > 0 here."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Standard pinball / quantile loss."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def coverage(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of observations inside [lo, hi]."""
    y_true = np.asarray(y_true, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def per_day_mape(index: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray) -> pd.Series:
    """MAPE grouped by the calendar date of each timestamp."""
    df = pd.DataFrame({"y": y_true, "yhat": y_pred}, index=index)
    df["date"] = df.index.date
    grouped = df.groupby("date").apply(lambda g: mape(g["y"].values, g["yhat"].values))
    grouped.index = pd.to_datetime(grouped.index)
    return grouped


def apply_style() -> None:
    """Consistent, minimal, publication-friendly matplotlib defaults."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def as_naive_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """darts wants tz-naive indices. Strip tz without shifting the wall clock."""
    if idx.tz is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx


def to_hourly_naive(series: pd.Series) -> pd.Series:
    """Return a copy of `series` with a tz-naive hourly index."""
    out = series.copy()
    out.index = as_naive_utc(out.index)
    out.index.freq = "h"
    return out


def iter_test_index(split: Split) -> pd.DatetimeIndex:
    return split.test.index


def hyperparameter_summary(d: dict[str, object]) -> dict[str, object]:
    """Make hyperparameters JSON-serialisable (scalars only)."""
    clean: dict[str, object] = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            clean[k] = v
        elif isinstance(v, Iterable):
            clean[k] = [x for x in v]
        else:
            clean[k] = str(v)
    return clean
