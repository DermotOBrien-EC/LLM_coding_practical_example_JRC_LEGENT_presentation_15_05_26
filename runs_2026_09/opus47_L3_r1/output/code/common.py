"""Shared utilities: data loading, split masks, metrics, plot style.

Every model module imports from here so figure colours, split boundaries,
and metric definitions are consistent across the study.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data path relative to the project root (the current working directory).
DATA_CSV: Path = Path(__file__).resolve().parent.parent / "opsd_de_load.csv"
FIG_DIR: Path = Path(__file__).resolve().parent.parent / "figures"
LOAD_COL: str = "DE_load_actual_entsoe_transparency"
TS_COL: str = "utc_timestamp"

# Split boundaries. All timestamps in UTC.
TRAIN_END: pd.Timestamp = pd.Timestamp("2019-09-30 23:00", tz="UTC")
VAL_START: pd.Timestamp = pd.Timestamp("2019-10-01 00:00", tz="UTC")
VAL_END: pd.Timestamp = pd.Timestamp("2019-12-31 23:00", tz="UTC")
TEST_START: pd.Timestamp = pd.Timestamp("2020-01-01 00:00", tz="UTC")
TEST_END: pd.Timestamp = pd.Timestamp("2020-01-07 23:00", tz="UTC")

# Deterministic colour map: each model keeps the same colour in every figure.
# tab10 chosen for good separation and print-safety.
MODEL_ORDER: list[str] = [
    "naive",
    "sarima",
    "prophet",
    "lightgbm",
    "nbeats",
    "patchtst",
]
_tab10 = plt.get_cmap("tab10").colors
MODEL_COLOURS: dict[str, tuple[float, float, float]] = {
    "naive": _tab10[7],     # grey
    "sarima": _tab10[0],    # blue
    "prophet": _tab10[2],   # green
    "lightgbm": _tab10[1],  # orange
    "nbeats": _tab10[3],    # red
    "patchtst": _tab10[4],  # purple
}
MODEL_DISPLAY: dict[str, str] = {
    "naive": "Seasonal-naive (168h)",
    "sarima": "SARIMA",
    "prophet": "Prophet (DE holidays)",
    "lightgbm": "LightGBM (features)",
    "nbeats": "N-BEATS",
    "patchtst": "TSMixer",  # PatchTST unavailable in installed darts 0.41.0
}

RANDOM_SEED: int = 20260906


@dataclass(frozen=True)
class Splits:
    """The full series and its three time-index slices."""

    series: pd.Series           # full hourly series, UTC-indexed
    train: pd.Series            # up to TRAIN_END inclusive
    val: pd.Series              # VAL_START to VAL_END inclusive
    test: pd.Series             # TEST_START to TEST_END inclusive
    train_val: pd.Series        # up to VAL_END inclusive (used for final fit)


def load_series() -> pd.Series:
    """Load the OPSD CSV and return the load column as a UTC-indexed Series."""
    df = pd.read_csv(DATA_CSV, parse_dates=[TS_COL])
    df[TS_COL] = pd.to_datetime(df[TS_COL], utc=True)
    df = df.set_index(TS_COL).sort_index()
    series = df[LOAD_COL].asfreq("h")
    if series.isna().any():
        raise ValueError(
            f"Unexpected NaN in load series: {int(series.isna().sum())} rows"
        )
    return series


def make_splits() -> Splits:
    """Slice the series into train / val / test / train+val."""
    s = load_series()
    return Splits(
        series=s,
        train=s.loc[:TRAIN_END],
        val=s.loc[VAL_START:VAL_END],
        test=s.loc[TEST_START:TEST_END],
        train_val=s.loc[:VAL_END],
    )


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, in percent."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean square error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss at level q in (0, 1)."""
    y_true = np.asarray(y_true, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    diff = y_true - q_pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def coverage(y_true: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Fraction of observations inside the [low, high] interval."""
    y_true = np.asarray(y_true, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    return float(np.mean((y_true >= low) & (y_true <= high)))


def score_point(
    y_true: pd.Series, y_pred: pd.Series
) -> dict[str, float]:
    """Return the standard three metrics plus the Jan 1 vs Jan 2-7 split."""
    yt = y_true.to_numpy()
    yp = y_pred.reindex(y_true.index).to_numpy()
    d = {
        "mape_test_pct": mape(yt, yp),
        "rmse_test_mw": rmse(yt, yp),
        "mae_test_mw": mae(yt, yp),
    }
    # Jan 1 (Wednesday, public holiday) vs Jan 2-7 (rest of the week).
    idx = y_true.index
    jan1 = (idx >= pd.Timestamp("2020-01-01", tz="UTC")) & (
        idx < pd.Timestamp("2020-01-02", tz="UTC")
    )
    rest = ~jan1
    d["mape_jan1_pct"] = mape(yt[jan1], yp[jan1])
    d["mape_jan2_to_jan7_pct"] = mape(yt[rest], yp[rest])
    return d


def apply_plot_style() -> None:
    """One place to pin the visual conventions used by every figure."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
