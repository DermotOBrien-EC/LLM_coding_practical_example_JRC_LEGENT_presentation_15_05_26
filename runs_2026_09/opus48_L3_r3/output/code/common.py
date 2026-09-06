"""Shared building blocks for the load-forecasting bake-off.

This module holds the parts every model script needs so the rest of the code
stays short and readable:

- loading and sanity-checking the load series,
- the fixed train / validation / test date boundaries,
- the error measures (MAPE, RMSE, MAE, pinball loss, interval coverage),
- one fixed colour per model so a model looks the same in every figure,
- a small result container each model fills in.

The audience for the comments is a policy economist or engineer, not a machine
learning specialist, so the intent is spelled out in plain language.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # write PNGs without needing a screen
import matplotlib.pyplot as plt  # noqa: E402

# One random seed used everywhere we can set one, so a rerun reproduces numbers.
SEED: int = 42

# The single load column in the CSV and a short label for plots.
LOAD_COL: str = "DE_load_actual_entsoe_transparency"
LOAD_LABEL: str = "Load (MW)"

# The data file sits in the run directory, one level above this code/ folder.
DATA_PATH: Path = Path(__file__).resolve().parent.parent / "opsd_de_load.csv"

# --- The three time windows, as plain naive-UTC timestamps. -----------------
# Train: the bulk of history used to fit each model.
TRAIN_START = pd.Timestamp("2015-01-01 00:00")
TRAIN_END = pd.Timestamp("2019-09-30 23:00")
# Validation: used only to choose hyperparameters / model orders, never to fit
# the final model and never to score the headline result.
VAL_START = pd.Timestamp("2019-10-01 00:00")
VAL_END = pd.Timestamp("2019-12-31 23:00")
# Test: the held-out week we score everyone on. Touched only at the very end.
TEST_START = pd.Timestamp("2020-01-01 00:00")
TEST_END = pd.Timestamp("2020-01-07 23:00")

# The forecast horizon in hours (one week).
HORIZON: int = 168

# The six models, in the order we like to list them (rising complexity).
MODEL_ORDER: list[str] = [
    "naive",
    "sarima",
    "prophet",
    "lightgbm",
    "nbeats",
    "patchtst",
]

# Human-readable names for titles and tables.
MODEL_DISPLAY: dict[str, str] = {
    "naive": "Seasonal-naive",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "PatchTST (TSMixer)",
}

# Fixed colour per model from the Set2 qualitative palette, so each model keeps
# the same colour in every figure.
_SET2 = plt.get_cmap("Set2").colors
MODEL_COLORS: dict[str, tuple[float, float, float]] = {
    name: _SET2[i] for i, name in enumerate(MODEL_ORDER)
}


def set_figure_style() -> None:
    """Set one consistent look for every figure (sans-serif, readable sizes)."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def load_series() -> pd.Series:
    """Load the hourly load series and check it is clean.

    Returns a pandas Series indexed by a naive-UTC hourly timestamp. We drop the
    time zone label (every stamp is already UTC) so the deep-learning library
    does not complain, without changing any actual clock time.
    """
    frame = pd.read_csv(DATA_PATH)
    stamps = pd.to_datetime(frame["utc_timestamp"], utc=True).dt.tz_localize(None)
    series = pd.Series(frame[LOAD_COL].to_numpy(), index=stamps, name=LOAD_COL)
    series = series.sort_index()

    # Sanity checks. If any fail we want to know, not paper over it.
    assert len(series) == 50_400, f"expected 50400 rows, got {len(series)}"
    assert series.isna().sum() == 0, "found NaN values in the load series"
    full = pd.date_range(series.index[0], series.index[-1], freq="h")
    assert series.index.equals(full), "found gaps in the hourly index"
    return series


def slice_window(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return the inclusive slice [start, end] of a series."""
    return series.loc[start:end]


def train_series(series: pd.Series) -> pd.Series:
    """The train window only (for fitting during selection)."""
    return slice_window(series, TRAIN_START, TRAIN_END)


def val_series(series: pd.Series) -> pd.Series:
    """The validation window only (for scoring candidate settings)."""
    return slice_window(series, VAL_START, VAL_END)


def train_plus_val_series(series: pd.Series) -> pd.Series:
    """Train + validation combined (used to refit the final chosen model)."""
    return slice_window(series, TRAIN_START, VAL_END)


def test_series(series: pd.Series) -> pd.Series:
    """The held-out test week (168 hours)."""
    return slice_window(series, TEST_START, TEST_END)


# --- Error measures ---------------------------------------------------------


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, in percent.

    This answers 'on average, how many percent off is the forecast?'. Load is
    always well above zero here, so dividing by it is safe.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100.0)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean square error, in MW. Punishes big misses more than small ones."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error, in MW. The plain average size of the miss."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def pinball_loss(actual: np.ndarray, quantile_pred: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss at level q.

    This scores a single quantile forecast: it charges q for under-prediction
    and (1 - q) for over-prediction, so it rewards a forecast that lands the
    quantile in the right place. Lower is better.
    """
    actual = np.asarray(actual, dtype=float)
    quantile_pred = np.asarray(quantile_pred, dtype=float)
    diff = actual - quantile_pred
    loss = np.where(diff >= 0, q * diff, (q - 1.0) * diff)
    return float(np.mean(loss))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actual values that fall inside [lower, upper]."""
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside))


# --- Holiday helper ---------------------------------------------------------


def is_public_holiday_de(index: pd.DatetimeIndex) -> np.ndarray:
    """Boolean array: is each timestamp a German nationwide public holiday?

    We use the `holidays` package with no federal state given, which yields the
    holidays that hold across all of Germany (New Year's Day among them). This
    is knowable in advance from the calendar, so it is not external data.
    """
    import holidays as holidays_pkg

    years = range(index.year.min(), index.year.max() + 1)
    de_holidays = holidays_pkg.Germany(years=list(years))
    dates = index.normalize()
    return np.array([d.date() in de_holidays for d in dates], dtype=bool)


# --- Result container -------------------------------------------------------


@dataclasses.dataclass
class ModelResult:
    """Everything one model hands back after forecasting the test week.

    point:        the 168-hour point forecast (MW).
    quantiles:    optional dict {quantile level -> 168-hour array}. Provided by
                  every model except the naive baseline, so we can score
                  prediction intervals for whichever model wins.
    runtime_seconds: wall-clock time to select, refit and forecast this model.
    hyperparameters: the settings actually used (for the transcript / metrics).
    extra:        model-specific extras (e.g. LightGBM feature importances or
                  the fitted Prophet object) used only for optional figures.
    """

    name: str
    point: np.ndarray
    runtime_seconds: float
    hyperparameters: dict
    quantiles: dict[float, np.ndarray] | None = None
    extra: dict = dataclasses.field(default_factory=dict)


# The quantile levels the winning model must supply so we can report 80% and
# 95% intervals (0.1/0.9 and 0.025/0.975) and pinball loss at 0.1/0.5/0.9.
WINNER_QUANTILES: tuple[float, ...] = (0.025, 0.1, 0.5, 0.9, 0.975)
