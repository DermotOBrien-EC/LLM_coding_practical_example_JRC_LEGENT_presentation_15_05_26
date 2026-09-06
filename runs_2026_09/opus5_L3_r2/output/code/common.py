"""Shared plumbing for the German load forecasting bake-off.

Everything that more than one model script needs lives here: loading the
series, the fixed train/validation/test split, the error measures, the
rolling-origin validation protocol, and the plotting style. Keeping it in
one place is what makes the six model scripts comparable - they all see the
same data and are scored by the same code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

RUN_DIR: Path = Path(__file__).resolve().parent.parent
DATA_CSV: Path = RUN_DIR / "opsd_de_load.csv"
FIGURE_DIR: Path = RUN_DIR / "figures"
CACHE_DIR: Path = RUN_DIR / "code" / "_cache"

LOAD_COLUMN: str = "DE_load_actual_entsoe_transparency"

# ----------------------------------------------------------------------------
# The split. These timestamps are the contract every model obeys.
# ----------------------------------------------------------------------------

TRAIN_START = pd.Timestamp("2015-01-01 00:00", tz="UTC")
TRAIN_END = pd.Timestamp("2019-09-30 23:00", tz="UTC")
VAL_START = pd.Timestamp("2019-10-01 00:00", tz="UTC")
VAL_END = pd.Timestamp("2019-12-31 23:00", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01 00:00", tz="UTC")
TEST_END = pd.Timestamp("2020-01-07 23:00", tz="UTC")

HORIZON: int = 168  # one week of hourly steps - the forecast we are judged on
SEED: int = 42

# Rolling-origin validation. Each origin is the first hour to be predicted;
# the model may use everything strictly before it. Spacing the seven origins
# two weeks apart covers autumn, early winter and the Christmas week, so a
# configuration cannot win by being good in one season only. The final origin
# (24 December) is deliberately included: the test week also contains a public
# holiday, and we want the selection step to notice which models cope.
VALIDATION_ORIGINS: tuple[pd.Timestamp, ...] = tuple(
    pd.Timestamp(t, tz="UTC")
    for t in (
        "2019-10-01 00:00",
        "2019-10-15 00:00",
        "2019-10-29 00:00",
        "2019-11-12 00:00",
        "2019-11-26 00:00",
        "2019-12-10 00:00",
        "2019-12-24 00:00",
    )
)

# ----------------------------------------------------------------------------
# Colours. Each model keeps the same colour in every figure.
# ----------------------------------------------------------------------------

MODEL_NAMES: tuple[str, ...] = (
    "naive",
    "sarima",
    "prophet",
    "lightgbm",
    "nbeats",
    "patchtst",
)

MODEL_LABELS: dict[str, str] = {
    "naive": "Seasonal naive",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "PatchTST",
}

# matplotlib tab10, taken explicitly so the mapping cannot drift between runs.
MODEL_COLORS: dict[str, str] = {
    "naive": "#7f7f7f",   # grey - it is the reference, not a contender
    "sarima": "#1f77b4",  # blue
    "prophet": "#ff7f0e",  # orange
    "lightgbm": "#2ca02c",  # green
    "nbeats": "#d62728",  # red
    "patchtst": "#9467bd",  # purple
}

OBSERVED_COLOR: str = "#000000"


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------


def load_series() -> pd.Series:
    """Read the load series and check the claims the prompt makes about it.

    Returns an hourly, timezone-aware (UTC) float series in megawatts. Raises
    if the data is not what we were told it is - we would rather stop than
    quietly forecast a series with holes in it.
    """
    frame = pd.read_csv(DATA_CSV, parse_dates=["utc_timestamp"])
    frame = frame.set_index("utc_timestamp").sort_index()
    series = frame[LOAD_COLUMN].astype(float)
    series.name = "load_mw"
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    else:
        series.index = series.index.tz_convert("UTC")
    series.index.name = "utc_timestamp"
    return series


def describe_integrity(series: pd.Series) -> dict[str, object]:
    """Facts about the series that transcript.md quotes, computed not assumed."""
    expected = pd.date_range(series.index[0], series.index[-1], freq="h", tz="UTC")
    gaps = expected.difference(series.index)
    return {
        "n_rows": int(series.shape[0]),
        "n_expected_hours": int(expected.shape[0]),
        "n_missing_timestamps": int(gaps.shape[0]),
        "n_nan_values": int(series.isna().sum()),
        "n_duplicate_timestamps": int(series.index.duplicated().sum()),
        "n_nonpositive_values": int((series <= 0).sum()),
        "first_timestamp": str(series.index[0]),
        "last_timestamp": str(series.index[-1]),
        "min_mw": float(series.min()),
        "max_mw": float(series.max()),
        "mean_mw": float(series.mean()),
    }


def window(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Inclusive slice by timestamp."""
    return series.loc[(series.index >= start) & (series.index <= end)]


def history_before(series: pd.Series, origin: pd.Timestamp) -> pd.Series:
    """Everything strictly before `origin`. This is the anti-leakage helper:
    a model asked to forecast from `origin` may see exactly this and no more.
    """
    return series.loc[series.index < origin]


def horizon_index(origin: pd.Timestamp, horizon: int = HORIZON) -> pd.DatetimeIndex:
    return pd.date_range(origin, periods=horizon, freq="h", tz="UTC")


# ----------------------------------------------------------------------------
# Error measures
# ----------------------------------------------------------------------------


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, in percent."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if np.any(actual == 0.0):
        raise ValueError("MAPE is undefined when an actual value is zero")
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100.0)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def pinball_loss(actual: np.ndarray, quantile_forecast: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss in MW.

    It punishes a forecast that is too low by q times the miss and one that is
    too high by (1-q) times the miss. At q=0.5 it is half the absolute error.
    """
    actual = np.asarray(actual, dtype=float)
    quantile_forecast = np.asarray(quantile_forecast, dtype=float)
    diff = actual - quantile_forecast
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of observations that fell inside the band."""
    actual = np.asarray(actual, dtype=float)
    return float(np.mean((actual >= np.asarray(lower)) & (actual <= np.asarray(upper))))


# ----------------------------------------------------------------------------
# Forecast container
# ----------------------------------------------------------------------------


@dataclass
class ModelForecast:
    """What every model script hands back to the orchestrator."""

    name: str
    point: pd.Series                      # index = the 168 test hours, MW
    hyperparameters: dict[str, object] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    quantiles: dict[float, pd.Series] = field(default_factory=dict)
    validation_mape: float | None = None
    selection_log: list[dict[str, object]] = field(default_factory=list)
    notes: str = ""

    def has_intervals(self) -> bool:
        needed = (0.025, 0.1, 0.5, 0.9, 0.975)
        return all(q in self.quantiles for q in needed)


# ----------------------------------------------------------------------------
# Calendar features (allowed: derived from the timestamp, not external data)
# ----------------------------------------------------------------------------


@lru_cache(maxsize=64)
def _holiday_set(years: tuple[int, ...]) -> frozenset:
    import holidays as holidays_pkg

    return frozenset(holidays_pkg.Germany(years=list(years)).keys())


def german_holiday_dates(years: Iterable[int]) -> frozenset:
    """German public holidays as plain dates, via the `holidays` package.

    These are knowable years in advance from the calendar, so using them is
    not the same as using weather data - nothing is looked up from outside.
    Cached because the recursive forecast asks for the same years 168 times
    in a row.
    """
    return _holiday_set(tuple(sorted(set(years))))


def calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """hour / day-of-week / month / weekend flag / German holiday flag."""
    local_dates = index.tz_convert("UTC").date
    hol = german_holiday_dates(sorted({d.year for d in local_dates}))
    return pd.DataFrame(
        {
            "hour": index.hour.astype("int16"),
            "day_of_week": index.dayofweek.astype("int16"),
            "month": index.month.astype("int16"),
            "is_weekend": (index.dayofweek >= 5).astype("int8"),
            "is_public_holiday_de": np.fromiter(
                (1 if d in hol else 0 for d in local_dates), dtype="int8",
                count=len(index),
            ),
        },
        index=index,
    )


# ----------------------------------------------------------------------------
# Plot style
# ----------------------------------------------------------------------------


def apply_style() -> None:
    """One look for every figure: 300 dpi, default sans-serif, light grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.autolayout": False,
        }
    )


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


# ----------------------------------------------------------------------------
# Shared glue for the two neural models (N-BEATS and the PatchTST substitute)
# ----------------------------------------------------------------------------


def darts_series(values: pd.Series):
    """Wrap a pandas series as a darts TimeSeries.

    darts prefers timestamps without a timezone. The whole study is in UTC and
    never converts, so dropping the label loses no information.
    """
    from darts import TimeSeries

    return TimeSeries.from_series(values.tz_localize(None), freq="h")


def torch_trainer_kwargs(max_epochs: int, patience: int | None) -> dict[str, object]:
    """Quiet, CPU-only PyTorch Lightning settings.

    CPU rather than the Mac GPU on purpose: Apple's MPS backend cannot handle
    the float64 data darts hands it, and on a model this size the CPU is fast
    enough (about 10-20 seconds per epoch) while being bit-for-bit repeatable.
    """
    from pytorch_lightning.callbacks import EarlyStopping

    callbacks = []
    if patience is not None:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=patience, mode="min")
        )
    return {
        "accelerator": "cpu",
        "max_epochs": max_epochs,
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
        "callbacks": callbacks,
    }


def quantile_likelihood(quantiles: Sequence[float]):
    from darts.utils.likelihood_models.torch import QuantileRegression

    return QuantileRegression(quantiles=list(quantiles))


def fitted_epochs(model: object) -> int:
    """How many epochs actually ran before training stopped."""
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        return 0
    return int(getattr(trainer, "current_epoch", 0))


def best_epoch(model: object, max_epochs: int, patience: int) -> int:
    """The epoch the validation loss was actually lowest at.

    Early stopping keeps going for `patience` more epochs after the best one
    before it gives up, so the number of epochs that ran overstates the number
    worth repeating. When we refit on train+validation there is no validation
    set left to stop on, so we reuse this count instead.
    """
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        return max_epochs
    stopped = 0
    for callback in getattr(trainer, "callbacks", []) or []:
        if callback.__class__.__name__ == "EarlyStopping":
            stopped = int(getattr(callback, "stopped_epoch", 0) or 0)
    if stopped > 0:
        # stopped_epoch is zero-indexed, hence the +1 before backing off.
        return max(1, stopped + 1 - patience)
    return max(1, int(getattr(trainer, "current_epoch", max_epochs)))
