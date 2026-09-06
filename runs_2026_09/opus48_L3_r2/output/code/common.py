"""Shared building blocks for the load-forecasting bake-off.

This module is the single place where we load the data, define the
train / validation / test split, score forecasts, and fix the look of the
figures. Every model script imports from here so that all six models see the
exact same data and are judged by the exact same yardstick.

Plain-language note on the split (see the prompt for the full rationale):

- Train:      2015-01-01 00:00 .. 2019-09-30 23:00  (used for initial fitting)
- Validation: 2019-10-01 00:00 .. 2019-12-31 23:00  (used only to pick settings)
- Test:       2020-01-01 00:00 .. 2020-01-07 23:00  (168 hours, judged once)

We never let a model see the test week while it is being fitted or tuned.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# code/ lives one level below the working directory; everything else is
# written next to the data file.
ROOT: Path = Path(__file__).resolve().parent.parent
DATA_CSV: Path = ROOT / "opsd_de_load.csv"
FIGURES_DIR: Path = ROOT / "figures"
LOAD_COLUMN: str = "DE_load_actual_entsoe_transparency"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

SEED: int = 42


def _pick_accelerator() -> str:
    """Use Apple's MPS GPU if present, otherwise the CPU.

    The two deep models train several times faster on MPS. The likely winner of
    the bake-off is a deterministic tree model, so the small amount of MPS
    run-to-run wobble does not move the headline result; it is noted in the
    reproducibility section of transcript.md.
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


TORCH_ACCELERATOR: str = _pick_accelerator()


def set_global_seeds(seed: int = SEED) -> None:
    """Pin every random-number source we can reach.

    Deep-learning results on CPU are still not bit-for-bit reproducible across
    machines, but this removes all the easy sources of run-to-run wobble.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


# --------------------------------------------------------------------------
# Split boundaries (all timestamps are UTC, hourly)
# --------------------------------------------------------------------------

TRAIN_START = pd.Timestamp("2015-01-01 00:00", tz="UTC")
TRAIN_END = pd.Timestamp("2019-09-30 23:00", tz="UTC")
VAL_START = pd.Timestamp("2019-10-01 00:00", tz="UTC")
VAL_END = pd.Timestamp("2019-12-31 23:00", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01 00:00", tz="UTC")
TEST_END = pd.Timestamp("2020-01-07 23:00", tz="UTC")

# The forecast horizon is one week. We tune every model on the first week of
# the validation period (fit on Train, predict this block, score MAPE), which
# mirrors the real test task exactly: one 168-hour forecast made from the end
# of the fitting data.
HORIZON: int = 168
VAL_SELECT_START = VAL_START
VAL_SELECT_END = VAL_START + pd.Timedelta(hours=HORIZON - 1)  # 2019-10-07 23:00


# --------------------------------------------------------------------------
# Colours: each model keeps the same colour in every figure
# --------------------------------------------------------------------------

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
    "patchtst": "PatchTST (TSMixer)",
}

# tab10, one fixed colour per model.
_TAB10 = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
]
MODEL_COLORS: dict[str, str] = {m: c for m, c in zip(MODEL_ORDER, _TAB10)}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_series() -> pd.Series:
    """Load the hourly German load series and verify it is clean.

    Returns a float Series indexed by a UTC hourly DatetimeIndex. Raises if the
    data is not exactly what the prompt promised (50,400 rows, no gaps, no
    NaN) so that a silent data problem cannot slip into the study.
    """
    df = pd.read_csv(DATA_CSV)
    ts = pd.to_datetime(df["utc_timestamp"], utc=True)
    series = pd.Series(df[LOAD_COLUMN].to_numpy(dtype=float), index=ts, name=LOAD_COLUMN)
    series = series.sort_index()

    # Verify the promises. If any fail we want to know loudly, not paper over it.
    if len(series) != 50_400:
        raise ValueError(f"expected 50400 rows, found {len(series)}")
    if series.isna().any():
        raise ValueError("found NaN values in the load series")
    full = pd.date_range(series.index[0], series.index[-1], freq="h", tz="UTC")
    if not series.index.equals(full):
        raise ValueError("timestamp index has gaps or duplicates")
    return series


def slice_inclusive(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Return the closed interval [start, end] of a time-indexed Series."""
    return series.loc[(series.index >= start) & (series.index <= end)]


@dataclass
class DataBundle:
    """Everything a model needs, pre-sliced so no model has to re-derive it."""

    full: pd.Series
    train: pd.Series
    val: pd.Series
    trainval: pd.Series
    test: pd.Series
    val_select: pd.Series  # first 168 h of validation, the tuning target

    @property
    def test_index(self) -> pd.DatetimeIndex:
        return self.test.index


def build_bundle() -> DataBundle:
    """Load once and hand out the standard slices."""
    full = load_series()
    train = slice_inclusive(full, TRAIN_START, TRAIN_END)
    val = slice_inclusive(full, VAL_START, VAL_END)
    trainval = slice_inclusive(full, TRAIN_START, VAL_END)
    test = slice_inclusive(full, TEST_START, TEST_END)
    val_select = slice_inclusive(full, VAL_SELECT_START, VAL_SELECT_END)
    if len(test) != HORIZON:
        raise ValueError(f"test window is {len(test)} hours, expected {HORIZON}")
    return DataBundle(
        full=full,
        train=train,
        val=val,
        trainval=trainval,
        test=test,
        val_select=val_select,
    )


# The five quantile levels every probabilistic model reports. 0.1/0.9 give the
# 80% interval, 0.025/0.975 the 95% interval, 0.5 is the median point forecast.
QUANTILE_LEVELS: list[float] = [0.025, 0.1, 0.5, 0.9, 0.975]


def ensure_real_prophet() -> None:
    """Guarantee that ``import prophet`` finds the installed package.

    Our own model file is called ``prophet.py`` because the study layout asks
    for it, but that name collides with the Prophet library that darts imports
    internally. This loads the real package once (with our own directory taken
    off the import path) and caches it, so darts always gets the library, never
    our file. Safe to call many times.
    """
    import importlib
    import sys

    existing = sys.modules.get("prophet")
    if existing is not None and getattr(existing, "Prophet", None) is not None:
        return

    code_dir = str(Path(__file__).resolve().parent)
    saved = list(sys.path)
    sys.path[:] = [p for p in sys.path if str(Path(p or ".").resolve()) != code_dir]
    sys.modules.pop("prophet", None)
    try:
        real = importlib.import_module("prophet")
    finally:
        sys.path[:] = saved
    if getattr(real, "Prophet", None) is None:
        raise ImportError("could not load the installed prophet package")


def darts_quantiles(pred) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Pull our standard quantile levels out of a stochastic darts forecast.

    Returns (point, quantiles) where the point forecast is the median.
    """

    def q(level: float) -> np.ndarray:
        try:
            ts = pred.quantile_timeseries(level)
        except AttributeError:
            ts = pred.quantile(level)
        return np.asarray(ts.values(copy=False).flatten(), dtype=float)

    quantiles = {lv: q(lv) for lv in QUANTILE_LEVELS}
    point = quantiles[0.5].copy()
    return point, quantiles


def to_darts(series: pd.Series):
    """Turn a UTC-indexed pandas Series into a darts TimeSeries.

    darts does not keep timezone information, so we drop the tz (the data is
    already UTC) and keep the hourly stamps. Values are float32, which is what
    the torch models want.
    """
    from darts import TimeSeries

    naive = series.copy()
    naive.index = naive.index.tz_localize(None)
    return TimeSeries.from_series(naive.astype("float32"))


# --------------------------------------------------------------------------
# German public holidays (allowed: knowable a priori, not exogenous data)
# --------------------------------------------------------------------------


def german_holiday_flags(index: pd.DatetimeIndex) -> np.ndarray:
    """1.0 where the (local German) calendar day is a nationwide holiday.

    We convert each UTC timestamp to Europe/Berlin first, because a public
    holiday is a property of the local calendar day, not the UTC day.
    holidays.Germany() with no state returns the nationwide holidays, which is
    what a national load forecast should care about (Jan 1 is one of them).
    """
    import holidays as holidays_pkg

    local = index.tz_convert("Europe/Berlin")
    years = sorted(set(local.year.tolist()))
    de = holidays_pkg.Germany(years=years)
    return np.array([1.0 if d.date() in de else 0.0 for d in local], dtype=float)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute percentage error, in percent. Load is never zero."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs((actual - forecast) / actual)) * 100.0)


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.mean(np.abs(actual - forecast)))


def pinball_loss(actual: np.ndarray, quantile_forecast: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss for a single quantile level q in (0, 1)."""
    actual = np.asarray(actual, dtype=float)
    quantile_forecast = np.asarray(quantile_forecast, dtype=float)
    diff = actual - quantile_forecast
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals that land inside [lower, upper]."""
    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((actual >= lower) & (actual <= upper)))


# --------------------------------------------------------------------------
# The shape every model returns
# --------------------------------------------------------------------------


@dataclass
class ForecastResult:
    """One model's answer for the 168-hour test week.

    point:       the point forecast, one value per test hour (length 168).
    quantiles:   optional map from quantile level to a length-168 forecast.
                 Models that produce prediction intervals fill in the levels
                 {0.025, 0.1, 0.5, 0.9, 0.975}; the naive baseline leaves it
                 empty.
    """

    name: str
    point: np.ndarray
    runtime_seconds: float
    hyperparameters: dict = field(default_factory=dict)
    val_mape: float | None = None
    quantiles: dict[float, np.ndarray] = field(default_factory=dict)
    # spare slot for model-specific artefacts a figure needs later, e.g. the
    # LightGBM feature importances. Never part of the metrics schema.
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Shared driver for the two deep-learning (torch/darts) models
# --------------------------------------------------------------------------


def deep_forecast(
    name: str,
    bundle: "DataBundle",
    make_model,
    max_epochs: int = 20,
    patience: int = 5,
    num_samples: int = 500,
) -> "ForecastResult":
    """Train, tune, refit and forecast one darts torch model.

    make_model(n_epochs, callbacks) must return a fresh darts model. We:

    1. Scale the load to roughly [0, 1] (torch models train better on that).
    2. Fit on Train with the validation set watched for early stopping; the
       number of epochs where the validation loss stopped improving is our
       chosen "how long to train".
    3. Score that model on the first validation week (for the record).
    4. Refit from scratch on Train+Validation for the chosen number of epochs.
    5. Forecast the test week with sampling, giving prediction intervals.
    """
    import time

    from darts.dataprocessing.transformers import Scaler
    from pytorch_lightning.callbacks import EarlyStopping

    t0 = time.time()
    set_global_seeds()

    train = to_darts(slice_inclusive(bundle.full, TRAIN_START, TRAIN_END))
    trainval = to_darts(slice_inclusive(bundle.full, TRAIN_START, VAL_END))

    sel_scaler = Scaler()
    train_s = sel_scaler.fit_transform(train)
    # Early stopping needs windows of length input+output = 336 h, so the tail
    # three weeks of the validation window are used as the early-stopping watch
    # set. The reported validation MAPE is scored separately on the first
    # validation week (Oct 1-7), a clean 168-hour held-out forecast.
    val_watch = to_darts(slice_inclusive(bundle.val, bundle.val.index[-504], VAL_END))
    val_s = sel_scaler.transform(val_watch)

    early_stop = EarlyStopping(
        monitor="val_loss", patience=patience, mode="min", min_delta=1e-4
    )
    sel_model = make_model(max_epochs, [early_stop])
    sel_model.fit(train_s, val_series=val_s)

    try:
        completed = int(sel_model.trainer.current_epoch)
    except Exception:
        completed = max_epochs
    selected_epochs = max(3, min(max_epochs, completed - patience))

    sel_pred = sel_scaler.inverse_transform(sel_model.predict(HORIZON, series=train_s))
    val_mape = mape(bundle.val_select.to_numpy(), sel_pred.values(copy=False).flatten())

    # --- Refit on Train+Validation for the chosen number of epochs -----------
    set_global_seeds()
    final_scaler = Scaler()
    trainval_s = final_scaler.fit_transform(trainval)
    final_model = make_model(selected_epochs, [])
    final_model.fit(trainval_s)

    pred = final_scaler.inverse_transform(
        final_model.predict(HORIZON, num_samples=num_samples)
    )
    point, quantiles = darts_quantiles(pred)

    runtime = time.time() - t0
    return ForecastResult(
        name=name,
        point=point,
        runtime_seconds=runtime,
        hyperparameters={
            "input_chunk_length": HORIZON,
            "output_chunk_length": HORIZON,
            "max_epochs": max_epochs,
            "selected_epochs": selected_epochs,
            "early_stopping_patience": patience,
            "batch_size": 1024,
            "accelerator": TORCH_ACCELERATOR,
        },
        val_mape=float(val_mape),
        quantiles=quantiles,
    )


# --------------------------------------------------------------------------
# Figure style
# --------------------------------------------------------------------------


def apply_figure_style() -> None:
    """One consistent look for every figure: 300 dpi, sans-serif, readable."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "figure.autolayout": False,
        }
    )


TS_FIGSIZE = (11, 6)
SQUARE_FIGSIZE = (6, 6)
