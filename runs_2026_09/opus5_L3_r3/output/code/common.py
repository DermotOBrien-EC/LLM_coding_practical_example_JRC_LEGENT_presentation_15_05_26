"""Shared machinery for the German load forecasting bake-off.

Everything that more than one model needs lives here: reading the CSV,
cutting it into train / validation / test, building calendar features,
scoring a forecast, and the colour scheme the figures use so that a model
keeps the same colour in every plot.

The audience for this code is an analyst who wants to check the numbers,
not an ML researcher, so the comments explain *why* rather than *what*.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import holidays
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Import-order repair (runs before anything else imports darts)
# ----------------------------------------------------------------------


def _ensure_real_prophet() -> None:
    """Stop `code/prophet.py` from masquerading as the installed `prophet` package.

    The study's file layout asks for a module called `prophet.py` sitting next
    to this one, and `code/` is first on the import path. Python therefore
    hands anything that writes `import prophet`, darts included, our little
    module instead of the real forecasting library, and darts then fails
    looking for a `Prophet` class that is not there. This finds the genuine
    package on the rest of the path and installs it under the name `prophet`
    before darts ever asks. Called once, at import time, by every module in
    this directory.
    """
    import importlib.machinery
    import importlib.util
    import sys

    existing = sys.modules.get("prophet")
    if existing is not None and hasattr(existing, "Prophet"):
        return

    here = Path(__file__).resolve().parent
    elsewhere = [entry for entry in sys.path if entry and Path(entry).resolve() != here]
    spec = importlib.machinery.PathFinder.find_spec("prophet", elsewhere)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["prophet"] = module
    spec.loader.exec_module(module)


_ensure_real_prophet()


# ----------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------

ROOT: Path = Path(__file__).resolve().parent.parent
CSV_PATH: Path = ROOT / "opsd_de_load.csv"
FIGURE_DIR: Path = ROOT / "figures"

LOAD_COL: str = "DE_load_actual_entsoe_transparency"

# One global seed. Every model that has a random component is handed this
# number so that a re-run reproduces the same table.
SEED: int = 42

# The three windows. All timestamps are UTC, matching the raw file.
TRAIN_START = pd.Timestamp("2015-01-01 00:00", tz="UTC")
TRAIN_END = pd.Timestamp("2019-09-30 23:00", tz="UTC")
VAL_START = pd.Timestamp("2019-10-01 00:00", tz="UTC")
VAL_END = pd.Timestamp("2019-12-31 23:00", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01 00:00", tz="UTC")
TEST_END = pd.Timestamp("2020-01-07 23:00", tz="UTC")

HORIZON: int = 168  # one week of hourly steps: the forecast horizon everywhere

# German electricity demand follows the local clock (CET/CEST), not UTC, so
# calendar features are derived from the Berlin-local timestamp even though
# the series itself stays on a clean UTC hourly grid.
LOCAL_TZ: str = "Europe/Berlin"

# ----------------------------------------------------------------------
# Model identity: names, labels, colours
# ----------------------------------------------------------------------

MODEL_ORDER: tuple[str, ...] = (
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
    "patchtst": "TSMixer (PatchTST slot)",
}

# Fixed slice of matplotlib's tab10 so a model is the same colour in every
# figure. Picked once, never recomputed from a sort order.
MODEL_COLORS: dict[str, str] = {
    "naive": "#7f7f7f",  # tab10 grey  - the baseline
    "sarima": "#1f77b4",  # tab10 blue
    "prophet": "#ff7f0e",  # tab10 orange
    "lightgbm": "#2ca02c",  # tab10 green
    "nbeats": "#d62728",  # tab10 red
    "patchtst": "#9467bd",  # tab10 purple
}


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------


def load_series(csv_path: Path | None = None) -> pd.Series:
    """Read the load series and prove it is the clean series we expect.

    Returns an hourly, timezone-aware (UTC) float series in megawatts. The
    assertions are deliberately loud: the study claims no imputation was
    needed, and this is where that claim is checked rather than assumed.
    """
    path = CSV_PATH if csv_path is None else csv_path
    frame = pd.read_csv(path, parse_dates=["utc_timestamp"])
    series = frame.set_index("utc_timestamp")[LOAD_COL].astype("float64")
    series.index = pd.DatetimeIndex(series.index).tz_convert("UTC")
    series = series.sort_index()

    expected = pd.date_range(series.index[0], series.index[-1], freq="h", tz="UTC")
    if len(series) != len(expected) or not series.index.equals(expected):
        raise ValueError("load series is not a gapless hourly grid")
    if series.isna().any():
        raise ValueError("load series contains missing values")
    if len(series) != 50_400:
        raise ValueError(f"expected 50400 rows, found {len(series)}")

    series.name = "load_mw"
    series.index.name = "utc_timestamp"
    return series.asfreq("h")


def split_series(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Cut the series into train, validation and test as defined in the study."""
    train = series.loc[TRAIN_START:TRAIN_END]
    val = series.loc[VAL_START:VAL_END]
    test = series.loc[TEST_START:TEST_END]
    if len(test) != HORIZON:
        raise ValueError(f"test window should be {HORIZON} hours, found {len(test)}")
    return train, val, test


def train_plus_val(series: pd.Series) -> pd.Series:
    """Everything strictly before the test window: the final refit sample."""
    return series.loc[TRAIN_START:VAL_END]


def validation_origins(n_blocks: int | None = None) -> list[pd.Timestamp]:
    """Forecast-origin timestamps used for hyperparameter scoring.

    The validation window is 2208 hours, which holds thirteen whole
    non-overlapping weeks. Each origin is the first hour of a week that the
    candidate model must predict without seeing it. Scoring a model the same
    way it will be used on the test window (168 hours ahead, from a cold
    start) is the point: a model tuned on one-step-ahead errors would be
    tuned for a different job.
    """
    origins = [VAL_START + pd.Timedelta(hours=168 * i) for i in range(13)]
    origins = [o for o in origins if o + pd.Timedelta(hours=HORIZON - 1) <= VAL_END]
    if n_blocks is not None:
        # Evenly spaced subset, keeping the first and last block.
        idx = np.unique(np.linspace(0, len(origins) - 1, n_blocks).round().astype(int))
        origins = [origins[i] for i in idx]
    return origins


# ----------------------------------------------------------------------
# Calendar features
# ----------------------------------------------------------------------

_DE_HOLIDAYS = holidays.Germany(years=range(2013, 2023))


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Features knowable in advance from the clock and the calendar alone.

    No external dataset is consulted: German federal public holidays come
    from the `holidays` package, which is a lookup table of dates, not a
    measurement of anything.
    """
    local = index.tz_convert(LOCAL_TZ)
    local_dates = local.date
    frame = pd.DataFrame(index=index)
    frame["hour"] = local.hour.astype("int16")
    frame["day_of_week"] = local.dayofweek.astype("int16")
    frame["month"] = local.month.astype("int16")
    frame["is_weekend"] = (local.dayofweek >= 5).astype("int8")
    frame["is_public_holiday_de"] = np.array(
        [1 if d in _DE_HOLIDAYS else 0 for d in local_dates], dtype="int8"
    )
    return frame


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------


def mape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean absolute percentage error, in percent."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Root mean square error, in the units of the series (MW)."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Mean absolute error, in the units of the series (MW)."""
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    return float(np.mean(np.abs(y_true - y_pred)))


def pinball_loss(
    y_true: np.ndarray | pd.Series, y_quantile: np.ndarray | pd.Series, tau: float
) -> float:
    """Pinball (quantile) loss in MW.

    It penalises being above the actual and being below the actual by
    different amounts, so that a forecast claiming to be the 10th percentile
    is rewarded only if roughly 10 percent of actuals fall below it.
    """
    y_true = np.asarray(y_true, dtype="float64")
    y_quantile = np.asarray(y_quantile, dtype="float64")
    diff = y_true - y_quantile
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def interval_coverage(
    y_true: np.ndarray | pd.Series,
    lower: np.ndarray | pd.Series,
    upper: np.ndarray | pd.Series,
) -> float:
    """Fraction of actuals that fall inside the interval (0 to 1)."""
    y_true = np.asarray(y_true, dtype="float64")
    lower = np.asarray(lower, dtype="float64")
    upper = np.asarray(upper, dtype="float64")
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def evaluate_point_forecast(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    """Headline test metrics plus the holiday / working-day split.

    New Year's Day is a German public holiday and it is the first day of the
    test window, so splitting it out shows immediately which models know
    about holidays and which merely know about weekdays.
    """
    forecast = forecast.reindex(actual.index)
    if forecast.isna().any():
        raise ValueError("forecast does not cover the whole evaluation window")

    jan1 = actual.index.tz_convert("UTC").normalize() == TEST_START
    rest = ~jan1

    return {
        "mape_test_pct": mape(actual, forecast),
        "rmse_test_mw": rmse(actual, forecast),
        "mae_test_mw": mae(actual, forecast),
        "mape_jan1_pct": mape(actual[jan1], forecast[jan1]),
        "mape_jan2_to_jan7_pct": mape(actual[rest], forecast[rest]),
    }


def per_day_mape(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    """MAPE for each of the seven test days, indexed by UTC date."""
    forecast = forecast.reindex(actual.index)
    days = actual.index.tz_convert("UTC").normalize()
    out: dict[pd.Timestamp, float] = {}
    for day in pd.unique(days):
        mask = days == day
        out[pd.Timestamp(day)] = mape(actual[mask], forecast[mask])
    return pd.Series(out).sort_index()


# ----------------------------------------------------------------------
# The object every model module returns
# ----------------------------------------------------------------------


@dataclass
class ForecastResult:
    """One model's answer for the 168-hour test window.

    `quantiles` maps a probability level (e.g. 0.1) to a forecast series.
    The naive baseline leaves it empty because it produces no intervals.
    """

    name: str
    point: pd.Series
    quantiles: dict[float, pd.Series] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    selection_log: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def has_intervals(self) -> bool:
        needed = {0.025, 0.1, 0.9, 0.975}
        return needed.issubset(set(self.quantiles))


# ----------------------------------------------------------------------
# Shared helper for the two deep-learning models
# ----------------------------------------------------------------------

QUANTILE_LEVELS: tuple[float, ...] = (0.025, 0.1, 0.5, 0.9, 0.975)


def to_darts(series: pd.Series) -> Any:
    """Convert a UTC pandas series into the tz-naive TimeSeries darts wants.

    darts refuses timezone-aware indexes. Dropping the tz label is safe here
    because the grid is already regular UTC hours; nothing about the values
    changes.
    """
    from darts import TimeSeries

    naive = series.copy()
    naive.index = naive.index.tz_convert("UTC").tz_localize(None)
    return TimeSeries.from_series(naive.astype("float32"), freq="h")


def from_darts_quantiles(
    prediction: Any, index: pd.DatetimeIndex, levels: Sequence[float] = QUANTILE_LEVELS
) -> dict[float, pd.Series]:
    """Pull the requested quantiles out of a darts probabilistic prediction."""
    samples = prediction.all_values(copy=False)[:, 0, :]  # (time, samples)
    out: dict[float, pd.Series] = {}
    for level in levels:
        out[float(level)] = pd.Series(
            np.quantile(samples, level, axis=1).astype("float64"), index=index
        )
    return out


class EpochLossRecorder:
    """Lightning callback that remembers the validation loss of each epoch.

    Used to answer one question after training: which epoch was actually the
    best one, so that the refit on train+validation can stop there instead of
    guessing.
    """

    def __init__(self) -> None:
        self.val_losses: list[float] = []

    def __call__(self) -> Any:
        import pytorch_lightning as pl

        recorder = self

        class _Recorder(pl.Callback):
            def on_validation_epoch_end(self, trainer: Any, module: Any) -> None:
                if trainer.sanity_checking:
                    return
                value = trainer.callback_metrics.get("val_loss")
                if value is not None:
                    recorder.val_losses.append(float(value))

        return _Recorder()

    def best_epoch(self) -> int:
        """1-based index of the epoch with the lowest validation loss."""
        if not self.val_losses:
            return 1
        return int(np.argmin(self.val_losses)) + 1


def trainer_kwargs(callbacks: Iterable[Any] | None = None) -> dict[str, Any]:
    """Quiet, CPU-pinned Lightning settings shared by both deep models.

    CPU rather than the Mac GPU on purpose: the run is small enough that
    speed is not the constraint, and CPU keeps the result bit-reproducible
    across machines.
    """
    return {
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
        "callbacks": list(callbacks) if callbacks else [],
    }


# ----------------------------------------------------------------------
# Figure style
# ----------------------------------------------------------------------

TIMESERIES_FIGSIZE: tuple[float, float] = (11.0, 6.0)
SQUARE_FIGSIZE: tuple[float, float] = (6.0, 6.0)
DPI: int = 300


def setup_matplotlib() -> None:
    """One place for the look of every figure in the study."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.4,
        }
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def select_and_refit_torch(
    name: str,
    factory: Any,
    series: pd.Series,
    candidates: Sequence[dict[str, Any]],
    max_epochs: int = 30,
    patience: int = 5,
    n_samples: int = 500,
) -> "ForecastResult":
    """Train / select / refit loop shared by N-BEATS and the TSMixer entry.

    `factory(params, n_epochs, callbacks)` must return a fresh, unfitted darts
    model. The routine then does three things for each candidate:

    1. Fits it on the training years only, watching the validation window and
       stopping early when the validation loss stops improving. The epoch with
       the lowest validation loss is remembered.
    2. Scores it by forecasting each of the thirteen validation weeks from a
       cold start, exactly the way it will be asked to forecast the test week.
    3. After the best candidate is known, refits it from scratch on training
       plus validation data for the number of epochs that was best in step 1,
       then produces the test forecast.

    Scaling matters here: neural networks train far better on numbers near
    zero than on tens of thousands of megawatts, so the series is min-max
    scaled and the forecast is scaled back afterwards.
    """
    import time

    import torch
    from darts.dataprocessing.transformers import Scaler
    from pytorch_lightning.callbacks import EarlyStopping

    start = time.perf_counter()

    train = series.loc[TRAIN_START:TRAIN_END]
    val = series.loc[VAL_START:VAL_END]
    fit_all = series.loc[TRAIN_START:VAL_END]
    test_index = series.loc[TEST_START:TEST_END].index

    scaler_stage1 = Scaler()
    train_scaled = scaler_stage1.fit_transform(to_darts(train))
    val_scaled = scaler_stage1.transform(to_darts(val))

    origins = validation_origins()
    log: list[dict[str, Any]] = []

    for params in candidates:
        torch.manual_seed(SEED)
        t0 = time.perf_counter()
        recorder = EpochLossRecorder()
        stopper = EarlyStopping(monitor="val_loss", patience=patience, mode="min")
        model = factory(params, max_epochs, [recorder(), stopper])
        model.fit(train_scaled, val_series=val_scaled, verbose=False)
        best_epoch = recorder.best_epoch()

        scores: list[float] = []
        for origin in origins:
            history_series = series.loc[TRAIN_START : origin - pd.Timedelta(hours=1)]
            history = scaler_stage1.transform(to_darts(history_series))
            pred = model.predict(HORIZON, series=history, num_samples=200, verbose=False)
            unscaled = scaler_stage1.inverse_transform(pred)
            median = np.median(unscaled.all_values(copy=False)[:, 0, :], axis=1)
            actual = series.loc[origin : origin + pd.Timedelta(hours=HORIZON - 1)]
            scores.append(mape(actual.to_numpy(), median))

        log.append(
            {
                **params,
                "best_epoch": best_epoch,
                "epochs_run": len(recorder.val_losses),
                "val_mape_pct": float(np.mean(scores)),
                "fit_and_score_seconds": round(time.perf_counter() - t0, 1),
            }
        )

    best = min(log, key=lambda row: row["val_mape_pct"])
    chosen = {key: best[key] for key in candidates[0]}
    best_epoch = int(best["best_epoch"])

    # Refit from scratch on train + validation for the epoch count that was
    # best on validation. No early stopping here: there is nothing left to
    # watch that is not also being trained on.
    torch.manual_seed(SEED)
    scaler_final = Scaler()
    all_scaled = scaler_final.fit_transform(to_darts(fit_all))
    final_model = factory(chosen, best_epoch, [])
    final_model.fit(all_scaled, verbose=False)

    prediction = final_model.predict(HORIZON, num_samples=n_samples, verbose=False)
    prediction = scaler_final.inverse_transform(prediction)
    quantiles = from_darts_quantiles(prediction, test_index, QUANTILE_LEVELS)
    point = quantiles[0.5].copy()

    return ForecastResult(
        name=name,
        point=point,
        quantiles=quantiles,
        hyperparameters={
            **chosen,
            "input_chunk_length": HORIZON,
            "output_chunk_length": HORIZON,
            "epochs_used_for_final_fit": best_epoch,
            "max_epochs": max_epochs,
            "early_stopping_patience": patience,
            "likelihood": "QuantileRegression(0.025, 0.1, 0.5, 0.9, 0.975)",
            "scaler": "min-max on the training sample",
            "selection_metric": "mean MAPE over 13 held-out validation weeks",
            "interval_method": f"quantiles of {n_samples} predictive draws",
        },
        runtime_seconds=time.perf_counter() - start,
        selection_log=log,
    )
