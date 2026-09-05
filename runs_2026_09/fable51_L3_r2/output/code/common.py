"""Shared pieces of the load-forecasting bake-off.

Everything the six model scripts and the orchestrator have in common lives
here: where the data is, how it is split, how a forecast is scored, what a
forecast result looks like, and the figure style. Keeping this in one place
means every model is judged by exactly the same yardstick.
"""

from __future__ import annotations

import copy
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import holidays
import matplotlib as mpl
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths, dates, constants
# --------------------------------------------------------------------------


def _import_real_prophet_package() -> None:
    """Load the real `prophet` package before `code/prophet.py` can shadow it.

    The study layout puts a module called prophet.py in this directory,
    which is first on sys.path when the scripts run. darts imports the
    Prophet library by that same name, so the library is imported here,
    with this directory temporarily hidden, before anything else happens.
    The orchestrator loads the local prophet.py by file path instead.
    """
    here = Path(__file__).resolve().parent
    saved = list(sys.path)
    sys.path = [p for p in saved if Path(p or ".").resolve() != here]
    try:
        import prophet  # noqa: F401
    finally:
        sys.path = saved


_import_real_prophet_package()

RUN_DIR: Path = Path(__file__).resolve().parents[1]
DATA_PATH: Path = RUN_DIR / "opsd_de_load.csv"
FIG_DIR: Path = RUN_DIR / "figures"
SCRATCH_DIR: Path = RUN_DIR / "code" / "_scratch"

SEED: int = 42
HORIZON: int = 168  # hours in the test week

TRAIN_START = pd.Timestamp("2015-01-01 00:00")
TRAIN_END = pd.Timestamp("2019-09-30 23:00")
VAL_START = pd.Timestamp("2019-10-01 00:00")
VAL_END = pd.Timestamp("2019-12-31 23:00")
TEST_START = pd.Timestamp("2020-01-01 00:00")
TEST_END = pd.Timestamp("2020-01-07 23:00")
EXPECTED_ROWS: int = 50_400

# Quantiles every probabilistic model must produce. 0.025/0.975 bound the
# 95 percent interval, 0.1/0.9 bound the 80 percent interval and are also
# the pinball-loss quantiles, 0.5 is the median.
QUANTILES: tuple[float, ...] = (0.025, 0.1, 0.5, 0.9, 0.975)

MODEL_ORDER: tuple[str, ...] = (
    "naive",
    "sarima",
    "prophet",
    "lightgbm",
    "nbeats",
    "patchtst",
)
MODEL_LABELS: dict[str, str] = {
    "naive": "Seasonal naive (t-168 h)",
    "sarima": "SARIMA",
    "prophet": "Prophet (DE holidays)",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "TSMixer (PatchTST slot)",
}
# One fixed tab10 colour per model, used in every figure. Red is kept free
# for the test-window highlight in the overview figure.
MODEL_COLORS: dict[str, str] = {
    "naive": "#7f7f7f",
    "sarima": "#1f77b4",
    "prophet": "#ff7f0e",
    "lightgbm": "#2ca02c",
    "nbeats": "#9467bd",
    "patchtst": "#17becf",
}
OBSERVED_COLOR: str = "black"
TEST_HIGHLIGHT_COLOR: str = "#d62728"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_series() -> pd.Series:
    """Read the CSV and return the load as an hourly series indexed in UTC.

    The index is timezone-naive but is UTC throughout the study. The
    function checks the promises made about the file (row count, no gaps,
    no missing values) and refuses to continue if any of them is broken,
    so a corrupted input can never be silently patched over.
    """
    df = pd.read_csv(DATA_PATH, parse_dates=["utc_timestamp"])
    ts = pd.DatetimeIndex(df["utc_timestamp"]).tz_convert("UTC").tz_localize(None)
    series = pd.Series(
        df["DE_load_actual_entsoe_transparency"].to_numpy(dtype=float),
        index=ts,
        name="load_mw",
    )
    if len(series) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(series)}")
    if series.isna().any():
        raise ValueError("load series contains missing values")
    steps = np.diff(series.index.to_numpy()).astype("timedelta64[h]").astype(int)
    if not np.all(steps == 1):
        raise ValueError("load series is not a gap-free hourly grid")
    if series.index[0] != TRAIN_START or series.index[-1] != pd.Timestamp("2020-09-30 23:00"):
        raise ValueError("unexpected coverage of the load series")
    series.index.freq = "h"
    return series


@dataclass(frozen=True)
class Splits:
    """The three windows of the study plus the combined refit window."""

    train: pd.Series
    val: pd.Series
    test: pd.Series
    train_val: pd.Series


def split_series(series: pd.Series) -> Splits:
    """Cut the series into train, validation, test and train+validation."""
    train = series.loc[TRAIN_START:TRAIN_END]
    val = series.loc[VAL_START:VAL_END]
    test = series.loc[TEST_START:TEST_END]
    train_val = series.loc[TRAIN_START:VAL_END]
    if len(test) != HORIZON:
        raise ValueError(f"test window has {len(test)} rows, expected {HORIZON}")
    return Splits(train=train, val=val, test=test, train_val=train_val)


def validation_blocks(
    val_index: pd.DatetimeIndex, horizon: int = HORIZON
) -> list[pd.DatetimeIndex]:
    """Cut the validation window into consecutive week-long blocks.

    Hyperparameters are judged the way the test will judge the model: by
    forecasting a full week from a fixed origin, then moving the origin one
    week forward. The validation window holds 13 whole weeks; the last 24
    hours are left over and not used.
    """
    n_blocks = len(val_index) // horizon
    return [val_index[i * horizon : (i + 1) * horizon] for i in range(n_blocks)]


# --------------------------------------------------------------------------
# Calendar features (derived from the timestamp only)
# --------------------------------------------------------------------------


def german_holidays(years: range) -> holidays.HolidayBase:
    """Nationwide German public holidays for the given years."""
    return holidays.country_holidays("DE", years=years)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour, weekday, month, weekend and public-holiday flags in local time.

    The timestamps are UTC, but people switch their lights on by the Berlin
    clock, so the calendar features are computed in Europe/Berlin local
    time. Nothing here needs any data beyond the timestamp itself.
    """
    local = index.tz_localize("UTC").tz_convert("Europe/Berlin")
    hol = german_holidays(range(index[0].year - 1, index[-1].year + 2))
    local_dates = local.date
    is_holiday = np.fromiter((d in hol for d in local_dates), dtype=bool, count=len(local))
    return pd.DataFrame(
        {
            "hour": local.hour,
            "day_of_week": local.dayofweek,
            "month": local.month,
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "is_public_holiday_de": is_holiday.astype(int),
        },
        index=index,
    )


# --------------------------------------------------------------------------
# Forecast container and scoring
# --------------------------------------------------------------------------


@dataclass
class ForecastResult:
    """What every model script returns for the test week.

    point:      the central forecast, one value per test hour
    quantiles:  forecast quantiles keyed by probability (None for the naive
                baseline, which has no notion of uncertainty)
    """

    name: str
    point: pd.Series
    quantiles: dict[float, pd.Series] | None
    runtime_seconds: float
    hyperparameters: dict[str, Any]
    validation: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


def mape(actual: pd.Series, forecast: pd.Series) -> float:
    """Mean absolute percentage error, in percent."""
    a = actual.to_numpy(dtype=float)
    f = forecast.reindex(actual.index).to_numpy(dtype=float)
    return float(np.mean(np.abs((a - f) / a)) * 100.0)


def rmse(actual: pd.Series, forecast: pd.Series) -> float:
    a = actual.to_numpy(dtype=float)
    f = forecast.reindex(actual.index).to_numpy(dtype=float)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def mae(actual: pd.Series, forecast: pd.Series) -> float:
    a = actual.to_numpy(dtype=float)
    f = forecast.reindex(actual.index).to_numpy(dtype=float)
    return float(np.mean(np.abs(a - f)))


def per_day_mape(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    """MAPE for each UTC calendar day in the window."""
    ape = ((actual - forecast.reindex(actual.index)).abs() / actual.abs()) * 100.0
    return ape.groupby(ape.index.normalize()).mean()


def coverage(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Share of observations that fall inside [lower, upper]."""
    a = actual.to_numpy(dtype=float)
    lo = lower.reindex(actual.index).to_numpy(dtype=float)
    hi = upper.reindex(actual.index).to_numpy(dtype=float)
    return float(np.mean((a >= lo) & (a <= hi)))


def pinball_loss(actual: pd.Series, quantile_forecast: pd.Series, tau: float) -> float:
    """Pinball (quantile) loss in MW; the proper score for a quantile forecast."""
    a = actual.to_numpy(dtype=float)
    q = quantile_forecast.reindex(actual.index).to_numpy(dtype=float)
    diff = a - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def block_mape(actual: pd.Series, forecasts: list[pd.Series]) -> float:
    """MAPE over a list of block forecasts, pooled over all their hours."""
    joined = pd.concat(forecasts).sort_index()
    return mape(actual.reindex(joined.index), joined)


def sort_quantiles(quantiles: dict[float, pd.Series]) -> dict[float, pd.Series]:
    """Make sure the quantile curves do not cross by sorting them per hour."""
    qs = sorted(quantiles)
    stacked = np.sort(np.column_stack([quantiles[q].to_numpy(dtype=float) for q in qs]), axis=1)
    index = quantiles[qs[0]].index
    return {q: pd.Series(stacked[:, i], index=index) for i, q in enumerate(qs)}


def score_test(result: ForecastResult, actual: pd.Series) -> dict[str, Any]:
    """All headline metrics for one model on the test week."""
    jan1 = actual.index.normalize() == TEST_START
    out: dict[str, Any] = {
        "name": result.name,
        "mape_test_pct": mape(actual, result.point),
        "rmse_test_mw": rmse(actual, result.point),
        "mae_test_mw": mae(actual, result.point),
        "mape_jan1_pct": mape(actual[jan1], result.point[jan1]),
        "mape_jan2_to_jan7_pct": mape(actual[~jan1], result.point[~jan1]),
        "runtime_seconds": result.runtime_seconds,
        "hyperparameters": result.hyperparameters,
    }
    if result.quantiles is not None:
        q = result.quantiles
        out["coverage_80pct"] = coverage(actual, q[0.1], q[0.9])
        out["coverage_95pct"] = coverage(actual, q[0.025], q[0.975])
        out["pinball_loss_q10"] = pinball_loss(actual, q[0.1], 0.1)
        out["pinball_loss_q50"] = pinball_loss(actual, q[0.5], 0.5)
        out["pinball_loss_q90"] = pinball_loss(actual, q[0.9], 0.9)
    return out


# --------------------------------------------------------------------------
# Reproducibility helpers
# --------------------------------------------------------------------------


def seed_everything(seed: int = SEED) -> None:
    """Seed every random number generator the study touches."""
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)


class Stopwatch:
    """Context manager that records elapsed wall-clock seconds."""

    def __enter__(self) -> "Stopwatch":
        self._t0 = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self._t0


def log(msg: str) -> None:
    """Timestamped progress line on stdout."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Deep-learning helpers shared by N-BEATS and TSMixer (darts + Lightning)
# --------------------------------------------------------------------------


def make_quantile_likelihood() -> Any:
    """Quantile-regression head whose validation metrics use the median.

    darts computes validation metrics on a random draw from the predicted
    quantiles, which would make early stopping noisy. This subclass reports
    the median instead, so the logged validation MAPE is deterministic.
    Sampling is never used for forecasting in this study; the quantile
    curves are read out directly.
    """
    import torch
    from darts.utils.likelihood_models import QuantileRegression

    class MedianQuantileRegression(QuantileRegression):  # type: ignore[misc]
        def sample(self, model_output: torch.Tensor) -> torch.Tensor:
            return model_output[..., self._median_idx]

    return MedianQuantileRegression(quantiles=list(QUANTILES))


def make_best_weights_callback(monitor: str) -> Any:
    """Lightning callback that remembers the best epoch's weights in memory."""
    from pytorch_lightning.callbacks import Callback

    class BestWeights(Callback):
        def __init__(self) -> None:
            self.best_value: float = float("inf")
            self.best_epoch: int = 0
            self.best_state: dict[str, Any] | None = None
            self.history: list[float] = []

        def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
            if trainer.sanity_checking:
                return
            value = float(trainer.callback_metrics[monitor])
            self.history.append(value)
            if value < self.best_value:
                self.best_value = value
                self.best_epoch = trainer.current_epoch + 1
                self.best_state = copy.deepcopy(pl_module.state_dict())

    return BestWeights()


def fit_with_early_stopping(
    build: Callable[[dict[str, Any]], Any],
    train: pd.Series,
    val: pd.Series,
    max_epochs: int,
    patience: int,
) -> tuple[Any, dict[str, Any]]:
    """Train a darts torch model on `train`, stopping early on validation MAPE.

    `build(trainer_kwargs)` must return a fresh, unfitted darts model with
    the given Lightning trainer settings. The validation series is prefixed
    with the last input window of the training series so that the first
    validation forecast starts exactly at the validation boundary. After
    training, the weights of the best epoch are restored.
    """
    from pytorch_lightning.callbacks import EarlyStopping

    monitor = "val_MeanAbsolutePercentageError"
    best = make_best_weights_callback(monitor)
    stopper = EarlyStopping(monitor=monitor, patience=patience, mode="min")
    model = build({"callbacks": [stopper, best], "max_epochs": max_epochs})
    train_ts = to_darts(train)
    val_ts = to_darts(pd.concat([train.iloc[-model.input_chunk_length :], val]))
    model.fit(train_ts, val_series=val_ts, verbose=False)
    if best.best_state is not None:
        model.model.load_state_dict(best.best_state)
    info = {
        "best_epoch": best.best_epoch,
        "epochs_run": len(best.history),
        "val_mape_history_pct": [round(v * 100.0, 3) for v in best.history],
    }
    return model, info


def torch_trainer_kwargs() -> dict[str, Any]:
    """Lightning trainer settings shared by the two neural models.

    Apple's GPU (MPS) is used when present because it roughly halves the
    training time; set FORCE_CPU=1 in the environment to train on the CPU
    instead, which is slower but bit-for-bit reproducible.
    """
    import os

    import torch

    use_mps = torch.backends.mps.is_available() and os.environ.get("FORCE_CPU", "0") != "1"
    return {
        "accelerator": "mps" if use_mps else "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
        "enable_checkpointing": False,
    }


def cached_sweep(key: str, compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one hyperparameter candidate, or reuse its saved validation result.

    Each neural candidate takes minutes to train, so its validation score
    and best epoch are written to code/_scratch/results/sweep_<key>.json.
    A rerun after an interruption picks up where it stopped; delete the
    file to force retraining.
    """
    import json

    path = SCRATCH_DIR / "results" / f"sweep_{key}.json"
    if path.exists():
        log(f"sweep {key}: reusing cached validation result")
        return dict(json.loads(path.read_text()))
    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


def to_darts(series: pd.Series) -> Any:
    """Wrap a pandas series as a darts TimeSeries (float32, hourly)."""
    from darts import TimeSeries

    return TimeSeries.from_series(series.astype(np.float32), freq="h")


def predict_quantiles(
    model: Any, history: pd.Series, n: int, scale: float
) -> dict[float, pd.Series]:
    """Read the model's quantile curves for the `n` hours after `history`."""
    ts = model.predict(
        n=n, series=to_darts(history), predict_likelihood_parameters=True, verbose=False
    )
    frame = ts.to_dataframe()
    out: dict[float, pd.Series] = {}
    for q in QUANTILES:
        col = [c for c in frame.columns if c.endswith(f"_q{q:.3f}".rstrip("0"))]
        if len(col) != 1:
            col = [c for c in frame.columns if abs(float(c.split("_q")[-1]) - q) < 1e-9]
        out[q] = pd.Series(frame[col[0]].to_numpy(dtype=float) * scale, index=frame.index)
    return sort_quantiles(out)


# --------------------------------------------------------------------------
# Figure style
# --------------------------------------------------------------------------

TS_FIGSIZE: tuple[float, float] = (11.0, 6.0)
SQ_FIGSIZE: tuple[float, float] = (6.0, 6.0)


def apply_figure_style() -> None:
    """Matplotlib defaults shared by every figure: 300 dpi, sans-serif, light grid."""
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
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
            "legend.fontsize": 9,
            "lines.linewidth": 1.4,
            "savefig.bbox": "tight",
        }
    )
