"""Shared plumbing for the German load forecasting bake-off.

Everything that more than one model needs lives here: loading the data,
cutting it into the three time slices, building calendar features,
scoring a forecast, and the plotting style so every figure looks the same.

The idea is that each model file stays short and only contains what is
special about that model.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import holidays
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------

ROOT: Path = Path(__file__).resolve().parent.parent
DATA_FILE: Path = ROOT / "opsd_de_load.csv"
FIGURE_DIR: Path = ROOT / "figures"
ARTIFACT_DIR: Path = ROOT / "artifacts"
LOAD_COLUMN: str = "DE_load_actual_entsoe_transparency"

# ---------------------------------------------------------------------------
# The three time slices. All timestamps are UTC, matching the raw file.
# ---------------------------------------------------------------------------

TRAIN_START: pd.Timestamp = pd.Timestamp("2015-01-01 00:00", tz="UTC")
TRAIN_END: pd.Timestamp = pd.Timestamp("2019-09-30 23:00", tz="UTC")
VAL_START: pd.Timestamp = pd.Timestamp("2019-10-01 00:00", tz="UTC")
VAL_END: pd.Timestamp = pd.Timestamp("2019-12-31 23:00", tz="UTC")
TEST_START: pd.Timestamp = pd.Timestamp("2020-01-01 00:00", tz="UTC")
TEST_END: pd.Timestamp = pd.Timestamp("2020-01-07 23:00", tz="UTC")

HORIZON: int = 168  # one week of hourly steps; the length of the test window
SEED: int = 42

# Germany is UTC+1 in winter and UTC+2 in summer. Human activity follows the
# clock on the wall, not UTC, so calendar features are derived from Berlin
# local time. Without this, the daily load shape would appear to jump by an
# hour twice a year and an "hour of day" feature would be blurred.
LOCAL_TZ: str = "Europe/Berlin"

# ---------------------------------------------------------------------------
# Model identities: one colour per model, used in every figure.
# ---------------------------------------------------------------------------

MODEL_NAMES: list[str] = ["naive", "sarima", "prophet", "lightgbm", "nbeats", "patchtst"]

MODEL_LABELS: dict[str, str] = {
    "naive": "Seasonal naive",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "TSMixer",
}

_TAB10 = plt.get_cmap("tab10").colors
MODEL_COLORS: dict[str, tuple[float, float, float]] = {
    "naive": _TAB10[7],  # grey
    "sarima": _TAB10[0],  # blue
    "prophet": _TAB10[2],  # green
    "lightgbm": _TAB10[1],  # orange
    "nbeats": _TAB10[4],  # purple
    "patchtst": _TAB10[3],  # red
}

OBSERVED_COLOR: str = "black"

# Quantiles every probabilistic model is asked to produce. The outer pair
# gives the 95% interval, the middle pair the 80% interval.
QUANTILES: tuple[float, ...] = (0.025, 0.1, 0.5, 0.9, 0.975)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_load_series() -> pd.Series:
    """Read the load series and check it really is hourly, complete and clean.

    We assert rather than repair. If the file ever stops being clean the run
    should stop loudly instead of quietly imputing something.
    """
    frame = pd.read_csv(DATA_FILE, parse_dates=["utc_timestamp"])
    series = frame.set_index("utc_timestamp")[LOAD_COLUMN].astype(float)
    series.index = pd.DatetimeIndex(series.index).tz_convert("UTC")
    series = series.sort_index()

    expected = pd.date_range(series.index[0], series.index[-1], freq="h", tz="UTC")
    if len(series) != len(expected) or not series.index.equals(expected):
        raise ValueError("The load series is not a complete hourly grid.")
    if series.isna().any():
        raise ValueError("The load series contains missing values.")
    if series.index.has_duplicates:
        raise ValueError("The load series contains duplicated timestamps.")

    series.index.freq = "h"
    series.name = "load_mw"
    return series


def data_integrity_report(series: pd.Series) -> dict[str, Any]:
    """A few facts about the raw data, quoted in the write-up."""
    return {
        "n_rows": int(len(series)),
        "start_utc": str(series.index[0]),
        "end_utc": str(series.index[-1]),
        "n_missing": int(series.isna().sum()),
        "n_gaps": 0,
        "n_duplicate_timestamps": int(series.index.duplicated().sum()),
        "min_mw": float(series.min()),
        "max_mw": float(series.max()),
        "mean_mw": float(series.mean()),
    }


def split_series(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Cut the series into train, validation and test."""
    train = series.loc[TRAIN_START:TRAIN_END]
    val = series.loc[VAL_START:VAL_END]
    test = series.loc[TEST_START:TEST_END]
    return train, val, test


def train_plus_val(series: pd.Series) -> pd.Series:
    """Everything strictly before the test window: the final fitting set."""
    return series.loc[TRAIN_START:VAL_END]


def to_naive_index(series: pd.Series) -> pd.Series:
    """Drop the timezone label (keeping UTC clock time).

    darts and Prophet are happier with a timezone-naive index, and since the
    whole series is UTC nothing is lost.
    """
    out = series.copy()
    out.index = out.index.tz_localize(None)
    out.index.freq = "h"
    return out


# ---------------------------------------------------------------------------
# Calendar features. Nothing here needs any dataset other than the clock.
# ---------------------------------------------------------------------------


def german_holiday_dates(years: Iterator[int] | list[int]) -> set[pd.Timestamp]:
    """German nationwide public holidays for the given years."""
    de = holidays.Germany(years=list(years))
    return {pd.Timestamp(d) for d in de.keys()}


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Hour, weekday, month, weekend flag and German public holiday flag.

    All of these are derived from Berlin wall-clock time (see LOCAL_TZ).
    """
    local = index.tz_convert(LOCAL_TZ) if index.tz is not None else index.tz_localize("UTC").tz_convert(LOCAL_TZ)
    local_dates = pd.DatetimeIndex(local.date)
    years = np.asarray(local.year)
    hol = german_holiday_dates(range(int(years.min()), int(years.max()) + 1))
    holiday_flag = np.fromiter((d in hol for d in local_dates), dtype=bool, count=len(local_dates))

    dow = np.asarray(local.dayofweek)
    return pd.DataFrame(
        {
            "hour": np.asarray(local.hour),
            "day_of_week": dow,
            "month": np.asarray(local.month),
            "is_weekend": (dow >= 5).astype(int),
            "is_public_holiday_de": holiday_flag.astype(int),
        },
        index=index,
    )


def fourier_terms(index: pd.DatetimeIndex, period_hours: float, n_harmonics: int, prefix: str) -> pd.DataFrame:
    """Smooth sine/cosine waves of a given period.

    Used to hand SARIMAX a weekly rhythm it cannot reach with a daily
    seasonal order alone.
    """
    hours = (index.asi8 // 3_600_000_000_000).astype(float)
    out: dict[str, np.ndarray] = {}
    for k in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * k * hours / period_hours
        out[f"{prefix}_sin{k}"] = np.sin(angle)
        out[f"{prefix}_cos{k}"] = np.cos(angle)
    return pd.DataFrame(out, index=index)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, in percent."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs((actual - predicted) / actual)) * 100.0)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error, same unit as the data (MW)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error, same unit as the data (MW)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def pinball_loss(actual: np.ndarray, quantile_forecast: np.ndarray, quantile: float) -> float:
    """Pinball (quantile) loss in MW.

    It penalises being above the actual differently from being below, with
    the asymmetry set by the quantile. A perfect median forecast gives half
    the mean absolute error at q=0.5.
    """
    actual = np.asarray(actual, dtype=float)
    quantile_forecast = np.asarray(quantile_forecast, dtype=float)
    diff = actual - quantile_forecast
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of observations that fell inside the interval."""
    actual = np.asarray(actual, dtype=float)
    return float(np.mean((actual >= np.asarray(lower)) & (actual <= np.asarray(upper))))


def utc_day_mask(index: pd.DatetimeIndex, day: str) -> np.ndarray:
    """Boolean mask selecting one UTC calendar day.

    The test window is seven whole UTC days, so slicing by UTC date gives
    seven blocks of exactly 24 hours. (Berlin local dates would split the
    window unevenly.)
    """
    target = pd.Timestamp(day).date()
    return np.array([ts.date() == target for ts in index])


def evaluate_point_forecast(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Headline errors plus the holiday / working-day split."""
    if not actual.index.equals(predicted.index):
        raise ValueError("Actual and predicted series are not aligned.")
    a = actual.to_numpy(dtype=float)
    p = predicted.to_numpy(dtype=float)

    jan1 = utc_day_mask(actual.index, "2020-01-01")
    rest = ~jan1

    return {
        "mape_test_pct": mape(a, p),
        "rmse_test_mw": rmse(a, p),
        "mae_test_mw": mae(a, p),
        "mape_jan1_pct": mape(a[jan1], p[jan1]),
        "mape_jan2_to_jan7_pct": mape(a[rest], p[rest]),
    }


def per_day_mape(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """MAPE for each of the seven UTC days of the test window."""
    out: dict[str, float] = {}
    for day in pd.date_range(TEST_START, TEST_END, freq="D"):
        key = day.strftime("%Y-%m-%d")
        mask = utc_day_mask(actual.index, key)
        out[key] = mape(actual.to_numpy()[mask], predicted.to_numpy()[mask])
    return out


# ---------------------------------------------------------------------------
# Rolling-origin evaluation on the validation window
# ---------------------------------------------------------------------------


def validation_origins(horizon: int = HORIZON) -> list[pd.Timestamp]:
    """Start timestamps of the back-to-back forecast blocks in validation.

    Hyperparameters are chosen by forecasting the validation quarter in
    168-hour chunks, which is exactly the job the model has to do on the
    test week. Scoring one long 2208-hour forecast instead would measure a
    different (harder) task.
    """
    n_blocks = int(len(pd.date_range(VAL_START, VAL_END, freq="h")) // horizon)
    return [VAL_START + pd.Timedelta(hours=horizon * i) for i in range(n_blocks)]


# ---------------------------------------------------------------------------
# What every model hands back
# ---------------------------------------------------------------------------


@dataclass
class ForecastResult:
    """One model's contribution to the bake-off."""

    name: str
    point_forecast: pd.Series
    runtime_seconds: float
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    quantile_forecast: pd.DataFrame | None = None  # columns are the QUANTILES
    validation_mape_pct: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def save(self) -> Path:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT_DIR / f"{self.name}.pkl"
        with path.open("wb") as handle:
            pickle.dump(self, handle)
        return path

    @staticmethod
    def load(name: str) -> "ForecastResult":
        with (ARTIFACT_DIR / f"{name}.pkl").open("rb") as handle:
            return pickle.load(handle)

    @staticmethod
    def exists(name: str) -> bool:
        return (ARTIFACT_DIR / f"{name}.pkl").exists()


def quantile_frame(index: pd.DatetimeIndex, samples: np.ndarray) -> pd.DataFrame:
    """Turn a (time, sample) array of simulated futures into quantile columns.

    Quantiles are sorted afterwards so that a noisy sample cloud can never
    produce an upper bound below a lower bound.
    """
    values = np.quantile(samples, QUANTILES, axis=1).T
    values = np.sort(values, axis=1)
    return pd.DataFrame(values, index=index, columns=[str(q) for q in QUANTILES])


def probabilistic_scores(actual: pd.Series, quantiles: pd.DataFrame) -> dict[str, float]:
    """Interval coverage and pinball losses for a probabilistic forecast."""
    a = actual.to_numpy(dtype=float)
    return {
        "coverage_80pct": interval_coverage(a, quantiles["0.1"], quantiles["0.9"]),
        "coverage_95pct": interval_coverage(a, quantiles["0.025"], quantiles["0.975"]),
        "pinball_loss_q10": pinball_loss(a, quantiles["0.1"].to_numpy(), 0.1),
        "pinball_loss_q50": pinball_loss(a, quantiles["0.5"].to_numpy(), 0.5),
        "pinball_loss_q90": pinball_loss(a, quantiles["0.9"].to_numpy(), 0.9),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def apply_figure_style() -> None:
    """One look for every figure in the study."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.6,
        }
    )


def set_seeds(seed: int = SEED) -> None:
    """Pin every random number generator we can reach."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


# ---------------------------------------------------------------------------
# Helpers for the two deep-learning entries (darts + PyTorch)
# ---------------------------------------------------------------------------


def to_darts_series(series: pd.Series) -> Any:
    """Convert a pandas load series into a float32 darts TimeSeries.

    float32 keeps PyTorch happy and halves the memory traffic; the load
    values are around 5e4 MW, far inside float32's precision.
    """
    from darts import TimeSeries

    return TimeSeries.from_series(to_naive_index(series)).astype(np.float32)


def torch_trainer_kwargs(patience: int | None = None) -> dict[str, Any]:
    """PyTorch Lightning settings shared by both neural models.

    Training runs on the CPU: on this machine an Apple-silicon GPU run was
    measured at 28 s per epoch against 23 s on the CPU, so there is nothing
    to gain and CPU runs are easier to reproduce exactly.
    """
    kwargs: dict[str, Any] = {
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "logger": False,
    }
    if patience is not None:
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping

        kwargs["callbacks"] = [
            EarlyStopping(monitor="val_loss", patience=patience, min_delta=1e-4, mode="min")
        ]
    return kwargs


def rolling_origin_mape_torch(model: Any, scaled_full: Any, scaler: Any, actual: pd.Series) -> float:
    """Score a trained darts model on the validation window, block by block.

    For each 168-hour block the model is given only the scaled history up to
    the block's first hour and asked for the next 168 values. Nothing is
    retrained, which matches how such a model would be run week to week.
    """
    errors: list[np.ndarray] = []
    for origin in validation_origins():
        naive_origin = origin.tz_localize(None)
        history = scaled_full.drop_after(naive_origin)
        prediction = scaler.inverse_transform(model.predict(n=HORIZON, series=history, num_samples=1))
        predicted = np.asarray(prediction.values(), dtype=float).ravel()
        block = pd.date_range(origin, periods=HORIZON, freq="h", tz="UTC")
        observed = actual.loc[block].to_numpy(dtype=float)
        errors.append(np.abs((observed - predicted) / observed))
    return float(np.mean(np.concatenate(errors)) * 100.0)


def deep_sweep_stage(
    name: str,
    build: Any,
    candidate: dict[str, Any],
    candidate_index: int,
    series: pd.Series,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    """Train one deep-learning candidate and score it on the validation window.

    Written as its own step, with its result cached to disk, so a long
    hyperparameter search can be run one candidate at a time and picked up
    again after an interruption.
    """
    import time

    from darts.dataprocessing.transformers import Scaler

    path = ARTIFACT_DIR / f"{name}_candidate{candidate_index}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    started = time.time()
    train = to_darts_series(series.loc[:TRAIN_END])
    full = to_darts_series(train_plus_val(series))

    scaler = Scaler()
    train_scaled = scaler.fit_transform(train)
    full_scaled = scaler.transform(full)
    # The validation series handed to early stopping needs one input window of
    # run-up before the validation window itself, or its first target has no
    # history to be predicted from.
    val_scaled = full_scaled.drop_before(
        pd.Timestamp(VAL_START.tz_localize(None)) - pd.Timedelta(hours=int(candidate["input_chunk_length"]) + 1)
    )

    set_seeds()
    model = build(candidate, max_epochs, patience, str(ARTIFACT_DIR))
    model.fit(train_scaled, val_series=val_scaled, verbose=False)
    epochs_used = int(model.trainer.current_epoch)

    record = {
        **{k: v for k, v in candidate.items() if k != "input_chunk_length"},
        "candidate_index": candidate_index,
        "epochs_used": epochs_used,
        "validation_mape_pct": rolling_origin_mape_torch(model, full_scaled, scaler, series),
        "seconds": time.time() - started,
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def deep_final_stage(
    name: str,
    build: Any,
    candidates: list[dict[str, Any]],
    series: pd.Series,
    n_samples: int,
    extra_hyperparameters: dict[str, Any],
) -> ForecastResult:
    """Refit the best candidate on train+validation and forecast the test week.

    Early stopping cannot be used here: the December data that the final model
    must learn from is the same data that would have to be held back to stop
    on. So the model is retrained from scratch for the number of epochs the
    winning candidate used during the search.
    """
    import time

    from darts.dataprocessing.transformers import Scaler

    started = time.time()
    records = []
    for index in range(len(candidates)):
        path = ARTIFACT_DIR / f"{name}_candidate{index}.json"
        if not path.exists():
            raise FileNotFoundError(f"Candidate {index} for {name} has not been evaluated yet.")
        records.append(json.loads(path.read_text(encoding="utf-8")))

    best = min(records, key=lambda record: record["validation_mape_pct"])
    best_candidate = candidates[int(best["candidate_index"])]

    fitting_data = to_darts_series(train_plus_val(series))
    scaler = Scaler()
    fitting_scaled = scaler.fit_transform(fitting_data)

    set_seeds()
    model = build(best_candidate, max(int(best["epochs_used"]), 1), None, str(ARTIFACT_DIR))
    model.fit(fitting_scaled, verbose=False)

    prediction = scaler.inverse_transform(model.predict(n=HORIZON, num_samples=n_samples))
    samples = np.asarray(prediction.all_values(), dtype=float)[:, 0, :]

    test = series.loc[TEST_START:TEST_END]
    quantiles = quantile_frame(test.index, samples)
    # With a quantile-regression loss the natural point forecast is the
    # predicted median, not the average of the sample cloud.
    point = pd.Series(quantiles["0.5"].to_numpy(), index=test.index, name=name)

    total_seconds = float(sum(record["seconds"] for record in records) + (time.time() - started))
    return ForecastResult(
        name=name,
        point_forecast=point,
        runtime_seconds=total_seconds,
        hyperparameters={
            **{k: v for k, v in best_candidate.items()},
            "epochs_used_after_early_stopping": int(best["epochs_used"]),
            "n_samples_for_intervals": n_samples,
            **extra_hyperparameters,
        },
        quantile_forecast=quantiles,
        validation_mape_pct=float(best["validation_mape_pct"]),
        extras={"grid_search": records},
    )
