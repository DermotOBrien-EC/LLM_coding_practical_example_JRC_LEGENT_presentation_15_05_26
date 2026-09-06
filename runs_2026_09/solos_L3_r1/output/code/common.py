from __future__ import annotations

import json
import math
import platform
import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

import holidays
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "opsd_de_load.csv"
FIGURES_DIR = ROOT / "figures"
TARGET_COLUMN = "DE_load_actual_entsoe_transparency"
TIMESTAMP_COLUMN = "utc_timestamp"

TRAIN_START = pd.Timestamp("2015-01-01 00:00:00", tz="UTC")
VALIDATION_START = pd.Timestamp("2019-10-01 00:00:00", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
TEST_END_EXCLUSIVE = pd.Timestamp("2020-01-08 00:00:00", tz="UTC")

MODEL_ORDER = ["naive", "sarima", "prophet", "lightgbm", "nbeats", "patchtst"]
MODEL_DISPLAY_NAMES = {
    "naive": "Seasonal naive",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "TSMixer substitute",
}

# This is a reordered six-colour subset of matplotlib tab10. The order passes
# the data-visualisation palette validator for adjacent categorical marks.
MODEL_COLORS = {
    "naive": "#1f77b4",
    "sarima": "#ff7f0e",
    "prophet": "#9467bd",
    "lightgbm": "#17becf",
    "nbeats": "#d62728",
    "patchtst": "#e377c2",
}
MODEL_HATCHES = {
    "naive": "",
    "sarima": "///",
    "prophet": "\\\\\\",
    "lightgbm": "xx",
    "nbeats": "..",
    "patchtst": "++",
}

SEED = 20200930
QUANTILES = (0.025, 0.1, 0.5, 0.9, 0.975)


@dataclass(frozen=True)
class DataSplits:
    full: pd.Series
    train: pd.Series
    validation: pd.Series
    train_validation: pd.Series
    test: pd.Series


@dataclass
class ForecastResult:
    name: str
    forecast: pd.Series
    runtime_seconds: float
    hyperparameters: dict[str, Any]
    quantiles: dict[float, pd.Series] = field(default_factory=dict)
    feature_importance: pd.Series | None = None
    decomposition: pd.DataFrame | None = None

    @property
    def lower_80(self) -> pd.Series | None:
        return self.quantiles.get(0.1)

    @property
    def upper_80(self) -> pd.Series | None:
        return self.quantiles.get(0.9)

    @property
    def lower_95(self) -> pd.Series | None:
        return self.quantiles.get(0.025)

    @property
    def upper_95(self) -> pd.Series | None:
        return self.quantiles.get(0.975)


@dataclass(frozen=True)
class Standardizer:
    mean: float
    standard_deviation: float

    @classmethod
    def fit(cls, series: pd.Series) -> Standardizer:
        standard_deviation = float(series.std(ddof=0))
        if standard_deviation <= 0.0:
            raise ValueError("The load series has zero variance.")
        return cls(mean=float(series.mean()), standard_deviation=standard_deviation)

    def transform(self, series: pd.Series) -> pd.Series:
        values = (series.astype(float) - self.mean) / self.standard_deviation
        return pd.Series(values.to_numpy(dtype=np.float32), index=series.index, name=series.name)

    def inverse_array(self, values: np.ndarray) -> np.ndarray:
        return values * self.standard_deviation + self.mean


@dataclass(frozen=True)
class PositiveMeanScaler:
    scale: float

    @classmethod
    def fit(cls, series: pd.Series) -> PositiveMeanScaler:
        scale = float(series.mean())
        if scale <= 0.0:
            raise ValueError("Positive mean scaling requires a positive load mean.")
        return cls(scale=scale)

    def transform(self, series: pd.Series) -> pd.Series:
        values = series.astype(float) / self.scale
        return pd.Series(values.to_numpy(dtype=np.float32), index=series.index, name=series.name)

    def inverse_array(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale


def configure_figure_style() -> None:
    mpl.use("Agg", force=True)
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": "#0b0b0b",
            "axes.titlecolor": "#0b0b0b",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e1e0d9",
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "font.family": "sans-serif",
            "font.size": 9,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "savefig.dpi": 300,
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
        }
    )


def set_random_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


def load_data(path: Path = DATA_PATH) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=[TIMESTAMP_COLUMN])
    expected_columns = [TIMESTAMP_COLUMN, TARGET_COLUMN]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"Expected columns {expected_columns}, found {list(frame.columns)}")
    if len(frame) != 50_400:
        raise ValueError(f"Expected 50,400 rows, found {len(frame):,}")

    timestamp = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    if timestamp.duplicated().any():
        raise ValueError("Duplicate timestamps were found.")
    if not timestamp.is_monotonic_increasing:
        raise ValueError("Timestamps are not increasing.")
    expected_index = pd.date_range(timestamp.iloc[0], timestamp.iloc[-1], freq="h")
    if len(expected_index) != len(timestamp) or not np.array_equal(
        expected_index.asi8, pd.DatetimeIndex(timestamp).asi8
    ):
        raise ValueError("The timestamp sequence is not complete at hourly frequency.")

    load = pd.to_numeric(frame[TARGET_COLUMN], errors="raise")
    if load.isna().any():
        raise ValueError("Missing load values were found.")
    if not np.isfinite(load.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite load values were found.")

    series = pd.Series(
        load.to_numpy(dtype=float),
        index=pd.DatetimeIndex(timestamp),
        name=TARGET_COLUMN,
    )
    if series.index[0] != TRAIN_START or series.index[-1] != pd.Timestamp(
        "2020-09-30 23:00:00", tz="UTC"
    ):
        raise ValueError("The dataset coverage differs from the stated study coverage.")
    return series


def make_splits(series: pd.Series) -> DataSplits:
    train = series.loc[(series.index >= TRAIN_START) & (series.index < VALIDATION_START)]
    validation = series.loc[
        (series.index >= VALIDATION_START) & (series.index < TEST_START)
    ]
    train_validation = series.loc[
        (series.index >= TRAIN_START) & (series.index < TEST_START)
    ]
    test = series.loc[
        (series.index >= TEST_START) & (series.index < TEST_END_EXCLUSIVE)
    ]
    expected_counts = (41_616, 2_208, 43_824, 168)
    observed_counts = (len(train), len(validation), len(train_validation), len(test))
    if observed_counts != expected_counts:
        raise ValueError(
            f"Split counts differ from the frozen design: {observed_counts} != {expected_counts}"
        )
    return DataSplits(
        full=series,
        train=train,
        validation=validation,
        train_validation=train_validation,
        test=test,
    )


def german_holiday_indicator(index: pd.DatetimeIndex) -> np.ndarray:
    years = range(int(index.year.min()), int(index.year.max()) + 1)
    calendar = holidays.Germany(years=years)
    return np.fromiter(
        (1 if timestamp.date() in calendar else 0 for timestamp in index),
        dtype=np.int8,
        count=len(index),
    )


def mean_absolute_percentage_error(actual: pd.Series, forecast: pd.Series) -> float:
    actual_aligned, forecast_aligned = actual.align(forecast, join="inner")
    if len(actual_aligned) != len(actual) or len(actual_aligned) != len(forecast):
        raise ValueError("Actual and forecast indices do not match.")
    return float(
        np.mean(
            np.abs(
                (actual_aligned.to_numpy(dtype=float) - forecast_aligned.to_numpy(dtype=float))
                / actual_aligned.to_numpy(dtype=float)
            )
        )
        * 100.0
    )


def root_mean_squared_error(actual: pd.Series, forecast: pd.Series) -> float:
    errors = actual.to_numpy(dtype=float) - forecast.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(errors))))


def mean_absolute_error(actual: pd.Series, forecast: pd.Series) -> float:
    errors = actual.to_numpy(dtype=float) - forecast.to_numpy(dtype=float)
    return float(np.mean(np.abs(errors)))


def pinball_loss(actual: pd.Series, quantile_forecast: pd.Series, quantile: float) -> float:
    errors = actual.to_numpy(dtype=float) - quantile_forecast.to_numpy(dtype=float)
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1.0) * errors)))


def interval_coverage(
    actual: pd.Series, lower: pd.Series, upper: pd.Series
) -> float:
    actual_values = actual.to_numpy(dtype=float)
    lower_values = lower.to_numpy(dtype=float)
    upper_values = upper.to_numpy(dtype=float)
    return float(np.mean((actual_values >= lower_values) & (actual_values <= upper_values)))


def per_day_mape(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    absolute_percentage_error = (
        np.abs(actual.to_numpy(dtype=float) - forecast.to_numpy(dtype=float))
        / actual.to_numpy(dtype=float)
        * 100.0
    )
    values = pd.Series(absolute_percentage_error, index=actual.index)
    return values.groupby(values.index.floor("D")).mean()


def evaluate_forecast(actual: pd.Series, result: ForecastResult) -> dict[str, Any]:
    validate_result(actual, result)
    jan1_mask = actual.index.date == pd.Timestamp("2020-01-01").date()
    jan2_to_jan7_mask = ~jan1_mask
    return {
        "name": result.name,
        "mape_test_pct": mean_absolute_percentage_error(actual, result.forecast),
        "rmse_test_mw": root_mean_squared_error(actual, result.forecast),
        "mae_test_mw": mean_absolute_error(actual, result.forecast),
        "mape_jan1_pct": mean_absolute_percentage_error(
            actual.loc[jan1_mask], result.forecast.loc[jan1_mask]
        ),
        "mape_jan2_to_jan7_pct": mean_absolute_percentage_error(
            actual.loc[jan2_to_jan7_mask], result.forecast.loc[jan2_to_jan7_mask]
        ),
        "runtime_seconds": float(result.runtime_seconds),
        "hyperparameters": result.hyperparameters,
    }


def validate_result(actual: pd.Series, result: ForecastResult) -> None:
    if result.name not in MODEL_ORDER:
        raise ValueError(f"Unknown model name: {result.name}")
    if not result.forecast.index.equals(actual.index):
        raise ValueError(f"{result.name} forecast index does not match the test index.")
    if len(result.forecast) != 168:
        raise ValueError(f"{result.name} did not forecast exactly 168 hours.")
    if not np.isfinite(result.forecast.to_numpy(dtype=float)).all():
        raise ValueError(f"{result.name} forecast contains non-finite values.")

    if result.quantiles:
        if set(result.quantiles) != set(QUANTILES):
            raise ValueError(f"{result.name} does not provide the required quantiles.")
        matrix = np.column_stack(
            [result.quantiles[quantile].to_numpy(dtype=float) for quantile in QUANTILES]
        )
        if not all(series.index.equals(actual.index) for series in result.quantiles.values()):
            raise ValueError(f"{result.name} quantile indices do not match the test index.")
        if not np.isfinite(matrix).all():
            raise ValueError(f"{result.name} quantiles contain non-finite values.")
        if np.any(np.diff(matrix, axis=1) < -1e-8):
            raise ValueError(f"{result.name} quantile forecasts cross.")


def quantile_series_from_samples(
    samples: np.ndarray, index: pd.DatetimeIndex
) -> dict[float, pd.Series]:
    if samples.ndim != 2 or samples.shape[0] != len(index):
        raise ValueError(f"Expected a time by sample matrix, found shape {samples.shape}")
    quantile_values = np.quantile(samples, QUANTILES, axis=1).T
    quantile_values.sort(axis=1)
    return {
        quantile: pd.Series(quantile_values[:, position], index=index, dtype=float)
        for position, quantile in enumerate(QUANTILES)
    }


def normal_quantile_series(
    mean: pd.Series, standard_error: pd.Series
) -> dict[float, pd.Series]:
    output: dict[float, pd.Series] = {}
    for quantile in QUANTILES:
        z_score = NormalDist().inv_cdf(quantile)
        output[quantile] = mean + z_score * standard_error
    return output


def enforce_non_crossing_quantiles(
    quantiles: dict[float, pd.Series], index: pd.DatetimeIndex
) -> dict[float, pd.Series]:
    matrix = np.column_stack(
        [quantiles[quantile].reindex(index).to_numpy(dtype=float) for quantile in QUANTILES]
    )
    matrix.sort(axis=1)
    return {
        quantile: pd.Series(matrix[:, position], index=index, dtype=float)
        for position, quantile in enumerate(QUANTILES)
    }


def make_darts_series(series: pd.Series) -> Any:
    from darts import TimeSeries

    naive_utc_index = series.index.tz_convert("UTC").tz_localize(None)
    return TimeSeries.from_times_and_values(
        naive_utc_index,
        series.to_numpy(dtype=np.float32),
        columns=[TARGET_COLUMN],
        fill_missing_dates=False,
        freq="h",
    )


def darts_samples_to_quantiles(
    prediction: Any,
    scaler: Standardizer | PositiveMeanScaler,
    target_index: pd.DatetimeIndex,
) -> dict[float, pd.Series]:
    values = prediction.all_values(copy=False)
    if values.ndim != 3 or values.shape[1] != 1:
        raise ValueError(f"Unexpected Darts forecast shape: {values.shape}")
    samples = scaler.inverse_array(values[:, 0, :].astype(float))
    return quantile_series_from_samples(samples, target_index)


def torch_accelerator() -> tuple[str, int | str]:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps", 1
    except ImportError:
        pass
    return "cpu", "auto"


def uname_output() -> str:
    try:
        return subprocess.check_output(["uname", "-a"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.platform()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite numbers cannot be written to JSON.")
        return number
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
