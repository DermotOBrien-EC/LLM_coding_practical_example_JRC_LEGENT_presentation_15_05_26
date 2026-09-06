from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for key, value in {
    "MPLCONFIGDIR": str(ROOT / ".cache" / "matplotlib"),
    "XDG_CACHE_HOME": str(ROOT / ".cache"),
    "TORCH_HOME": str(ROOT / ".cache" / "torch"),
    "TMPDIR": str(ROOT / ".cache" / "tmp"),
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "VECLIB_MAXIMUM_THREADS": "4",
}.items():
    os.environ[key] = value
for directory in ["artifacts", "figures", ".cache/tmp", ".cache/matplotlib"]:
    (ROOT / directory).mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEED = 2026
HORIZON = 168
QUANTILES = [0.025, 0.1, 0.5, 0.9, 0.975]
TRAIN_END = pd.Timestamp("2019-09-30 23:00", tz="UTC")
VALIDATION_START = pd.Timestamp("2019-10-01", tz="UTC")
TEST_START = pd.Timestamp("2020-01-01", tz="UTC")
TEST_INDEX = pd.date_range(TEST_START, periods=HORIZON, freq="h")
NAMES = ["naive", "sarima", "prophet", "lightgbm", "nbeats", "patchtst"]
LABELS = {
    "naive": "Seasonal naive",
    "sarima": "SARIMA + weekly Fourier",
    "prophet": "Prophet + DE holidays",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "Transformer (substitute)",
}
COLORS = dict(zip(NAMES, ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#e377c2", "#d62728"]))


@dataclass
class ForecastResult:
    name: str
    point: np.ndarray
    quantiles: np.ndarray | None
    runtime_seconds: float
    hyperparameters: dict[str, Any]
    validation: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def load_data() -> pd.Series:
    frame = pd.read_csv(ROOT / "opsd_de_load.csv")
    expected = ["utc_timestamp", "DE_load_actual_entsoe_transparency"]
    if list(frame.columns) != expected:
        raise ValueError(f"Unexpected columns: {list(frame.columns)}")
    index = pd.DatetimeIndex(pd.to_datetime(frame.utc_timestamp, utc=True))
    values = frame[expected[1]].to_numpy(dtype=float)
    required = pd.date_range("2015-01-01", "2020-09-30 23:00", freq="h", tz="UTC")
    if len(frame) != 50400 or not index.equals(required):
        raise ValueError(
            "Data anomaly: unexpected count, endpoints, gaps or duplicate hours. No imputation performed."
        )
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(
            "Data anomaly: missing, non-finite or nonpositive load. No imputation performed."
        )
    return pd.Series(values, index=index, name="load_mw")


def validation_blocks(pretest: pd.Series) -> Iterator[tuple[pd.Series, pd.Series]]:
    if pretest.index[-1] >= TEST_START:
        raise ValueError("Validation function must never receive test observations")
    start = pretest.index.get_loc(VALIDATION_START)
    for position in range(start, len(pretest), HORIZON):
        yield pretest.iloc[:position], pretest.iloc[position : position + HORIZON]


def mape(actual: np.ndarray, prediction: np.ndarray) -> float:
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    if actual.shape != prediction.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid prediction shape or non-finite values")
    if not np.isfinite(actual).all() or (actual <= 0).any():
        raise ValueError("MAPE requires finite positive observations")
    return float(np.mean(np.abs(actual - prediction) / actual) * 100.0)


def score(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.asarray(actual) - np.asarray(prediction)
    return {
        "mape_test_pct": mape(actual, prediction),
        "rmse_test_mw": float(np.sqrt(np.mean(error**2))),
        "mae_test_mw": float(np.mean(np.abs(error))),
    }


def quantile_diagnostics(actual: np.ndarray, quantiles: np.ndarray) -> dict[str, float]:
    if quantiles.shape != (len(actual), 5) or not np.isfinite(quantiles).all():
        raise ValueError("Expected five finite forecast quantiles per hour")
    if (np.diff(quantiles, axis=1) < 0).any():
        raise ValueError("Quantile crossing")
    output = {
        "winner_coverage_80pct": float(
            np.mean((actual >= quantiles[:, 1]) & (actual <= quantiles[:, 3]))
        ),
        "winner_coverage_95pct": float(
            np.mean((actual >= quantiles[:, 0]) & (actual <= quantiles[:, 4]))
        ),
    }
    for level, column, suffix in [(0.1, 1, "q10"), (0.5, 2, "q50"), (0.9, 3, "q90")]:
        error = actual - quantiles[:, column]
        output[f"winner_pinball_loss_{suffix}"] = float(
            np.maximum(level * error, (level - 1) * error).mean()
        )
    return output


def as_darts(series: pd.Series, scale: float = 1.0) -> Any:
    from darts import TimeSeries

    return TimeSeries.from_times_and_values(
        series.index.tz_convert("UTC").tz_localize(None),
        (series.to_numpy() / scale).astype(np.float32),
        columns=["load"],
        freq="h",
    )


def write_json(path: Path, content: Any) -> None:
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def save_result(result: ForecastResult) -> None:
    if result.point.shape != (HORIZON,) or not np.isfinite(result.point).all():
        raise ValueError(f"Invalid point forecast for {result.name}")
    frame = pd.DataFrame({"utc_timestamp": TEST_INDEX, "point_mw": result.point})
    if result.quantiles is not None:
        quantile_diagnostics(np.ones(HORIZON), result.quantiles)
        for level, values in zip(QUANTILES, result.quantiles.T):
            frame[f"q{level:g}_mw"] = values
    frame.to_csv(ROOT / "artifacts" / f"{result.name}_forecast.csv", index=False)
    write_json(
        ROOT / "artifacts" / f"{result.name}_run.json",
        {
            "name": result.name,
            "runtime_seconds": result.runtime_seconds,
            "hyperparameters": result.hyperparameters,
            "validation": result.validation,
            "diagnostics": result.diagnostics,
        },
    )


def load_result(name: str) -> ForecastResult:
    frame = pd.read_csv(ROOT / "artifacts" / f"{name}_forecast.csv")
    if not pd.DatetimeIndex(pd.to_datetime(frame.utc_timestamp, utc=True)).equals(TEST_INDEX):
        raise ValueError("Saved forecast has the wrong timestamps")
    metadata = json.loads((ROOT / "artifacts" / f"{name}_run.json").read_text())
    quantiles = None if name == "naive" else frame[[f"q{q:g}_mw" for q in QUANTILES]].to_numpy()
    return ForecastResult(point=frame.point_mw.to_numpy(), quantiles=quantiles, **metadata)


def source_digest() -> str:
    return hashlib.sha256((ROOT / "opsd_de_load.csv").read_bytes()).hexdigest()
