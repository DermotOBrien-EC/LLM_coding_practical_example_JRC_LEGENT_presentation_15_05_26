from __future__ import annotations

# ruff: noqa: E402
# Cache paths and thread limits must precede library imports.

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "TMPDIR"):
    path = ROOT / ".cache" / key.lower()
    path.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(path)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "4"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

SEED = 206
HORIZON = 168
TRAIN_END = pd.Timestamp("2019-10-01")
TEST_START = pd.Timestamp("2020-01-01")
TEST_END = pd.Timestamp("2020-01-08")
QUANTILES = np.array([0.025, 0.1, 0.5, 0.9, 0.975])
NAMES = ["naive", "sarima", "prophet", "lightgbm", "nbeats", "patchtst"]
LABELS = dict(
    zip(
        NAMES,
        [
            "Seasonal naive",
            "SARIMA",
            "Prophet + DE holidays",
            "LightGBM",
            "N-BEATS",
            "Transformer (substitute)",
        ],
    )
)
COLORS = dict(zip(NAMES, ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#e377c2", "#d62728"]))


@dataclass
class ModelResult:
    name: str
    point: np.ndarray
    quantiles: np.ndarray | None
    runtime_seconds: float
    hyperparameters: dict[str, Any]
    validation: list[dict[str, Any]]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def load_data() -> pd.Series:
    frame = pd.read_csv(ROOT / "opsd_de_load.csv")
    expected = ["utc_timestamp", "DE_load_actual_entsoe_transparency"]
    if frame.columns.tolist() != expected:
        raise ValueError(f"Unexpected columns: {frame.columns.tolist()}")
    times = pd.DatetimeIndex(pd.to_datetime(frame[expected[0]], utc=True)).tz_localize(None)
    values = frame[expected[1]].to_numpy(dtype=float)
    wanted = pd.date_range("2015-01-01", "2020-09-30 23:00", freq="h")
    if not times.equals(wanted) or len(values) != 50400:
        raise ValueError(
            "Data anomaly: timestamps or row count differ from the declared hourly grid. No imputation performed."
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Data anomaly: non-finite or non-positive load. No imputation performed.")
    return pd.Series(values, index=wanted, name="load_mw")


def validation_blocks(pretest: pd.Series) -> list[tuple[pd.Series, pd.Series]]:
    if pretest.index[-1] >= TEST_START:
        raise ValueError("Validation input must end before the test window")
    val = pretest.loc[TRAIN_END:]
    return [
        (pretest.loc[: val.index[start] - pd.Timedelta(hours=1)], val.iloc[start : start + HORIZON])
        for start in range(0, len(val), HORIZON)
    ]


def mape(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(100 * np.mean(np.abs((np.asarray(actual) - prediction) / actual)))


def gaussian_quantiles(mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return np.asarray(mean)[:, None] + np.asarray(sigma)[:, None] * norm.ppf(QUANTILES)[None, :]


def score(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = actual - prediction
    return {
        "mape_test_pct": mape(actual, prediction),
        "rmse_test_mw": float(np.sqrt(np.mean(residual**2))),
        "mae_test_mw": float(np.mean(np.abs(residual))),
        "mape_jan1_pct": mape(actual[:24], prediction[:24]),
        "mape_jan2_to_jan7_pct": mape(actual[24:], prediction[24:]),
    }


def probability_scores(actual: np.ndarray, quantiles: np.ndarray) -> dict[str, float]:
    def coverage(low: int, high: int) -> float:
        return float(np.mean((actual >= quantiles[:, low]) & (actual <= quantiles[:, high])))

    output = {"winner_coverage_80pct": coverage(1, 3), "winner_coverage_95pct": coverage(0, 4)}
    for column, label in [(1, "q10"), (2, "q50"), (3, "q90")]:
        error = actual - quantiles[:, column]
        q = QUANTILES[column]
        output[f"winner_pinball_loss_{label}"] = float(
            np.mean(np.maximum(q * error, (q - 1) * error))
        )
    return output


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def model_source_digests(name: str) -> dict[str, str]:
    module = "lightgbm_features" if name == "lightgbm" else name
    names = ["__init__.py", "common.py", f"{module}.py"]
    if name in ("nbeats", "patchtst"):
        names.append("neural.py")
    return {name: hashlib.sha256((ROOT / "code" / name).read_bytes()).hexdigest() for name in names}


def save_result(result: ModelResult) -> None:
    if result.point.shape != (168,) or not np.isfinite(result.point).all():
        raise ValueError(f"Invalid forecast for {result.name}")
    if result.quantiles is not None:
        if result.quantiles.shape != (168, 5) or not np.isfinite(result.quantiles).all():
            raise ValueError(f"Invalid quantiles for {result.name}")
        if np.any(np.diff(result.quantiles, axis=1) < 0):
            raise ValueError(f"Crossed quantiles for {result.name}")
    frame = pd.DataFrame(
        {
            "utc_timestamp": pd.date_range(TEST_START, periods=168, freq="h"),
            "point_mw": result.point,
        }
    )
    if result.quantiles is not None:
        for i, q in enumerate(QUANTILES):
            frame[f"q{q:g}_mw"] = result.quantiles[:, i]
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    frame.to_csv(output / f"{result.name}_forecast.csv", index=False)
    write_json(
        output / f"{result.name}.json",
        {
            "name": result.name,
            "runtime_seconds": result.runtime_seconds,
            "hyperparameters": result.hyperparameters,
            "validation": result.validation,
            "diagnostics": result.diagnostics,
            "model_source_sha256": model_source_digests(result.name),
            "input_sha256": hashlib.sha256((ROOT / "opsd_de_load.csv").read_bytes()).hexdigest(),
        },
    )


def read_result(name: str) -> ModelResult:
    meta = json.loads((ROOT / "artifacts" / f"{name}.json").read_text())
    digest = hashlib.sha256((ROOT / "opsd_de_load.csv").read_bytes()).hexdigest()
    if meta["input_sha256"] != digest:
        raise ValueError("Cached result input digest mismatch")
    if meta["model_source_sha256"] != model_source_digests(name):
        raise ValueError(
            f"Cached {name} result does not match current modeling source; rerun forecasts"
        )
    frame = pd.read_csv(ROOT / "artifacts" / f"{name}_forecast.csv")
    q = None if name == "naive" else frame[[f"q{x:g}_mw" for x in QUANTILES]].to_numpy()
    return ModelResult(
        name,
        frame.point_mw.to_numpy(),
        q,
        meta["runtime_seconds"],
        meta["hyperparameters"],
        meta["validation"],
        meta["diagnostics"],
    )


def figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#777777",
            "axes.linewidth": 0.6,
            "grid.color": "#e7e7e7",
            "grid.linewidth": 0.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / name, dpi=300)
    plt.close(fig)
