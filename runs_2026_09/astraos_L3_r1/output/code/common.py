from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

SEED = 20200101
HORIZON = 168
QUANTILES = [0.025, 0.1, 0.5, 0.9, 0.975]
NAMES = ["naive", "sarima", "prophet", "lightgbm", "nbeats", "patchtst"]
LABELS = {
    "naive": "Seasonal naive",
    "sarima": "SARIMA",
    "prophet": "Prophet + DE holidays",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "Transformer (PatchTST substitute)",
}
SHORT_LABELS = {**LABELS, "prophet": "Prophet", "patchtst": "Transformer*"}
COLORS = dict(zip(NAMES, ["#1f77b4", "#ff7f0e", "#9467bd", "#17becf", "#2ca02c", "#e377c2"]))


@dataclass
class ModelResult:
    name: str
    point: np.ndarray
    quantiles: np.ndarray | None
    validation_point: np.ndarray
    hyperparameters: dict[str, Any]
    validation_candidates: list[dict[str, Any]]
    runtime_seconds: float
    extras: dict[str, Any] = field(default_factory=dict)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def load_data() -> pd.Series:
    frame = pd.read_csv(ROOT / "opsd_de_load.csv")
    expected = {"utc_timestamp", "DE_load_actual_entsoe_transparency"}
    if set(frame.columns) != expected or len(frame) != 50400:
        raise ValueError("Unexpected CSV columns or row count; no automatic repair is allowed.")
    dates = pd.to_datetime(frame["utc_timestamp"], utc=True)
    expected_dates = pd.date_range("2015-01-01", "2020-09-30 23:00", freq="h", tz="UTC")
    if not pd.DatetimeIndex(dates).equals(expected_dates):
        raise ValueError("Timestamps differ from the complete expected hourly UTC grid.")
    values = frame["DE_load_actual_entsoe_transparency"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Load must be finite and positive; no imputation is performed.")
    # Darts and Prophet require timezone-naive indices. These instants remain UTC.
    return pd.Series(values, index=expected_dates.tz_localize(None), name="load_mw")


def validation_blocks(pretest: pd.Series, train_size: int) -> list[tuple[pd.Series, pd.Series]]:
    if pretest.index[-1] >= pd.Timestamp("2020-01-01"):
        raise ValueError("Test observations must not enter validation.")
    return [(pretest.iloc[:origin], pretest.iloc[origin:origin + HORIZON])
            for origin in range(train_size, len(pretest), HORIZON)]


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    a = np.asarray(actual, dtype=float).reshape(-1)
    p = np.asarray(predicted, dtype=float).reshape(-1)
    if a.shape != p.shape or not np.isfinite(p).all() or (a <= 0).any():
        raise ValueError("MAPE requires matching finite forecasts and positive actuals.")
    return float(np.mean(np.abs((a - p) / a)) * 100)


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    p = np.asarray(predicted).reshape(-1)
    a = actual.to_numpy()
    jan1 = actual.index.normalize() == pd.Timestamp("2020-01-01")
    return {
        "mape_test_pct": mape(a, p),
        "rmse_test_mw": float(np.sqrt(np.mean((a - p) ** 2))),
        "mae_test_mw": float(np.mean(np.abs(a - p))),
        "mape_jan1_pct": mape(a[jan1], p[jan1]),
        "mape_jan2_to_jan7_pct": mape(a[~jan1], p[~jan1]),
    }


def quantile_frame(values: np.ndarray, index: pd.Index | None = None) -> pd.DataFrame:
    q = np.asarray(values, dtype=float)
    if q.ndim != 2 or q.shape[1] != len(QUANTILES) or not np.isfinite(q).all():
        raise ValueError("Expected five finite forecast quantiles per timestamp.")
    return pd.DataFrame(np.sort(q, axis=1), index=index, columns=[f"q{x:g}" for x in QUANTILES])


def probabilistic_metrics(actual: np.ndarray, q: np.ndarray) -> dict[str, float]:
    a = np.asarray(actual)
    q = quantile_frame(q).to_numpy()
    out = {
        "winner_coverage_80pct": float(np.mean((a >= q[:, 1]) & (a <= q[:, 3]))),
        "winner_coverage_95pct": float(np.mean((a >= q[:, 0]) & (a <= q[:, 4]))),
    }
    for col, prob, name in [(1, 0.1, "q10"), (2, 0.5, "q50"), (3, 0.9, "q90")]:
        error = a - q[:, col]
        out[f"winner_pinball_loss_{name}"] = float(np.mean(np.maximum(prob * error, (prob - 1) * error)))
    return out


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_digest() -> str:
    return hashlib.sha256((ROOT / "opsd_de_load.csv").read_bytes()).hexdigest()


def figure_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#b8b8b8", "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": "#e7e7e7", "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.linewidth": 1.5,
        "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 100,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save_figure(fig: Any, name: str) -> None:
    import matplotlib.pyplot as plt
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / name, dpi=300)
    plt.close(fig)
