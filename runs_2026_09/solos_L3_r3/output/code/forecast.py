from __future__ import annotations

# Third-party import time is part of the reported study runtime.
# ruff: noqa: E402

import importlib.util
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

_PROCESS_STARTED_AT = time.perf_counter()

# The required local filename prophet.py would otherwise shadow Prophet's package.
_CODE_DIRECTORY = Path(__file__).resolve().parent
_ORIGINAL_SYS_PATH = sys.path.copy()
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != _CODE_DIRECTORY]
try:
    _DARTS_PROPHET_CLASS = getattr(importlib.import_module("darts.models"), "Prophet")
finally:
    sys.path[:] = _ORIGINAL_SYS_PATH

import numpy as np

from common import (
    MODEL_ORDER,
    QUANTILES,
    ForecastResult,
    get_hardware_description,
    load_and_split_data,
    model_metrics,
    plot_daily_mape,
    plot_feature_importance,
    plot_forecast_comparison,
    plot_metric_comparison,
    plot_overview,
    plot_prophet_decomposition,
    plot_residuals,
    plot_winner_with_intervals,
    winner_probabilistic_metrics,
    write_metrics_files,
    write_transcript,
)
from lightgbm_features import run_lightgbm
from naive import run_naive
from nbeats import run_nbeats
from patchtst import run_patchtst_substitute
from sarima import run_sarima


def _load_local_prophet_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "study_prophet_model", _CODE_DIRECTORY / "prophet.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load the local Prophet model module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_prophet = _load_local_prophet_module().run_prophet
ModelRunner = Callable[[], ForecastResult]


def _validate_result(result: ForecastResult, expected_name: str, expected_length: int) -> None:
    if result.name != expected_name:
        raise ValueError(f"Runner {expected_name} returned result name {result.name}")
    if result.name not in MODEL_ORDER:
        raise ValueError(f"Unknown model name {result.name}")
    if result.point.shape != (expected_length,):
        raise ValueError(f"{result.name} returned point shape {result.point.shape}")
    if not np.isfinite(result.point).all():
        raise ValueError(f"{result.name} returned non-finite point forecasts")
    for probability, values in result.quantiles.items():
        if np.asarray(values).shape != (expected_length,):
            raise ValueError(f"{result.name} quantile {probability:g} has the wrong shape")
        if not np.isfinite(values).all():
            raise ValueError(f"{result.name} quantile {probability:g} is non-finite")
    if expected_name == "naive":
        if result.quantiles:
            raise ValueError("Seasonal naive unexpectedly returned quantiles")
    elif set(result.quantiles) != set(QUANTILES):
        raise ValueError(f"{result.name} returned an incomplete quantile set")


def run_study(base_directory: Path, started_at: float | None = None) -> None:
    total_started = time.perf_counter() if started_at is None else started_at
    figures_directory = base_directory / "figures"
    figures_directory.mkdir(exist_ok=True)
    splits = load_and_split_data(base_directory / "opsd_de_load.csv")
    test_index = splits.test.index

    runners: tuple[tuple[str, ModelRunner], ...] = (
        ("naive", lambda: run_naive(splits.train_validation, test_index)),
        (
            "sarima",
            lambda: run_sarima(
                splits.train,
                splits.validation,
                splits.train_validation,
                test_index,
            ),
        ),
        (
            "prophet",
            lambda: run_prophet(
                splits.train,
                splits.validation,
                splits.train_validation,
                test_index,
            ),
        ),
        (
            "lightgbm",
            lambda: run_lightgbm(
                splits.train,
                splits.validation,
                splits.train_validation,
                test_index,
            ),
        ),
        (
            "nbeats",
            lambda: run_nbeats(
                splits.train,
                splits.validation,
                splits.train_validation,
                test_index,
            ),
        ),
        (
            "patchtst",
            lambda: run_patchtst_substitute(
                splits.train,
                splits.validation,
                splits.train_validation,
                test_index,
            ),
        ),
    )

    results: dict[str, ForecastResult] = {}
    for name, runner in runners:
        print(f"Running {name}...", flush=True)
        result = runner()
        _validate_result(result, name, len(splits.test))
        results[name] = result
        print(f"Completed {name} in {result.runtime_seconds:.1f} seconds", flush=True)

    metrics: list[dict[str, Any]] = [
        model_metrics(results[name], splits.test) for name in MODEL_ORDER
    ]
    metrics_by_name: dict[str, dict[str, Any]] = {row["name"]: row for row in metrics}
    winner_name: str = min(metrics, key=lambda row: row["mape_test_pct"])["name"]
    winner: ForecastResult = results[winner_name]
    if not winner.quantiles:
        raise RuntimeError(
            "The seasonal-naive baseline won but has no probabilistic forecast; "
            "winner interval metrics cannot be computed under the study protocol"
        )
    probabilistic_metrics: dict[str, float] = winner_probabilistic_metrics(winner, splits.test)

    plot_overview(splits.full, figures_directory / "01_overview.png")
    plot_forecast_comparison(
        results,
        metrics_by_name,
        splits.test,
        figures_directory / "02_forecast_comparison.png",
    )
    plot_metric_comparison(metrics, figures_directory / "03_metric_comparison.png")
    daily = plot_daily_mape(
        results,
        metrics,
        splits.test,
        figures_directory / "04_per_day_mape.png",
    )
    plot_winner_with_intervals(
        winner,
        metrics_by_name[winner_name],
        probabilistic_metrics,
        splits.test,
        figures_directory / "05_winner_with_intervals.png",
    )
    plot_residuals(winner, splits.test, figures_directory / "06_residuals.png")

    top_two: set[str] = {
        row["name"] for row in sorted(metrics, key=lambda row: row["mape_test_pct"])[:2]
    }
    feature_importance_path = figures_directory / "07_feature_importance.png"
    decomposition_path = figures_directory / "08_decomposition.png"
    feature_importance_path.unlink(missing_ok=True)
    decomposition_path.unlink(missing_ok=True)
    if "lightgbm" in top_two:
        plot_feature_importance(results["lightgbm"], feature_importance_path)
    if "prophet" in top_two:
        plot_prophet_decomposition(results["prophet"], decomposition_path)

    total_runtime_seconds = time.perf_counter() - total_started
    write_metrics_files(
        metrics,
        winner_name,
        probabilistic_metrics,
        total_runtime_seconds,
        base_directory / "metrics.json",
        base_directory / "metrics.csv",
    )
    write_transcript(
        base_directory / "transcript.md",
        metrics,
        winner_name,
        probabilistic_metrics,
        total_runtime_seconds,
        get_hardware_description(),
        daily,
    )
    print(f"Winner: {winner_name}", flush=True)
    print(f"Total runtime: {total_runtime_seconds:.1f} seconds", flush=True)


if __name__ == "__main__":
    run_study(Path(__file__).resolve().parents[1], started_at=_PROCESS_STARTED_AT)
