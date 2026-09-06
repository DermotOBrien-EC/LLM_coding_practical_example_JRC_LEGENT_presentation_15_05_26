from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROCESS_STARTED = time.perf_counter()
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent


def _preload_vendor_prophet() -> None:
    original_path = list(sys.path)
    try:
        sys.path = [
            entry
            for entry in sys.path
            if Path(entry or ".").resolve() != SCRIPT_DIR
        ]
        importlib.import_module("prophet")
    finally:
        sys.path = original_path


_preload_vendor_prophet()

from common import (  # noqa: E402
    FIGURES_DIR,
    MODEL_COLORS,
    MODEL_DISPLAY_NAMES,
    MODEL_ORDER,
    QUANTILES,
    SEED,
    DataSplits,
    ForecastResult,
    configure_figure_style,
    evaluate_forecast,
    interval_coverage,
    load_data,
    make_splits,
    mean_absolute_percentage_error,
    per_day_mape,
    pinball_loss,
    set_random_seeds,
    uname_output,
    validate_result,
    write_json,
)
from lightgbm_features import run as run_lightgbm  # noqa: E402
from naive import run as run_naive  # noqa: E402
from nbeats import run as run_nbeats  # noqa: E402
from patchtst import run as run_patchtst  # noqa: E402
from sarima import run as run_sarima  # noqa: E402

CACHE_DIR = SCRIPT_DIR / ".study_cache"


def _load_local_prophet_runner() -> Callable[[DataSplits], ForecastResult]:
    module_path = SCRIPT_DIR / "prophet.py"
    specification = importlib.util.spec_from_file_location(
        "forecast_study_prophet", module_path
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module.run


def _date_axis(axis: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _save_figure(figure: plt.Figure, filename: str) -> None:
    path = FIGURES_DIR / filename
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _run_models(splits: DataSplits, resume: bool) -> list[ForecastResult]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    runners: dict[str, Callable[[DataSplits], ForecastResult]] = {
        "naive": run_naive,
        "sarima": run_sarima,
        "prophet": _load_local_prophet_runner(),
        "lightgbm": run_lightgbm,
        "nbeats": run_nbeats,
        "patchtst": run_patchtst,
    }
    results: list[ForecastResult] = []
    for model_name in MODEL_ORDER:
        cache_path = CACHE_DIR / f"{model_name}.pkl"
        if resume and cache_path.exists():
            with cache_path.open("rb") as handle:
                result = pickle.load(handle)
            validate_result(splits.test, result)
            print(f"[{model_name}] loaded validated cache", flush=True)
        else:
            print(f"[{model_name}] fitting and forecasting", flush=True)
            result = runners[model_name](splits)
            validate_result(splits.test, result)
            with cache_path.open("wb") as handle:
                pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(
                f"[{model_name}] completed in {result.runtime_seconds:.1f} seconds",
                flush=True,
            )
        results.append(result)
    return results


def _metrics_frame(model_metrics: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(model_metrics)
    return frame.set_index("name").loc[MODEL_ORDER].reset_index()


def _plot_overview(splits: DataSplits) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(
        splits.full.index,
        splits.full.to_numpy(dtype=float),
        color="#b8b8b8",
        linewidth=0.45,
        label="Observed hourly load",
        rasterized=True,
    )
    axis.axvspan(
        splits.test.index[0],
        splits.test.index[-1] + pd.Timedelta(hours=1),
        color="#d62728",
        alpha=0.28,
        label="Held-out test window",
    )
    axis.plot(
        splits.test.index,
        splits.test.to_numpy(dtype=float),
        color="#d62728",
        linewidth=1.4,
    )
    axis.set_title("German national hourly electricity load and held-out test week")
    axis.set_xlabel("Date (UTC)")
    axis.set_ylabel("Load (MW)")
    axis.legend(loc="upper right", ncol=2)
    _date_axis(axis)
    figure.tight_layout()
    _save_figure(figure, "01_overview.png")


def _plot_forecast_comparison(
    splits: DataSplits,
    results: list[ForecastResult],
    metrics_by_name: dict[str, dict[str, Any]],
) -> None:
    all_values = [splits.test.to_numpy(dtype=float)] + [
        result.forecast.to_numpy(dtype=float) for result in results
    ]
    lower = min(float(values.min()) for values in all_values)
    upper = max(float(values.max()) for values in all_values)
    padding = 0.04 * (upper - lower)

    figure, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for axis, result in zip(axes.flat, results, strict=True):
        axis.plot(
            splits.test.index,
            splits.test.to_numpy(dtype=float),
            color="#0b0b0b",
            linewidth=1.8,
            label="Observed",
        )
        axis.plot(
            result.forecast.index,
            result.forecast.to_numpy(dtype=float),
            color=MODEL_COLORS[result.name],
            linewidth=1.8,
            label="Forecast",
        )
        axis.set_title(
            f"{MODEL_DISPLAY_NAMES[result.name]} | "
            f"MAPE {metrics_by_name[result.name]['mape_test_pct']:.2f}%"
        )
        axis.set_ylim(lower - padding, upper + padding)
        axis.legend(loc="upper right", fontsize=7)
        axis.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for axis in axes[:, 0]:
        axis.set_ylabel("Load (MW)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Date (UTC)")
    figure.suptitle("Observed and forecast German load during the held-out test week")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(figure, "02_forecast_comparison.png")


def _plot_metric_comparison(metrics: pd.DataFrame) -> None:
    ordered = metrics.sort_values("mape_test_pct").reset_index(drop=True)
    metric_specs = [
        ("mape_test_pct", "MAPE", "%", ""),
        ("rmse_test_mw", "RMSE", "MW", "///"),
        ("mae_test_mw", "MAE", "MW", "xx"),
    ]
    positions = np.arange(len(ordered), dtype=float)
    width = 0.23

    figure, axis = plt.subplots(figsize=(6, 6))
    for metric_position, (column, label, unit, hatch) in enumerate(metric_specs):
        raw = ordered[column].to_numpy(dtype=float)
        relative = raw / raw.min()
        offset = (metric_position - 1) * width
        bars = axis.bar(
            positions + offset,
            relative,
            width=width,
            color=[MODEL_COLORS[name] for name in ordered["name"]],
            edgecolor="white",
            linewidth=0.8,
            hatch=hatch,
            label=label,
        )
        for bar, value in zip(bars, raw, strict=True):
            text = f"{value:.2f}%" if unit == "%" else f"{value:,.0f} MW"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.035,
                text,
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=6,
                color="#0b0b0b",
            )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [MODEL_DISPLAY_NAMES[name] for name in ordered["name"]],
        rotation=25,
        ha="right",
    )
    axis.set_ylabel("Relative error (best model for each metric = 1.0)")
    axis.set_xlabel("Model, sorted by test MAPE")
    axis.set_title("Test error comparison across metrics\nLabels show raw values and units")
    axis.legend(loc="upper center", ncol=3)
    axis.grid(axis="x", visible=False)
    axis.set_ylim(0, axis.get_ylim()[1] * 1.22)
    figure.tight_layout()
    _save_figure(figure, "03_metric_comparison.png")


def _plot_per_day_mape(
    splits: DataSplits,
    results: list[ForecastResult],
    metrics: pd.DataFrame,
) -> None:
    order = metrics.sort_values("mape_test_pct")["name"].tolist()
    results_by_name = {result.name: result for result in results}
    matrix = np.vstack(
        [
            per_day_mape(splits.test, results_by_name[name].forecast).to_numpy(
                dtype=float
            )
            for name in order
        ]
    )

    figure, axis = plt.subplots(figsize=(6, 6))
    image = axis.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0)
    colourbar = figure.colorbar(image, ax=axis, shrink=0.82)
    colourbar.set_label("MAPE (%)")
    axis.set_xticks(np.arange(7))
    axis.set_xticklabels(
        [timestamp.strftime("%b %d") for timestamp in splits.test.index[::24]]
    )
    axis.set_yticks(np.arange(len(order)))
    axis.set_yticklabels([MODEL_DISPLAY_NAMES[name] for name in order])
    axis.set_xlabel("Test day (UTC)")
    axis.set_ylabel("Model, sorted by overall test MAPE")
    axis.set_title("Daily MAPE during the held-out week")
    threshold = float(np.nanmax(matrix)) * 0.55
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[row, column] > threshold else "#0b0b0b",
            )
    axis.grid(False)
    figure.tight_layout()
    _save_figure(figure, "04_per_day_mape.png")


def _plot_winner_intervals(
    splits: DataSplits,
    winner: ForecastResult,
    winner_metric: dict[str, Any],
    coverage_80: float,
    coverage_95: float,
) -> None:
    if any(
        series is None
        for series in [
            winner.lower_80,
            winner.upper_80,
            winner.lower_95,
            winner.upper_95,
        ]
    ):
        raise ValueError("The winning model does not provide both prediction intervals.")

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.fill_between(
        splits.test.index,
        winner.lower_95.to_numpy(dtype=float),
        winner.upper_95.to_numpy(dtype=float),
        color=MODEL_COLORS[winner.name],
        alpha=0.13,
        label="95% prediction interval",
    )
    axis.fill_between(
        splits.test.index,
        winner.lower_80.to_numpy(dtype=float),
        winner.upper_80.to_numpy(dtype=float),
        color=MODEL_COLORS[winner.name],
        alpha=0.27,
        label="80% prediction interval",
    )
    axis.plot(
        winner.forecast.index,
        winner.forecast.to_numpy(dtype=float),
        color=MODEL_COLORS[winner.name],
        label="Point forecast",
    )
    axis.plot(
        splits.test.index,
        splits.test.to_numpy(dtype=float),
        color="#0b0b0b",
        linewidth=1.8,
        label="Observed",
    )
    axis.set_title(f"Winning forecast with uncertainty: {MODEL_DISPLAY_NAMES[winner.name]}")
    axis.set_xlabel("Date (UTC)")
    axis.set_ylabel("Load (MW)")
    axis.legend(loc="upper right", ncol=2)
    _date_axis(axis)
    figure.text(
        0.5,
        0.01,
        f"Test MAPE {winner_metric['mape_test_pct']:.2f}% | "
        f"actual 80% coverage {coverage_80:.1%} | "
        f"actual 95% coverage {coverage_95:.1%}",
        ha="center",
        color="#52514e",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    _save_figure(figure, "05_winner_with_intervals.png")


def _style_boxplot(boxplot: dict[str, Any], colour: str) -> None:
    for box in boxplot["boxes"]:
        box.set(facecolor=colour, alpha=0.55, edgecolor="#52514e")
    for median in boxplot["medians"]:
        median.set(color="#0b0b0b", linewidth=1.3)
    for key in ["whiskers", "caps"]:
        for line in boxplot[key]:
            line.set(color="#52514e", linewidth=0.8)
    for flier in boxplot["fliers"]:
        flier.set(marker="o", markersize=2.5, alpha=0.45, markerfacecolor=colour)


def _plot_residuals(splits: DataSplits, winner: ForecastResult) -> None:
    residuals = splits.test - winner.forecast
    by_hour = [
        residuals.loc[residuals.index.hour == hour].to_numpy(dtype=float)
        for hour in range(24)
    ]
    days = list(residuals.index.floor("D").unique())
    by_day = [
        residuals.loc[residuals.index.floor("D") == day].to_numpy(dtype=float)
        for day in days
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11, 6))
    hour_boxplot = axes[0].boxplot(
        by_hour, positions=np.arange(24), widths=0.65, patch_artist=True
    )
    _style_boxplot(hour_boxplot, MODEL_COLORS[winner.name])
    axes[0].axhline(0.0, color="#898781", linewidth=1.0)
    axes[0].set_xticks(np.arange(0, 24, 3))
    axes[0].set_xticklabels([f"{hour:02d}:00" for hour in range(0, 24, 3)])
    axes[0].set_xlabel("Hour of day (UTC)")
    axes[0].set_ylabel("Residual, actual minus forecast (MW)")
    axes[0].set_title("Residuals by hour of day")

    day_boxplot = axes[1].boxplot(
        by_day, positions=np.arange(7), widths=0.65, patch_artist=True
    )
    _style_boxplot(day_boxplot, MODEL_COLORS[winner.name])
    axes[1].axhline(0.0, color="#898781", linewidth=1.0)
    axes[1].set_xticks(np.arange(7))
    axes[1].set_xticklabels(
        [pd.Timestamp(day).strftime("%a\n%b %d") for day in days]
    )
    axes[1].set_xlabel("Day in held-out test week (UTC)")
    axes[1].set_ylabel("Residual, actual minus forecast (MW)")
    axes[1].set_title("Residuals by test day")

    figure.suptitle(
        f"Where the winning {MODEL_DISPLAY_NAMES[winner.name]} forecast misses"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(figure, "06_residuals.png")


def _plot_feature_importance(result: ForecastResult) -> None:
    if result.feature_importance is None:
        raise ValueError("LightGBM feature importance is unavailable.")
    importance = result.feature_importance.sort_values(ascending=True)
    importance_pct = importance / importance.sum() * 100.0
    figure, axis = plt.subplots(figsize=(6, 6))
    bars = axis.barh(
        importance_pct.index,
        importance_pct.to_numpy(dtype=float),
        color=MODEL_COLORS["lightgbm"],
        edgecolor="white",
        linewidth=0.8,
    )
    axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=7)
    axis.set_xlabel("Gain importance (%)")
    axis.set_ylabel("Feature")
    axis.set_title("LightGBM feature importance after final refit")
    axis.grid(axis="y", visible=False)
    axis.set_xlim(0, max(importance_pct.max() * 1.15, 1.0))
    figure.tight_layout()
    _save_figure(figure, "07_feature_importance.png")


def _plot_prophet_decomposition(result: ForecastResult) -> None:
    if result.decomposition is None:
        raise ValueError("Prophet decomposition is unavailable.")
    figure, axes = plt.subplots(3, 1, figsize=(11, 6))
    specifications = [
        ("trend", "Trend", "Load level (MW)"),
        ("weekly", "Weekly component", "Contribution (MW)"),
        ("yearly", "Yearly component", "Contribution (MW)"),
    ]
    for axis, (component, title, ylabel) in zip(
        axes, specifications, strict=True
    ):
        subset = result.decomposition.loc[
            result.decomposition["component"] == component
        ]
        axis.plot(
            pd.to_datetime(subset["timestamp"]),
            subset["value"].to_numpy(dtype=float),
            color=MODEL_COLORS["prophet"],
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Date")
        _date_axis(axis)
    figure.suptitle("Prophet components from the final pre-test refit")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(figure, "08_decomposition.png")


def _render_figures(
    splits: DataSplits,
    results: list[ForecastResult],
    metrics: pd.DataFrame,
    coverage_80: float,
    coverage_95: float,
) -> None:
    configure_figure_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics_by_name = {
        row["name"]: row for row in metrics.to_dict(orient="records")
    }
    results_by_name = {result.name: result for result in results}
    winner_name = str(metrics.sort_values("mape_test_pct").iloc[0]["name"])
    winner = results_by_name[winner_name]

    _plot_overview(splits)
    _plot_forecast_comparison(splits, results, metrics_by_name)
    _plot_metric_comparison(metrics)
    _plot_per_day_mape(splits, results, metrics)
    _plot_winner_intervals(
        splits,
        winner,
        metrics_by_name[winner_name],
        coverage_80,
        coverage_95,
    )
    _plot_residuals(splits, winner)

    top_two = set(metrics.nsmallest(2, "mape_test_pct")["name"])
    if "lightgbm" in top_two:
        _plot_feature_importance(results_by_name["lightgbm"])
    if "prophet" in top_two:
        _plot_prophet_decomposition(results_by_name["prophet"])


def _build_metrics_payload(
    metrics: pd.DataFrame,
    winner: ForecastResult,
    actual: pd.Series,
    total_runtime_seconds: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    if set(winner.quantiles) != set(QUANTILES):
        raise ValueError(
            "The lowest-MAPE model lacks the probabilistic forecast needed by the study."
        )
    coverage_80 = interval_coverage(actual, winner.quantiles[0.1], winner.quantiles[0.9])
    coverage_95 = interval_coverage(
        actual, winner.quantiles[0.025], winner.quantiles[0.975]
    )
    probabilistic = {
        "winner_coverage_80pct": coverage_80,
        "winner_coverage_95pct": coverage_95,
        "winner_pinball_loss_q10": pinball_loss(actual, winner.quantiles[0.1], 0.1),
        "winner_pinball_loss_q50": pinball_loss(actual, winner.quantiles[0.5], 0.5),
        "winner_pinball_loss_q90": pinball_loss(actual, winner.quantiles[0.9], 0.9),
    }
    model_rows = metrics.loc[:, [
        "name",
        "mape_test_pct",
        "rmse_test_mw",
        "mae_test_mw",
        "mape_jan1_pct",
        "mape_jan2_to_jan7_pct",
        "runtime_seconds",
        "hyperparameters",
    ]].to_dict(orient="records")
    payload: dict[str, Any] = {
        "test_start": "2020-01-01",
        "test_end": "2020-01-07",
        "n_test_observations": 168,
        "models": model_rows,
        "winner": winner.name,
        **probabilistic,
        "total_runtime_seconds": total_runtime_seconds,
    }
    return payload, probabilistic


def _write_metrics_csv(
    metrics: pd.DataFrame,
    payload: dict[str, Any],
    probabilistic: dict[str, float],
) -> None:
    output = metrics.copy()
    output.insert(0, "test_start", payload["test_start"])
    output.insert(1, "test_end", payload["test_end"])
    output.insert(2, "n_test_observations", payload["n_test_observations"])
    output["winner"] = payload["winner"]
    output["is_winner"] = output["name"] == payload["winner"]
    for key, value in probabilistic.items():
        output[key] = np.where(output["is_winner"], value, np.nan)
    output["total_runtime_seconds"] = payload["total_runtime_seconds"]
    output["hyperparameters"] = output["hyperparameters"].map(
        lambda value: json.dumps(value, sort_keys=True, allow_nan=False)
    )
    output.to_csv(ROOT / "metrics.csv", index=False, float_format="%.6f")


def _results_markdown_table(metrics: pd.DataFrame) -> str:
    ordered = metrics.sort_values("mape_test_pct")
    rows = [
        "| Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2-7 MAPE (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in ordered.to_dict(orient="records"):
        rows.append(
            "| "
            + MODEL_DISPLAY_NAMES[record["name"]]
            + f" | {record['mape_test_pct']:.2f}"
            + f" | {record['rmse_test_mw']:,.0f}"
            + f" | {record['mae_test_mw']:,.0f}"
            + f" | {record['mape_jan1_pct']:.2f}"
            + f" | {record['mape_jan2_to_jan7_pct']:.2f} |"
        )
    return "\n".join(rows)


def _discussion(
    metrics: pd.DataFrame,
    winner: ForecastResult,
    probabilistic: dict[str, float],
    actual: pd.Series,
) -> str:
    ordered = metrics.sort_values("mape_test_pct").reset_index(drop=True)
    records = {record["name"]: record for record in metrics.to_dict(orient="records")}
    winner_record = records[winner.name]
    runner_up_name = str(ordered.iloc[1]["name"])
    runner_up = records[runner_up_name]
    rank_text = ", ".join(
        MODEL_DISPLAY_NAMES[name] for name in ordered["name"].tolist()
    )

    residuals = actual - winner.forecast
    daily_median_residual = residuals.groupby(residuals.index.floor("D")).median()
    largest_bias_day = daily_median_residual.abs().idxmax()
    largest_bias = float(daily_median_residual.loc[largest_bias_day])
    bias_direction = "overpredicted" if largest_bias < 0.0 else "underpredicted"

    segment_sentences: list[str] = []
    model_interpretations = {
        "naive": "Its weekly copy rule is strong when adjacent weeks resemble one another, but it cannot distinguish different holiday and working-day roles across weeks.",
        "sarima": "Its daily seasonal state smooths repeated cycles, but there is no explicit holiday signal and long validation forecasts tend to return toward learned average dynamics.",
        "prophet": "It includes an explicit German holiday calendar, but its Jan 1 error shows that a named-holiday flag alone did not reproduce the observed profile.",
        "lightgbm": "Its calendar, annual, weekly and rolling features give it several relevant anchors, although recursive use of its own predictions can compound an early error.",
        "nbeats": "Its one-week input can learn recurring load shapes, but the target-only network has no direct way to know that Jan 1 is a holiday.",
        "patchtst": "The TSMixer substitute also sees only one target-history week, so unusual calendar events must be inferred indirectly from load shape alone.",
    }
    for name in MODEL_ORDER:
        record = records[name]
        segment_sentences.append(
            f"{MODEL_DISPLAY_NAMES[name]} recorded {record['mape_jan1_pct']:.2f}% "
            f"MAPE on Jan 1 and {record['mape_jan2_to_jan7_pct']:.2f}% on Jan 2-7. "
            f"{model_interpretations[name]}"
        )

    return (
        f"{MODEL_DISPLAY_NAMES[winner.name]} won with {winner_record['mape_test_pct']:.2f}% "
        f"MAPE, compared with {runner_up['mape_test_pct']:.2f}% for "
        f"{MODEL_DISPLAY_NAMES[runner_up_name]}. Its nominal 80% and 95% intervals "
        f"covered {probabilistic['winner_coverage_80pct']:.1%} and "
        f"{probabilistic['winner_coverage_95pct']:.1%} of observations. The full ranking was "
        f"{rank_text}. The ranking is not a simple complexity ladder: this one-week test rewards "
        "models that reproduce the New Year transition, not models that are merely more flexible. "
        f"The winner {bias_direction} most strongly on {largest_bias_day.strftime('%a %b %d')}, "
        f"where its median residual was {largest_bias:,.0f} MW.\n\n"
        + " ".join(segment_sentences)
        + "\n\n"
        "These results are deliberately narrow. They cover one winter week, so they cannot establish "
        "annual average superiority or robustness during heat waves, industrial disruptions, or later "
        "pandemic conditions. No weather or market data were allowed, which isolates temporal structure "
        "but places a known ceiling on national load accuracy. Validation used one contiguous quarter, "
        "and the deep models were capped at short training schedules for a reproducible bake-off. "
        "The interval coverage estimates also use only 168 observations, so modest departures from the "
        "nominal rates should not be over-interpreted."
    )


def _write_transcript(
    splits: DataSplits,
    metrics: pd.DataFrame,
    winner: ForecastResult,
    probabilistic: dict[str, float],
    total_runtime_seconds: float,
) -> None:
    winner_name = MODEL_DISPLAY_NAMES[winner.name]
    text = f"""# German hourly load forecasting bake-off

## Data

The study uses German national hourly electricity load from Open Power System Data, derived from the ENTSO-E Transparency Platform. The file covers 2015-01-01 00:00 UTC through 2020-09-30 23:00 UTC and contains exactly 50,400 observations. Timestamps were verified as unique, increasing and complete at hourly frequency; load was finite with no missing values, so no imputation or row removal was needed. The only inputs were load, UTC calendar fields, and German federal-holiday dates knowable in advance. Holiday indicators use the UTC calendar date so they align exactly with the frozen evaluation days.

## Why these six models

The six models form a deliberate complexity gradient. Seasonal naive tests whether copying the same hour from the previous week is already enough. SARIMA represents classical stochastic dynamics with daily seasonality. Prophet adds smooth daily, weekly and yearly structure plus German holidays. LightGBM combines calendar variables with lag and rolling summaries. N-BEATS tests a target-only deep basis-expansion network. Darts 0.41.0 does not expose `PatchTSTModel`, so the sixth slot uses the prompt-approved `TSMixerModel` substitute, an all-MLP sequence mixer with the same 168-hour input and output chunks. This substitution is recorded rather than silently changing the model class.

## Validation strategy

The initial training period contains {len(splits.train):,} hours from 2015-01-01 through 2019-09-30. The separate validation period contains {len(splits.validation):,} hours from 2019-10-01 through 2019-12-31, and the untouched test contains {len(splits.test):,} hours from 2020-01-01 through 2020-01-07. SARIMA order, Prophet priors and LightGBM tree settings were selected by validation MAPE. The deep models used validation MAPE for early stopping and epoch selection. Every selected configuration was then refit from scratch on all {len(splits.train_validation):,} pre-test hours. LightGBM validation and test forecasts were recursive from the forecast origin, so no observed validation or test load leaked through 24-hour lags or rolling windows.

## Results table

{_results_markdown_table(metrics)}

The winning model's pinball losses were {probabilistic['winner_pinball_loss_q10']:,.1f} MW at q=0.1, {probabilistic['winner_pinball_loss_q50']:,.1f} MW at q=0.5, and {probabilistic['winner_pinball_loss_q90']:,.1f} MW at q=0.9. Nominal 80% and 95% prediction intervals achieved {probabilistic['winner_coverage_80pct']:.1%} and {probabilistic['winner_coverage_95pct']:.1%} empirical coverage.

## Discussion

{_discussion(metrics, winner, probabilistic, splits.test)}

## Recommendation

If exactly one of these models had to enter a JRC short-term load pipeline, this bake-off supports {winner_name}, because it produced the lowest error under the frozen comparison. Before deployment, I would run rolling-origin tests across at least a full year, calibrate intervals on multiple seasons, stress-test every German holiday separately, and monitor forecast drift by hour. I would then test weather covariates as a separate production extension because temperature is a major physical driver of national load; weather was intentionally excluded here to preserve the study's univariate question and should not be added without retaining this benchmark as an ablation.

## Reproducibility note

All stochastic code used seed {SEED}. Total wall-clock runtime for the successful model fits and output assembly was {total_runtime_seconds:.1f} seconds. Hardware and operating-system record: `{uname_output()}`. The environment used the pre-installed project virtual environment and installed no packages. The initial N-BEATS attempt made the MPS backend exit with code 139, so both final deep-model runs used CPU. Reproduce from this directory with `../../.venv/bin/python code/forecast.py --force`.
"""
    if chr(0x2014) in text:
        raise ValueError("The transcript contains a banned em dash.")
    (ROOT / "transcript.md").write_text(text, encoding="utf-8")


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load validated per-model caches when present.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore caches and rerun every model. This is the default study mode.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    if arguments.resume and arguments.force:
        raise ValueError("Choose either --resume or --force, not both.")

    study_started = time.perf_counter()
    set_random_seeds(SEED)
    splits = make_splits(load_data())
    results = _run_models(splits, resume=bool(arguments.resume and not arguments.force))
    model_metrics = [evaluate_forecast(splits.test, result) for result in results]
    metrics = _metrics_frame(model_metrics)
    results_by_name = {result.name: result for result in results}
    winner_name = str(metrics.sort_values("mape_test_pct").iloc[0]["name"])
    winner = results_by_name[winner_name]

    provisional_payload, provisional_probabilistic = _build_metrics_payload(
        metrics, winner, splits.test, time.perf_counter() - study_started
    )
    _render_figures(
        splits,
        results,
        metrics,
        provisional_probabilistic["winner_coverage_80pct"],
        provisional_probabilistic["winner_coverage_95pct"],
    )

    observed_process_runtime = time.perf_counter() - PROCESS_STARTED
    total_runtime_seconds = (
        sum(result.runtime_seconds for result in results) + observed_process_runtime
        if arguments.resume
        else observed_process_runtime
    )
    payload, probabilistic = _build_metrics_payload(
        metrics, winner, splits.test, total_runtime_seconds
    )
    write_json(ROOT / "metrics.json", payload)
    _write_metrics_csv(metrics, payload, probabilistic)
    _write_transcript(
        splits,
        metrics,
        winner,
        probabilistic,
        total_runtime_seconds,
    )
    print(
        f"Study complete. Winner: {winner.name}; "
        f"MAPE {mean_absolute_percentage_error(splits.test, winner.forecast):.3f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
