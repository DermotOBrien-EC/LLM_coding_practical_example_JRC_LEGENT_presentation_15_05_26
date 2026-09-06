from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from code.common import (
    DataSplits,
    ForecastResult,
    MODEL_COLORS,
    MODEL_LABELS,
    interval_coverage,
    per_day_mape,
    pinball_loss,
    score_result,
    set_figure_style,
)


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _metric_lookup(metric_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in metric_rows}


def _format_test_axis(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(mdates.DayLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axis.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))


def plot_overview(splits: DataSplits, figures_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(
        splits.full.index,
        splits.full.to_numpy(),
        color="#b8b8b8",
        linewidth=0.45,
        label="Observed load",
        rasterized=True,
    )
    axis.axvspan(
        splits.test.index[0],
        splits.test.index[-1] + pd.Timedelta(hours=1),
        color="#d03b3b",
        alpha=0.25,
        label="Held-out test week",
    )
    axis.plot(
        splits.test.index,
        splits.test.to_numpy(),
        color="#d03b3b",
        linewidth=1.4,
    )
    axis.set_title("German national hourly electricity load and held-out test window")
    axis.set_xlabel("Date (UTC)")
    axis.set_ylabel("Load (MW)")
    axis.legend(loc="upper right")
    figure.tight_layout()
    _save_figure(figure, figures_dir / "01_overview.png")


def plot_forecast_comparison(
    splits: DataSplits,
    results: dict[str, ForecastResult],
    metric_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    metrics = _metric_lookup(metric_rows)
    figure, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    actual_min = float(splits.test.min())
    actual_max = float(splits.test.max())
    forecast_min = min(float(result.forecast.min()) for result in results.values())
    forecast_max = max(float(result.forecast.max()) for result in results.values())
    lower = min(actual_min, forecast_min)
    upper = max(actual_max, forecast_max)
    padding = 0.05 * (upper - lower)

    for axis, name in zip(axes.flat, results, strict=True):
        result = results[name]
        axis.plot(
            splits.test.index,
            splits.test.to_numpy(),
            color="#0b0b0b",
            linewidth=1.7,
            label="Observed",
        )
        axis.plot(
            result.forecast.index,
            result.forecast.to_numpy(),
            color=MODEL_COLORS[name],
            linewidth=1.7,
            label="Forecast",
        )
        axis.set_title(f"{MODEL_LABELS[name]} | MAPE {metrics[name]['mape_test_pct']:.2f}%")
        axis.set_ylim(lower - padding, upper + padding)
        axis.legend(loc="lower right", fontsize=7)
        _format_test_axis(axis)
        axis.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(1, 32, 2)))

    for axis in axes[:, 0]:
        axis.set_ylabel("Load (MW)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Date (UTC)")
    figure.suptitle("Observed and forecast load during the held-out test week", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    _save_figure(figure, figures_dir / "02_forecast_comparison.png")


def plot_metric_comparison(
    metric_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    sorted_rows = sorted(metric_rows, key=lambda row: row["mape_test_pct"])
    names = [row["name"] for row in sorted_rows]
    metric_specs = [
        ("mape_test_pct", "MAPE", "%", "//"),
        ("rmse_test_mw", "RMSE", "MW", "\\\\"),
        ("mae_test_mw", "MAE", "MW", ".."),
    ]
    x_positions = np.arange(len(names), dtype=float)
    width = 0.24
    maximum_ratio = max(
        max(float(row[key]) for row in sorted_rows) / min(float(row[key]) for row in sorted_rows)
        for key, _, _, _ in metric_specs
    )
    figure, axis = plt.subplots(figsize=(6, 6))

    for offset_index, (key, label, unit, hatch) in enumerate(metric_specs):
        raw_values = np.asarray([row[key] for row in sorted_rows], dtype=float)
        normalized = raw_values / raw_values.min()
        offset = (offset_index - 1) * width
        bars = axis.bar(
            x_positions + offset,
            normalized,
            width=width - 0.025,
            color=[MODEL_COLORS[name] for name in names],
            edgecolor="white",
            linewidth=0.8,
            hatch=hatch,
            label=label,
        )
        for bar, raw_value in zip(bars, raw_values, strict=True):
            raw_label = f"{raw_value:.2f}%" if unit == "%" else f"{raw_value:,.0f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.035,
                raw_label,
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=6.5,
                color="#52514e",
            )

    legend_handles = [
        Patch(facecolor="#d8d8d8", edgecolor="#898781", hatch=hatch, label=label)
        for _, label, _, hatch in metric_specs
    ]
    axis.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        title="Metric",
    )
    axis.set_xticks(x_positions, [MODEL_LABELS[name] for name in names], rotation=28, ha="right")
    axis.set_ylabel("Ratio to the best model for each metric")
    axis.set_xlabel("Model, sorted by test MAPE")
    axis.set_title(
        "Test metrics normalized for a common axis\n(raw values printed above bars)",
        pad=68,
    )
    axis.set_ylim(0.0, max(maximum_ratio * 1.18, 1.35))
    figure.tight_layout()
    _save_figure(figure, figures_dir / "03_metric_comparison.png")


def plot_per_day_mape(
    splits: DataSplits,
    results: dict[str, ForecastResult],
    metric_rows: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    names = [row["name"] for row in sorted(metric_rows, key=lambda row: row["mape_test_pct"])]
    daily = np.vstack(
        [per_day_mape(splits.test, results[name].forecast).to_numpy() for name in names]
    )
    figure, axis = plt.subplots(figsize=(6, 6))
    image = axis.imshow(daily, aspect="auto", cmap="Blues", vmin=0.0)
    day_labels = [
        pd.Timestamp(day).strftime("%b %d")
        for day in per_day_mape(splits.test, results[names[0]].forecast).index
    ]
    axis.set_xticks(np.arange(7), day_labels)
    axis.set_yticks(np.arange(len(names)), [MODEL_LABELS[name] for name in names])
    axis.set_xlabel("Test date (UTC)")
    axis.set_ylabel("Model, sorted by weekly MAPE")
    axis.set_title("Daily mean absolute percentage error")
    axis.grid(False)

    threshold = float(np.nanmax(daily)) * 0.55
    for row_index in range(daily.shape[0]):
        for column_index in range(daily.shape[1]):
            value = daily[row_index, column_index]
            text_color = "white" if value > threshold else "#0b0b0b"
            axis.text(
                column_index,
                row_index,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )
    axis.add_patch(
        Rectangle(
            (-0.5, -0.5),
            1.0,
            len(names),
            fill=False,
            edgecolor="#d03b3b",
            linewidth=2.0,
        )
    )
    axis.text(0, -0.78, "Public holiday", ha="center", va="bottom", color="#d03b3b", fontsize=8)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.04)
    colorbar.set_label("MAPE (%)")
    figure.tight_layout()
    _save_figure(figure, figures_dir / "04_per_day_mape.png")


def plot_winner_intervals(
    splits: DataSplits,
    winner: ForecastResult,
    winner_metrics: dict[str, float],
    figures_dir: Path,
) -> None:
    required = {0.025, 0.1, 0.5, 0.9, 0.975}
    if not required.issubset(winner.quantiles):
        raise ValueError(f"Winning model {winner.name} does not provide all required quantiles")
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.fill_between(
        splits.test.index,
        winner.quantiles[0.025].to_numpy(),
        winner.quantiles[0.975].to_numpy(),
        color=MODEL_COLORS[winner.name],
        alpha=0.12,
        label="95% prediction interval",
        linewidth=0,
    )
    axis.fill_between(
        splits.test.index,
        winner.quantiles[0.1].to_numpy(),
        winner.quantiles[0.9].to_numpy(),
        color=MODEL_COLORS[winner.name],
        alpha=0.28,
        label="80% prediction interval",
        linewidth=0,
    )
    axis.plot(
        splits.test.index,
        splits.test.to_numpy(),
        color="#0b0b0b",
        linewidth=1.8,
        label="Observed",
    )
    axis.plot(
        winner.forecast.index,
        winner.forecast.to_numpy(),
        color=MODEL_COLORS[winner.name],
        linewidth=1.8,
        label="Point forecast",
    )
    axis.set_title(
        f"{MODEL_LABELS[winner.name]}: test MAPE {winner_metrics['mape_test_pct']:.2f}%, "
        f"80% coverage {winner_metrics['coverage_80']:.1%}, "
        f"95% coverage {winner_metrics['coverage_95']:.1%}"
    )
    axis.set_xlabel("Date (UTC)")
    axis.set_ylabel("Load (MW)")
    _format_test_axis(axis)
    axis.legend(loc="upper left", ncol=2)
    figure.tight_layout()
    _save_figure(figure, figures_dir / "05_winner_with_intervals.png")


def plot_residuals(
    splits: DataSplits,
    winner: ForecastResult,
    figures_dir: Path,
) -> None:
    residuals = splits.test - winner.forecast
    by_hour = [residuals[residuals.index.hour == hour].to_numpy() for hour in range(24)]
    unique_dates = list(dict.fromkeys(residuals.index.date))
    by_day = [residuals[residuals.index.date == date].to_numpy() for date in unique_dates]
    day_labels = [pd.Timestamp(date).strftime("%a\n%b %d") for date in unique_dates]

    figure, axes = plt.subplots(1, 2, figsize=(11, 6))
    first = axes[0].boxplot(
        by_hour,
        positions=np.arange(24),
        widths=0.65,
        patch_artist=True,
        showfliers=False,
    )
    for box in first["boxes"]:
        box.set_facecolor(MODEL_COLORS[winner.name])
        box.set_alpha(0.65)
    axes[0].axhline(0.0, color="#52514e", linewidth=1.0)
    axes[0].set_xticks(np.arange(0, 24, 3), [f"{hour:02d}:00" for hour in range(0, 24, 3)])
    axes[0].set_xlabel("Hour of day (UTC)")
    axes[0].set_ylabel("Residual, actual minus forecast (MW)")
    axes[0].set_title("Residual distribution by hour of day")

    second = axes[1].boxplot(
        by_day,
        positions=np.arange(7),
        widths=0.65,
        patch_artist=True,
        showfliers=False,
    )
    for box in second["boxes"]:
        box.set_facecolor(MODEL_COLORS[winner.name])
        box.set_alpha(0.65)
    axes[1].axhline(0.0, color="#52514e", linewidth=1.0)
    axes[1].set_xticks(np.arange(7), day_labels)
    axes[1].set_xlabel("Day in held-out test week (UTC)")
    axes[1].set_ylabel("Residual, actual minus forecast (MW)")
    axes[1].set_title("Residual distribution by test day")
    figure.suptitle(f"Where {MODEL_LABELS[winner.name]} overpredicts and underpredicts", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    _save_figure(figure, figures_dir / "06_residuals.png")


def plot_feature_importance(result: ForecastResult, figures_dir: Path) -> None:
    if result.feature_importance is None:
        raise ValueError("LightGBM feature importance was not retained")
    importance = result.feature_importance.sort_values(ascending=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.barh(
        importance.index,
        importance.to_numpy(),
        color=MODEL_COLORS["lightgbm"],
        edgecolor="white",
    )
    axis.set_xlabel("Share of total split gain (%)")
    axis.set_ylabel("Feature")
    axis.set_title("LightGBM feature importance by gain")
    for position, value in enumerate(importance.to_numpy()):
        axis.text(value + 0.2, position, f"{value:.1f}%", va="center", fontsize=8)
    figure.tight_layout()
    _save_figure(figure, figures_dir / "07_feature_importance.png")


def plot_prophet_decomposition(result: ForecastResult, figures_dir: Path) -> None:
    if result.prophet_components is None:
        raise ValueError("Prophet components were not retained")
    components = result.prophet_components
    figure, axes = plt.subplots(3, 1, figsize=(11, 6))
    axes[0].plot(
        components.index, components["trend"], color=MODEL_COLORS["prophet"], linewidth=1.3
    )
    axes[0].set_title("Trend")
    axes[0].set_ylabel("Component (MW)")

    test_week = components.loc["2020-01-01":"2020-01-07"]
    axes[1].plot(test_week.index, test_week["weekly"], color=MODEL_COLORS["prophet"], linewidth=1.4)
    axes[1].set_title("Weekly component during the test week")
    axes[1].set_ylabel("Component (MW)")
    axes[1].xaxis.set_major_locator(mdates.DayLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    axes[2].plot(
        components.index, components["yearly"], color=MODEL_COLORS["prophet"], linewidth=1.3
    )
    axes[2].set_title("Yearly component")
    axes[2].set_xlabel("Date (UTC)")
    axes[2].set_ylabel("Component (MW)")
    figure.suptitle("Prophet decomposition from the final fitted model", y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    _save_figure(figure, figures_dir / "08_decomposition.png")


def _markdown_results_table(metric_rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Rank | Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2 to 7 MAPE (%) | Runtime (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(
        sorted(metric_rows, key=lambda item: item["mape_test_pct"]), start=1
    ):
        lines.append(
            f"| {rank} | {MODEL_LABELS[row['name']]} | {row['mape_test_pct']:.3f} | "
            f"{row['rmse_test_mw']:.1f} | {row['mae_test_mw']:.1f} | "
            f"{row['mape_jan1_pct']:.3f} | {row['mape_jan2_to_jan7_pct']:.3f} | "
            f"{row['runtime_seconds']:.1f} |"
        )
    return "\n".join(lines)


def _discussion(metric_rows: list[dict[str, Any]], winner_metrics: dict[str, float]) -> str:
    sorted_rows = sorted(metric_rows, key=lambda row: row["mape_test_pct"])
    lookup = _metric_lookup(metric_rows)
    winner = sorted_rows[0]
    runner_up = sorted_rows[1]
    naive = lookup["naive"]
    improvement = (
        (naive["mape_test_pct"] - winner["mape_test_pct"]) / naive["mape_test_pct"] * 100.0
    )
    jan1_best = min(metric_rows, key=lambda row: row["mape_jan1_pct"])
    working_best = min(metric_rows, key=lambda row: row["mape_jan2_to_jan7_pct"])

    return (
        f"{MODEL_LABELS[winner['name']]} won the held-out week with a MAPE of "
        f"{winner['mape_test_pct']:.2f}%, ahead of {MODEL_LABELS[runner_up['name']]} at "
        f"{runner_up['mape_test_pct']:.2f}%. Its MAPE was {improvement:.1f}% lower than the "
        f"seasonal-naive anchor. {MODEL_LABELS[jan1_best['name']]} handled New Year's Day "
        f"best ({jan1_best['mape_jan1_pct']:.2f}% MAPE) and "
        f"{MODEL_LABELS[working_best['name']]} was also best over Jan 2 to Jan 7 "
        f"({working_best['mape_jan2_to_jan7_pct']:.2f}%). Prophet's explicit German holiday "
        f"calendar did not guarantee good holiday performance: its Jan 1 MAPE was "
        f"{lookup['prophet']['mape_jan1_pct']:.2f}%. N-BEATS and TSMixer failed most sharply on "
        f"Jan 1, with MAPE of {lookup['nbeats']['mape_jan1_pct']:.2f}% and "
        f"{lookup['patchtst']['mape_jan1_pct']:.2f}% respectively.\n\n"
        f"The seasonal-naive model copied the preceding week's shape exactly, so it could not adjust "
        f"for differences between Christmas week and the first week of January. SARIMA represented "
        f"daily autocorrelation and differencing but had no explicit holiday switch. Both happened to "
        f"track Jan 1 more closely than the later days, then degraded to "
        f"{naive['mape_jan2_to_jan7_pct']:.2f}% and "
        f"{lookup['sarima']['mape_jan2_to_jan7_pct']:.2f}% MAPE from Jan 2 to Jan 7. Prophet had "
        f"daily, weekly, yearly, and holiday components, but its smooth additive or multiplicative "
        f"structure did not capture the New Year's Day level shift well in this test. LightGBM could "
        f"combine the holiday flag with annual, weekly, and daily lags and rolling summaries, although "
        f"recursive forecasting allowed its own errors to feed later lag and rolling features. "
        f"N-BEATS and TSMixer learned week-sized mappings directly from the univariate series. Their "
        f"results reflect a single random seed and deliberately short, validation-monitored, capped "
        f"training, as well as model class.\n\n"
        f"LightGBM's nominal 80% interval covered {winner_metrics['coverage_80']:.1%} of observations, "
        f"while its nominal 95% interval covered {winner_metrics['coverage_95']:.1%}. Covering every "
        f"observation suggests that the 95% interval was conservative or overwide for this single "
        f"week, not that its calibration is proven. The test window contains only one week and one "
        f"federal holiday, so it cannot establish year-round robustness, rare-event performance, or "
        f"interval calibration across seasons."
    )


def _hardware_description() -> str:
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        memory_bytes = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return f"{chip} with {memory_bytes / 1024**3:.0f} GiB RAM"
    except (OSError, subprocess.CalledProcessError, ValueError):
        return platform.machine()


def write_transcript(
    path: Path,
    splits: DataSplits,
    metric_rows: list[dict[str, Any]],
    winner: ForecastResult,
    winner_metrics: dict[str, float],
    total_runtime_seconds: float,
) -> None:
    uname_output = subprocess.run(
        ["uname", "-a"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    table = _markdown_results_table(metric_rows)
    discussion = _discussion(metric_rows, winner_metrics)
    recommendation = (
        f"If exactly one of these models had to enter a JRC short-term load pipeline, I would take "
        f"{MODEL_LABELS[winner.name]}, the model that won the untouched test week. Before production, "
        f"I would repeat the comparison with rolling-origin tests across all seasons and several years, "
        f"calibrate its prediction intervals on multiple validation windows, monitor holiday and "
        f"daylight-saving transitions separately, and then test whether adding operationally available "
        f"weather forecasts improves accuracy without weakening data governance."
    )
    text = f"""# German hourly load forecasting bake-off

## Data

The study uses German national hourly load from Open Power System Data, derived from the ENTSO-E Transparency Platform. The file covers 2015-01-01 00:00 UTC through 2020-09-30 23:00 UTC and contains exactly 50,400 hourly observations in megawatts. The timestamp index is unique and continuous, and the load column has no missing values, so no rows were dropped and no imputation was needed.

## Why these six models

The six models form a deliberate complexity gradient. Seasonal naive tests whether a model can beat a one-week copy. SARIMA adds classical autoregressive and daily seasonal structure. Prophet adds smooth daily, weekly, and yearly components plus German public holidays. LightGBM adds nonlinear interactions among calendar fields, lags, and prior-only rolling summaries. N-BEATS learns a direct week-to-week mapping with a deep multilayer perceptron. Darts 0.41.0 does not expose `PatchTSTModel`, so the sixth slot uses the prompt-approved `TSMixerModel` substitute, another global neural forecaster with the same 168-hour input and output chunks. No weather, price, fuel, or other external series was used.

## Validation strategy

The initial training window contains {len(splits.train):,} hours from 2015-01-01 through 2019-09-30. The validation window contains {len(splits.validation):,} hours from 2019-10-01 through 2019-12-31, and the untouched test contains {len(splits.test)} hours from 2020-01-01 through 2020-01-07. SARIMA candidates used explicit 24-hour differencing and Burg estimates for AR orders 1, 24, and 168, equivalent to `(p,0,0)(0,1,0,24)` models; Prophet regularization and seasonality mode, and LightGBM tree settings were also selected by validation MAPE. N-BEATS and TSMixer used validation-MAPE monitoring under capped training, with a maximum of 8 and 15 epochs respectively; both selected the maximum permitted epoch count. Each selected configuration was then rebuilt and refit on all {len(splits.train_validation):,} pre-test hours. LightGBM validation and test forecasts were recursive: after the forecast origin, lag and rolling features used the model's own earlier predictions rather than held-out actual values. This keeps the 168-hour test genuinely untouched.

## Results

{table}

The winning probabilistic forecast had actual coverage of {winner_metrics["coverage_80"]:.1%} for the nominal 80% interval and {winner_metrics["coverage_95"]:.1%} for the nominal 95% interval. Its pinball losses were {winner_metrics["pinball_q10"]:.1f} MW at q=0.1, {winner_metrics["pinball_q50"]:.1f} MW at q=0.5, and {winner_metrics["pinball_q90"]:.1f} MW at q=0.9.

## Discussion

{discussion}

## Recommendation

{recommendation}

## Reproducibility note

All stochastic models and sampling calls used random seed 2020. N-BEATS used one training window every 336 hours and TSMixer used one every 168 hours to keep the default 168-hour architectures reproducible within the time budget while still spanning the full training history. The complete run took {total_runtime_seconds:.1f} seconds on {_hardware_description()}. System record: `{uname_output}`. Run from this directory with `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ../../.venv/bin/python code/forecast.py`. The study uses only `opsd_de_load.csv` plus deterministic timestamp-derived German federal holiday indicators from the installed `holidays` package.
"""
    path.write_text(text, encoding="utf-8")


def write_outputs(
    root: Path,
    splits: DataSplits,
    ordered_results: list[ForecastResult],
    total_runtime_seconds: float,
) -> dict[str, Any]:
    set_figure_style()
    figures_dir = root / "figures"
    figures_dir.mkdir(exist_ok=True)
    results = {result.name: result for result in ordered_results}
    metric_rows = [score_result(result, splits.test) for result in ordered_results]
    sorted_rows = sorted(metric_rows, key=lambda row: row["mape_test_pct"])
    winner_name = sorted_rows[0]["name"]
    winner = results[winner_name]
    required_quantiles = {0.025, 0.1, 0.5, 0.9, 0.975}
    if not required_quantiles.issubset(winner.quantiles):
        raise ValueError(
            f"The winning model {winner_name} lacks the required probabilistic forecast"
        )

    coverage_80 = interval_coverage(
        splits.test,
        winner.quantiles[0.1],
        winner.quantiles[0.9],
    )
    coverage_95 = interval_coverage(
        splits.test,
        winner.quantiles[0.025],
        winner.quantiles[0.975],
    )
    pinball_q10 = pinball_loss(splits.test, winner.quantiles[0.1], 0.1)
    pinball_q50 = pinball_loss(splits.test, winner.quantiles[0.5], 0.5)
    pinball_q90 = pinball_loss(splits.test, winner.quantiles[0.9], 0.9)
    winner_metrics = {
        **sorted_rows[0],
        "coverage_80": coverage_80,
        "coverage_95": coverage_95,
        "pinball_q10": pinball_q10,
        "pinball_q50": pinball_q50,
        "pinball_q90": pinball_q90,
    }

    payload = {
        "test_start": "2020-01-01",
        "test_end": "2020-01-07",
        "n_test_observations": 168,
        "models": metric_rows,
        "winner": winner_name,
        "winner_coverage_80pct": coverage_80,
        "winner_coverage_95pct": coverage_95,
        "winner_pinball_loss_q10": pinball_q10,
        "winner_pinball_loss_q50": pinball_q50,
        "winner_pinball_loss_q90": pinball_q90,
        "total_runtime_seconds": float(total_runtime_seconds),
    }
    (root / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    flattened_rows: list[dict[str, Any]] = []
    for row in metric_rows:
        flattened_rows.append(
            {
                "test_start": payload["test_start"],
                "test_end": payload["test_end"],
                "n_test_observations": payload["n_test_observations"],
                **{key: value for key, value in row.items() if key != "hyperparameters"},
                "hyperparameters": json.dumps(row["hyperparameters"], sort_keys=True),
                "study_winner": winner_name,
                "is_winner": row["name"] == winner_name,
                "winner_coverage_80pct": coverage_80,
                "winner_coverage_95pct": coverage_95,
                "winner_pinball_loss_q10": pinball_q10,
                "winner_pinball_loss_q50": pinball_q50,
                "winner_pinball_loss_q90": pinball_q90,
                "total_runtime_seconds": total_runtime_seconds,
            }
        )
    pd.DataFrame(flattened_rows).to_csv(root / "metrics.csv", index=False)

    plot_overview(splits, figures_dir)
    plot_forecast_comparison(splits, results, metric_rows, figures_dir)
    plot_metric_comparison(metric_rows, figures_dir)
    plot_per_day_mape(splits, results, metric_rows, figures_dir)
    plot_winner_intervals(splits, winner, winner_metrics, figures_dir)
    plot_residuals(splits, winner, figures_dir)

    top_two = {row["name"] for row in sorted_rows[:2]}
    feature_path = figures_dir / "07_feature_importance.png"
    if "lightgbm" in top_two:
        plot_feature_importance(results["lightgbm"], figures_dir)
    elif feature_path.exists():
        feature_path.unlink()
    decomposition_path = figures_dir / "08_decomposition.png"
    if "prophet" in top_two:
        plot_prophet_decomposition(results["prophet"], figures_dir)
    elif decomposition_path.exists():
        decomposition_path.unlink()

    write_transcript(
        root / "transcript.md",
        splits,
        metric_rows,
        winner,
        winner_metrics,
        total_runtime_seconds,
    )
    return payload
