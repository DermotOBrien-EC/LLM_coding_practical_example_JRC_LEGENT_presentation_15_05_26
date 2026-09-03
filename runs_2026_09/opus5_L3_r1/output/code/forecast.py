"""Orchestrator: run the six models, score them, draw the figures.

Each model lives in its own file and hands back a ForecastResult. This file
does not know anything about how a model works. It only knows how to ask for
a forecast, how to score one, and how to draw the comparison.

Results are cached under artifacts/ so figures can be redrawn without
refitting anything. Pass --refit to force a model to be fitted again.

    python code/forecast.py                 # run everything (uses the cache)
    python code/forecast.py --refit all     # fit everything from scratch
    python code/forecast.py --refit sarima  # refit just one model
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as c  # noqa: E402
import lightgbm_features  # noqa: E402
import naive  # noqa: E402
import nbeats  # noqa: E402
import patchtst  # noqa: E402
import sarima  # noqa: E402


def _load_prophet_module() -> Any:
    """Load code/prophet.py without letting it shadow the Prophet package.

    The study layout asks for a file per model, so this file has to be called
    prophet.py, which collides with the pip-installed package of the same
    name. Loading it under a different module name keeps both reachable.
    """
    path = Path(__file__).resolve().parent / "prophet.py"
    spec = importlib.util.spec_from_file_location("prophet_bakeoff", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNERS: dict[str, Callable[[pd.Series], c.ForecastResult]] = {
    "naive": naive.run,
    "sarima": sarima.run,
    "prophet": lambda series: _load_prophet_module().run(series),
    "lightgbm": lightgbm_features.run,
    "nbeats": nbeats.run,
    "patchtst": patchtst.run,
}


# ---------------------------------------------------------------------------
# Running the bake-off
# ---------------------------------------------------------------------------


def gather_results(series: pd.Series, refit: set[str]) -> dict[str, c.ForecastResult]:
    """Fit whatever is not cached, then return every model's forecast."""
    results: dict[str, c.ForecastResult] = {}
    for name in c.MODEL_NAMES:
        if name not in refit and c.ForecastResult.exists(name):
            print(f"[{name}] loading cached result", flush=True)
            results[name] = c.ForecastResult.load(name)
            continue
        print(f"[{name}] fitting", flush=True)
        result = RUNNERS[name](series)
        result.save()
        results[name] = result
    return results


def score_everything(series: pd.Series, results: dict[str, c.ForecastResult]) -> pd.DataFrame:
    """One row per model: the headline errors and the holiday breakdown."""
    test = series.loc[c.TEST_START : c.TEST_END]
    rows: list[dict[str, Any]] = []
    for name in c.MODEL_NAMES:
        result = results[name]
        scores = c.evaluate_point_forecast(test, result.point_forecast)
        rows.append(
            {
                "name": name,
                "label": c.MODEL_LABELS[name],
                **scores,
                "validation_mape_pct": result.validation_mape_pct,
                "runtime_seconds": result.runtime_seconds,
                "has_intervals": result.quantile_forecast is not None,
                **{f"mape_{day}": value for day, value in c.per_day_mape(test, result.point_forecast).items()},
            }
        )
    frame = pd.DataFrame(rows).set_index("name")
    return frame.sort_values("mape_test_pct")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_overview(series: pd.Series) -> None:
    """The whole series, with the test week marked."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(series.index, series.to_numpy(), color="0.72", linewidth=0.35, label="Observed load, 2015-2020")

    test = series.loc[c.TEST_START : c.TEST_END]
    ax.plot(test.index, test.to_numpy(), color="tab:red", linewidth=1.4, label="Test window (2020-01-01 to 2020-01-07)")
    ax.axvspan(c.TEST_START, c.TEST_END, color="tab:red", alpha=0.18)

    ax.annotate(
        "168-hour\nheld-out week",
        xy=(c.TEST_START, float(test.max())),
        xytext=(c.TEST_START - pd.Timedelta(days=330), 74000),
        color="tab:red",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "tab:red", "linewidth": 1.0},
    )

    ax.set_title("German hourly electricity load, and the week held out for testing")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (MW)")
    ax.set_xlim(series.index[0], series.index[-1])
    ax.set_ylim(float(series.min()) - 3500, float(series.max()) + 4500)
    ax.legend(loc="upper left", ncols=1)
    fig.savefig(c.FIGURE_DIR / "01_overview.png")
    plt.close(fig)


def figure_forecast_comparison(series: pd.Series, results: dict[str, c.ForecastResult], scores: pd.DataFrame) -> None:
    """Six small panels, one per model, all on the same axes."""
    test = series.loc[c.TEST_START : c.TEST_END]
    order = list(scores.index)

    all_values = np.concatenate([test.to_numpy()] + [results[n].point_forecast.to_numpy() for n in order])
    pad = 0.05 * (all_values.max() - all_values.min())
    ylim = (all_values.min() - pad, all_values.max() + pad)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), order):
        ax.plot(test.index, test.to_numpy(), color=c.OBSERVED_COLOR, linewidth=1.3, label="Observed")
        ax.plot(
            results[name].point_forecast.index,
            results[name].point_forecast.to_numpy(),
            color=c.MODEL_COLORS[name],
            linewidth=1.5,
            label="Forecast",
        )
        ax.axvspan(c.TEST_START, c.TEST_START + pd.Timedelta(hours=24), color="0.85", alpha=0.6, zorder=0)
        ax.set_title(f"{c.MODEL_LABELS[name]} - MAPE {scores.loc[name, 'mape_test_pct']:.2f}%")
        ax.set_ylim(*ylim)
        ax.legend(loc="lower right")

    for ax in axes[-1]:
        ax.set_xlabel("Time (UTC)")
        ax.tick_params(axis="x", rotation=30)
    for ax in axes[:, 0]:
        ax.set_ylabel("Load (MW)")

    fig.suptitle(
        "Test-week forecasts against observed load (grey band = 1 January, a German public holiday)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(c.FIGURE_DIR / "02_forecast_comparison.png")
    plt.close(fig)


def figure_metric_comparison(scores: pd.DataFrame) -> None:
    """MAPE, RMSE and MAE side by side.

    The three metrics live on different scales (percent against megawatts),
    so they get one panel each rather than one set of grouped bars, where the
    percentages would be invisible next to four-figure megawatt values.
    """
    metrics = [
        ("mape_test_pct", "MAPE (%)", "{:.2f}"),
        ("rmse_test_mw", "RMSE (MW)", "{:.0f}"),
        ("mae_test_mw", "MAE (MW)", "{:.0f}"),
    ]
    order = list(scores.index)
    colors = [c.MODEL_COLORS[n] for n in order]
    labels = [c.MODEL_LABELS[n] for n in order]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, (column, title, fmt) in zip(axes, metrics):
        values = scores[column].to_numpy()
        bars = ax.bar(range(len(order)), values, color=colors, edgecolor="white", linewidth=0.6)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02 * values.max(),
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_ylim(0, values.max() * 1.18)

    fig.suptitle("Test-window accuracy, models ordered by MAPE (lower is better)", fontsize=12)
    fig.tight_layout()
    fig.savefig(c.FIGURE_DIR / "03_metric_comparison.png")
    plt.close(fig)


def figure_per_day_mape(scores: pd.DataFrame) -> None:
    """Where in the week each model goes wrong."""
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(c.TEST_START, c.TEST_END, freq="D")]
    order = list(scores.index)
    matrix = np.array([[scores.loc[n, f"mape_{d}"] for d in days] for n in order])

    day_labels = [
        f"{pd.Timestamp(d).strftime('%a %d %b')}" + ("\n(holiday)" if d == "2020-01-01" else "")
        for d in days
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(day_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([c.MODEL_LABELS[n] for n in order])
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if matrix[i, j] < 0.65 * matrix.max() else "white",
            )
    bar = fig.colorbar(image, ax=ax)
    bar.set_label("MAPE (%)")
    ax.set_title("Daily MAPE by model")

    ax = axes[1]
    width = 0.13
    positions = np.arange(len(days))
    for k, name in enumerate(order):
        ax.bar(
            positions + (k - (len(order) - 1) / 2) * width,
            matrix[k],
            width=width,
            color=c.MODEL_COLORS[name],
            label=c.MODEL_LABELS[name],
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(day_labels, rotation=30, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("Day of the test week (UTC)")
    ax.set_title("The same numbers as bars")
    ax.set_ylim(0, float(matrix.max()) * 1.42)
    ax.legend(ncols=3, loc="upper center", fontsize=8)

    fig.suptitle("Per-day error: 1 January is a public holiday and separates the models", fontsize=12)
    fig.tight_layout()
    fig.savefig(c.FIGURE_DIR / "04_per_day_mape.png")
    plt.close(fig)


def figure_winner_with_intervals(
    series: pd.Series, result: c.ForecastResult, scores: pd.DataFrame, probabilistic: dict[str, float]
) -> None:
    """The winning model's forecast with its 80% and 95% bands."""
    test = series.loc[c.TEST_START : c.TEST_END]
    quantiles = result.quantile_forecast
    assert quantiles is not None
    color = c.MODEL_COLORS[result.name]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(
        test.index,
        quantiles["0.025"],
        quantiles["0.975"],
        color=color,
        alpha=0.15,
        linewidth=0,
        label="95% prediction interval",
    )
    ax.fill_between(
        test.index,
        quantiles["0.1"],
        quantiles["0.9"],
        color=color,
        alpha=0.30,
        linewidth=0,
        label="80% prediction interval",
    )
    ax.plot(test.index, result.point_forecast.to_numpy(), color=color, linewidth=1.8, label="Point forecast")
    ax.plot(test.index, test.to_numpy(), color=c.OBSERVED_COLOR, linewidth=1.3, label="Observed")
    ax.axvspan(c.TEST_START, c.TEST_START + pd.Timedelta(hours=24), color="0.85", alpha=0.6, zorder=0)

    ax.set_title(
        f"{c.MODEL_LABELS[result.name]}: test MAPE {scores.loc[result.name, 'mape_test_pct']:.2f}%, "
        f"80% interval covered {probabilistic['coverage_80pct'] * 100:.1f}% of hours "
        f"(nominal 80%), 95% interval covered {probabilistic['coverage_95pct'] * 100:.1f}% (nominal 95%)",
        fontsize=10.5,
    )
    ax.set_xlabel("Time (UTC); grey band is 1 January, a German public holiday")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="lower right", ncols=2)
    fig.savefig(c.FIGURE_DIR / "05_winner_with_intervals.png")
    plt.close(fig)


def figure_residuals(series: pd.Series, result: c.ForecastResult) -> None:
    """Where the winner is systematically too high or too low."""
    test = series.loc[c.TEST_START : c.TEST_END]
    residual = test.to_numpy() - result.point_forecast.to_numpy()
    color = c.MODEL_COLORS[result.name]

    local_hour = np.asarray(test.index.tz_convert(c.LOCAL_TZ).hour)
    day_key = [ts.strftime("%Y-%m-%d") for ts in test.index]
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(c.TEST_START, c.TEST_END, freq="D")]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    data = [residual[local_hour == h] for h in range(24)]
    box = ax.boxplot(data, positions=range(24), widths=0.65, patch_artist=True, medianprops={"color": "black"})
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([str(h) for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of day (Europe/Berlin local time)")
    ax.set_ylabel("Residual, observed minus forecast (MW)")
    ax.set_title("By time of day")

    ax = axes[1]
    data = [residual[np.array(day_key) == d] for d in days]
    box = ax.boxplot(data, positions=range(len(days)), widths=0.6, patch_artist=True, medianprops={"color": "black"})
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(
        [pd.Timestamp(d).strftime("%a\n%d %b") + ("\n(holiday)" if d == "2020-01-01" else "") for d in days],
        fontsize=8,
    )
    ax.set_xlabel("Day of the test week (UTC)")
    ax.set_ylabel("Residual, observed minus forecast (MW)")
    ax.set_title("By day of the test week")

    fig.suptitle(
        f"Residuals of the winning model ({c.MODEL_LABELS[result.name]}); "
        "positive means the model forecast too little load",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(c.FIGURE_DIR / "06_residuals.png")
    plt.close(fig)


def figure_feature_importance(result: c.ForecastResult) -> None:
    """Which features the boosted trees actually leaned on."""
    importance = pd.Series(result.extras["feature_importance_gain"]).sort_values(ascending=True)
    share = 100.0 * importance / importance.sum()

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(share)), share.to_numpy(), color=c.MODEL_COLORS["lightgbm"], edgecolor="white", linewidth=0.6)
    for i, value in enumerate(share.to_numpy()):
        ax.text(value + 0.6, i, f"{value:.1f}%", va="center", fontsize=8)
    ax.set_yticks(range(len(share)))
    ax.set_yticklabels(share.index)
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_ylabel("Feature")
    ax.set_xlim(0, float(share.max()) * 1.15)
    ax.set_title("LightGBM feature importance (gain), final model fitted on 2015-01-01 to 2019-12-31")
    fig.savefig(c.FIGURE_DIR / "07_feature_importance.png")
    plt.close(fig)


def figure_decomposition(result: c.ForecastResult) -> None:
    """Prophet's view of the world: trend plus repeating shapes."""
    components: pd.DataFrame = result.extras["components"]
    components = components.set_index("ds")

    panels = [
        ("trend", "Trend (MW)", "Slow movement of the overall level"),
        ("weekly", "Weekly component (MW)", "Shape repeated every week"),
        ("yearly", "Yearly component (MW)", "Shape repeated every year"),
        ("daily", "Daily component (MW)", "Shape repeated every day"),
    ]
    panels = [p for p in panels if p[0] in components.columns]

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.4 * len(panels)), sharex=False)
    color = c.MODEL_COLORS["prophet"]

    for ax, (column, ylabel, subtitle) in zip(np.atleast_1d(axes), panels):
        if column == "trend":
            ax.plot(components.index, components[column].to_numpy(), color=color, linewidth=1.3)
            ax.set_xlabel("Time (UTC)")
        elif column == "weekly":
            one_week = components.loc["2019-12-02":"2019-12-08"]
            ax.plot(range(len(one_week)), one_week[column].to_numpy(), color=color, linewidth=1.3)
            ax.set_xticks(range(0, 168, 24))
            ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            ax.set_xlabel("Day of week")
        elif column == "yearly":
            one_year = components.loc["2019-01-01":"2019-12-31"]
            ax.plot(one_year.index, one_year[column].to_numpy(), color=color, linewidth=1.3)
            ax.set_xlabel("Date in 2019")
        else:
            one_day = components.loc["2019-12-04":"2019-12-04 23:00"]
            ax.plot(range(len(one_day)), one_day[column].to_numpy(), color=color, linewidth=1.3)
            ax.set_xticks(range(0, 24, 3))
            ax.set_xlabel("Hour of day (UTC)")
        ax.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=10)

    fig.suptitle("Prophet decomposition of German hourly load", fontsize=12)
    fig.tight_layout()
    fig.savefig(c.FIGURE_DIR / "08_decomposition.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------------


def build_metrics(
    series: pd.Series,
    results: dict[str, c.ForecastResult],
    scores: pd.DataFrame,
    winner: str,
    probabilistic: dict[str, float],
    total_runtime: float,
) -> dict[str, Any]:
    test = series.loc[c.TEST_START : c.TEST_END]
    models = []
    for name in scores.index:
        row = scores.loc[name]
        models.append(
            {
                "name": name,
                "mape_test_pct": float(row["mape_test_pct"]),
                "rmse_test_mw": float(row["rmse_test_mw"]),
                "mae_test_mw": float(row["mae_test_mw"]),
                "mape_jan1_pct": float(row["mape_jan1_pct"]),
                "mape_jan2_to_jan7_pct": float(row["mape_jan2_to_jan7_pct"]),
                "runtime_seconds": float(row["runtime_seconds"]),
                "hyperparameters": results[name].hyperparameters,
                "validation_mape_pct": None
                if results[name].validation_mape_pct is None
                else float(results[name].validation_mape_pct),
                "per_day_mape_pct": c.per_day_mape(test, results[name].point_forecast),
            }
        )

    return {
        "test_start": "2020-01-01",
        "test_end": "2020-01-07",
        "n_test_observations": int(len(test)),
        "models": models,
        "winner": winner,
        "winner_coverage_80pct": probabilistic["coverage_80pct"],
        "winner_coverage_95pct": probabilistic["coverage_95pct"],
        "winner_nominal_coverage_80pct": 0.80,
        "winner_nominal_coverage_95pct": 0.95,
        "winner_pinball_loss_q10": probabilistic["pinball_loss_q10"],
        "winner_pinball_loss_q50": probabilistic["pinball_loss_q50"],
        "winner_pinball_loss_q90": probabilistic["pinball_loss_q90"],
        "total_runtime_seconds": total_runtime,
        "random_seed": c.SEED,
        "data": c.data_integrity_report(series),
    }


def write_metrics_csv(payload: dict[str, Any]) -> None:
    """The same content as metrics.json, one row per model."""
    rows: list[dict[str, Any]] = []
    for entry in payload["models"]:
        is_winner = entry["name"] == payload["winner"]
        row: dict[str, Any] = {
            "test_start": payload["test_start"],
            "test_end": payload["test_end"],
            "n_test_observations": payload["n_test_observations"],
            "name": entry["name"],
            "mape_test_pct": entry["mape_test_pct"],
            "rmse_test_mw": entry["rmse_test_mw"],
            "mae_test_mw": entry["mae_test_mw"],
            "mape_jan1_pct": entry["mape_jan1_pct"],
            "mape_jan2_to_jan7_pct": entry["mape_jan2_to_jan7_pct"],
            "validation_mape_pct": entry["validation_mape_pct"],
            "runtime_seconds": entry["runtime_seconds"],
            "is_winner": is_winner,
        }
        row.update({f"mape_{day}_pct": value for day, value in entry["per_day_mape_pct"].items()})
        row.update(
            {
                "winner_coverage_80pct": payload["winner_coverage_80pct"] if is_winner else "",
                "winner_coverage_95pct": payload["winner_coverage_95pct"] if is_winner else "",
                "winner_pinball_loss_q10": payload["winner_pinball_loss_q10"] if is_winner else "",
                "winner_pinball_loss_q50": payload["winner_pinball_loss_q50"] if is_winner else "",
                "winner_pinball_loss_q90": payload["winner_pinball_loss_q90"] if is_winner else "",
            }
        )
        row["hyperparameters_json"] = json.dumps(entry["hyperparameters"], default=str)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame["total_runtime_seconds"] = payload["total_runtime_seconds"]
    frame.to_csv(c.ROOT / "metrics.csv", index=False)


def write_results_table(scores: pd.DataFrame) -> None:
    """A markdown table for the write-up, so nobody retypes numbers by hand."""
    lines = [
        "| Model | Test MAPE (%) | RMSE (MW) | MAE (MW) | MAPE 1 Jan, holiday (%) | MAPE 2-7 Jan (%) | Validation MAPE (%) | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in scores.index:
        row = scores.loc[name]
        validation = "n/a" if pd.isna(row["validation_mape_pct"]) else f"{row['validation_mape_pct']:.2f}"
        lines.append(
            f"| {c.MODEL_LABELS[name]} | {row['mape_test_pct']:.2f} | {row['rmse_test_mw']:.0f} | "
            f"{row['mae_test_mw']:.0f} | {row['mape_jan1_pct']:.2f} | {row['mape_jan2_to_jan7_pct']:.2f} | "
            f"{validation} | {row['runtime_seconds']:.0f} |"
        )
    (c.ARTIFACT_DIR / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def environment_report() -> dict[str, str]:
    uname = subprocess.run(["uname", "-a"], capture_output=True, text=True, check=False).stdout.strip()
    return {
        "uname": uname,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the six-model load forecasting bake-off.")
    parser.add_argument(
        "--refit",
        nargs="*",
        default=[],
        help="Model names to refit from scratch, or 'all'.",
    )
    args = parser.parse_args()
    refit = set(c.MODEL_NAMES) if "all" in args.refit else set(args.refit)

    started = time.time()
    c.apply_figure_style()
    c.set_seeds()
    c.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    c.ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    series = c.load_load_series()
    print(json.dumps(c.data_integrity_report(series), indent=2))

    results = gather_results(series, refit)
    scores = score_everything(series, results)

    # The winner is the model with the lowest test MAPE that can also state
    # its own uncertainty; the naive baseline produces no intervals, so if it
    # ever won, the probabilistic part of the study would have nothing to
    # report. It does not win here, and the check is only a safeguard.
    ranked = [name for name in scores.index if results[name].quantile_forecast is not None]
    winner = ranked[0]
    if scores.index[0] != winner:
        print(f"NOTE: {scores.index[0]} has the lowest MAPE but produces no intervals.")

    test = series.loc[c.TEST_START : c.TEST_END]
    probabilistic = c.probabilistic_scores(test, results[winner].quantile_forecast)

    figure_overview(series)
    figure_forecast_comparison(series, results, scores)
    figure_metric_comparison(scores)
    figure_per_day_mape(scores)
    figure_winner_with_intervals(series, results[winner], scores, probabilistic)
    figure_residuals(series, results[winner])

    top_two = list(scores.index[:2])
    if "lightgbm" in top_two:
        figure_feature_importance(results["lightgbm"])
    if "prophet" in top_two:
        figure_decomposition(results["prophet"])

    total_runtime = float(sum(r.runtime_seconds for r in results.values()) + (time.time() - started))
    payload = build_metrics(series, results, scores, winner, probabilistic, total_runtime)
    payload["environment"] = environment_report()

    c.write_json(c.ROOT / "metrics.json", payload)
    write_metrics_csv(payload)
    write_results_table(scores)

    print(f"\nWinner: {winner} (test MAPE {scores.loc[winner, 'mape_test_pct']:.2f}%)")
    print(
        f"80% interval coverage {probabilistic['coverage_80pct']:.3f}, "
        f"95% interval coverage {probabilistic['coverage_95pct']:.3f}"
    )
    print(f"Total runtime accounted for: {total_runtime:.0f} s")


if __name__ == "__main__":
    main()
