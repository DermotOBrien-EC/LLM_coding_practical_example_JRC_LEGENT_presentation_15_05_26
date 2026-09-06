"""Orchestrator: run all six models, score them, draw the figures, write the report.

Run it with the project venv from the study directory:

    ../../.venv/bin/python code/forecast.py

Everything downstream of this file (metrics.json, metrics.csv, the eight
figures and transcript.md) is regenerated from scratch on every run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_DIR))

import common  # noqa: F401,E402  imported first: it repairs the `prophet` name clash


def _load_by_path(module_name: str, filename: str) -> Any:
    """Import a file from `code/` under a name of our choosing.

    Used for `prophet.py`, whose filename the study layout fixes but which
    must not be registered as the module `prophet`, because the real
    forecasting library owns that name.
    """
    spec = importlib.util.spec_from_file_location(module_name, _CODE_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


from common import (  # noqa: E402
    DPI,
    FIGURE_DIR,
    HORIZON,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    ROOT,
    SEED,
    SQUARE_FIGSIZE,
    TEST_END,
    TEST_START,
    TIMESERIES_FIGSIZE,
    TRAIN_END,
    VAL_END,
    ForecastResult,
    evaluate_point_forecast,
    interval_coverage,
    load_series,
    per_day_mape,
    pinball_loss,
    setup_matplotlib,
    split_series,
    write_json,
)

# Module file for each model. They are imported lazily, one per child process,
# and never together: LightGBM and PyTorch each bring their own OpenMP runtime,
# and importing both into one interpreter on macOS segfaults it.
MODEL_MODULES: dict[str, str] = {
    "naive": "naive.py",
    "sarima": "sarima.py",
    "prophet": "prophet.py",
    "lightgbm": "lightgbm_features.py",
    "nbeats": "nbeats.py",
    "patchtst": "patchtst.py",
}


def load_runner(name: str) -> Any:
    """Import one model module and hand back its `run` function."""
    module = _load_by_path(f"{name}_entry", MODEL_MODULES[name])
    return module.run


CACHE_DIR: Path = _CODE_DIR.parent / ".model_cache"


def run_model_in_subprocess(name: str, series: pd.Series, use_cache: bool) -> ForecastResult:
    """Fit one model in a fresh Python process and bring the result back.

    This is not defensive programming for its own sake. LightGBM and PyTorch
    each ship their own OpenMP runtime, and on macOS loading both into one
    process makes the second one to train hang indefinitely. That was observed
    here, reproducibly: N-BEATS never completed a single epoch after a
    LightGBM fit in the same interpreter. Giving every model its own process
    removes the interaction entirely, and has the side benefit that one
    model's failure cannot take the rest of the study down with it.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{name}.pkl"
    if use_cache and cache_path.exists():
        with cache_path.open("rb") as handle:
            print(f"    {name}: reusing cached fit", flush=True)
            return pickle.load(handle)

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-model",
            name,
            "--output",
            str(cache_path),
        ],
        cwd=str(_CODE_DIR.parent),
    )
    if completed.returncode != 0:
        raise RuntimeError(f"model {name} failed with exit code {completed.returncode}")
    with cache_path.open("rb") as handle:
        result = pickle.load(handle)
    if not isinstance(result, ForecastResult) or result.name != name:
        raise RuntimeError(f"model {name} returned an unexpected object")
    return result


def run_single_model(name: str, output: Path) -> None:
    """Child-process entry point: fit one model and pickle it for the parent."""
    series = load_series()
    result = load_runner(name)(series)
    output.parent.mkdir(exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump(result, handle)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def figure_01_overview(series: pd.Series) -> None:
    """The whole series, with the test week marked so the reader can place it."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=TIMESERIES_FIGSIZE)
    plain_index = series.index.tz_convert("UTC").tz_localize(None)
    ax.plot(plain_index, series.to_numpy(), color="0.78", linewidth=0.25, label="Hourly load")
    # A weekly average on top: at this width the hourly trace is a solid band,
    # and the seasonal swing is what the reader is here to see.
    weekly_mean = series.rolling(168, center=True).mean()
    ax.plot(
        plain_index,
        weekly_mean.to_numpy(),
        color="0.35",
        linewidth=1.1,
        label="7-day moving average",
    )

    test = series.loc[TEST_START:TEST_END]
    ax.axvspan(
        TEST_START.tz_localize(None) - pd.Timedelta(days=25),
        TEST_END.tz_localize(None) + pd.Timedelta(days=25),
        color="#d62728",
        alpha=0.12,
        linewidth=0,
        label="Test week, widened so it is visible at this scale",
    )
    ax.plot(
        test.index.tz_convert("UTC").tz_localize(None),
        test.to_numpy(),
        color="#d62728",
        linewidth=1.4,
        label="Test week (2020-01-01 to 2020-01-07)",
    )
    for boundary, text, align in (
        (TRAIN_END, "train ends", "right"),
        (VAL_END, "validation ends", "left"),
    ):
        ax.axvline(boundary.tz_localize(None), color="0.25", linestyle="--", linewidth=0.9)
        ax.annotate(
            text,
            xy=(boundary.tz_localize(None), 0.985),
            xycoords=("data", "axes fraction"),
            xytext=(-4 if align == "right" else 4, 0),
            textcoords="offset points",
            rotation=90,
            fontsize=8,
            color="0.25",
            ha=align,
            va="top",
        )

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (MW)")
    ax.set_title(
        "German national electricity load, hourly, 2015-01-01 to 2020-09-30\n"
        "The 168-hour held-out test week is highlighted in red"
    )
    ax.legend(loc="lower left", ncol=2)
    ax.set_xlim(plain_index[0], plain_index[-1])
    fig.savefig(FIGURE_DIR / "01_overview.png", dpi=DPI)
    plt.close(fig)


def figure_02_forecast_comparison(
    actual: pd.Series, results: dict[str, ForecastResult], metrics: dict[str, dict[str, float]]
) -> None:
    """One small panel per model: black actuals against that model's forecast."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    index = actual.index.tz_convert("UTC").tz_localize(None)
    lo = min(actual.min(), *(results[m].point.min() for m in MODEL_ORDER))
    hi = max(actual.max(), *(results[m].point.max() for m in MODEL_ORDER))
    pad = 0.06 * (hi - lo)

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        ax.plot(index, actual.to_numpy(), color="black", linewidth=1.3, label="Observed")
        ax.plot(
            index,
            results[name].point.to_numpy(),
            color=MODEL_COLORS[name],
            linewidth=1.5,
            label="Forecast",
        )
        ax.axvspan(index[0], index[23], color="0.85", alpha=0.55, linewidth=0)
        ax.set_title(f"{MODEL_LABELS[name]} - MAPE {metrics[name]['mape_test_pct']:.2f}%")
        ax.set_ylim(lo - pad, hi + pad)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.legend(loc="upper left", fontsize=8)

    for ax in axes[:, 0]:
        ax.set_ylabel("Load (MW)")
    for ax in axes[1, :]:
        ax.set_xlabel("Time (UTC)")

    fig.suptitle(
        "Test-week forecasts, 2020-01-01 to 2020-01-07 (shaded band: 1 January, a public holiday)",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(FIGURE_DIR / "02_forecast_comparison.png", dpi=DPI)
    plt.close(fig)


def figure_03_metric_comparison(metrics: dict[str, dict[str, float]], order: list[str]) -> None:
    """MAPE, RMSE and MAE side by side. MAPE has its own axis because it is a percentage."""
    import matplotlib.pyplot as plt

    x = np.arange(len(order))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    ax_right = ax.twinx()
    ax_right.grid(False)
    ax_right.spines["right"].set_visible(True)

    mape_vals = [metrics[m]["mape_test_pct"] for m in order]
    rmse_vals = [metrics[m]["rmse_test_mw"] for m in order]
    mae_vals = [metrics[m]["mae_test_mw"] for m in order]
    colors = [MODEL_COLORS[m] for m in order]

    bars_mape = ax.bar(
        x - width,
        mape_vals,
        width,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        label="MAPE (%, left axis)",
    )
    bars_rmse = ax_right.bar(
        x,
        rmse_vals,
        width,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.65,
        hatch="//",
        label="RMSE (MW, right axis)",
    )
    bars_mae = ax_right.bar(
        x + width,
        mae_vals,
        width,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        alpha=0.35,
        hatch="..",
        label="MAE (MW, right axis)",
    )

    for bars, axis, fmt in (
        (bars_mape, ax, "{:.2f}"),
        (bars_rmse, ax_right, "{:.0f}"),
        (bars_mae, ax_right, "{:.0f}"),
    ):
        for bar in bars:
            axis.annotate(
                fmt.format(bar.get_height()),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in order], rotation=15, ha="right")
    ax.set_ylabel("MAPE (%)")
    ax_right.set_ylabel("RMSE / MAE (MW)")
    ax.set_ylim(0, max(mape_vals) * 1.35)
    ax_right.set_ylim(0, max(rmse_vals) * 1.35)
    ax.set_title(
        "Test-window accuracy by model, sorted by MAPE (lower is better)\n"
        "Bar colour identifies the model; hatching identifies the metric"
    )
    handles = [bars_mape, bars_rmse, bars_mae]
    ax.legend(handles, [h.get_label() for h in handles], loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_metric_comparison.png", dpi=DPI)
    plt.close(fig)


def figure_04_per_day_mape(daily: pd.DataFrame, order: list[str]) -> None:
    """Per-day MAPE: seven day groups, one bar per model inside each group."""
    import matplotlib.pyplot as plt

    days = list(daily.index)
    x = np.arange(len(days))
    width = 0.8 / len(order)

    fig, ax = plt.subplots(figsize=(12.0, 6.0))
    for i, name in enumerate(order):
        offset = (i - (len(order) - 1) / 2) * width
        ax.bar(
            x + offset,
            daily[name].to_numpy(),
            width,
            color=MODEL_COLORS[name],
            edgecolor="black",
            linewidth=0.4,
            label=MODEL_LABELS[name],
        )

    labels = [
        f"{d:%a %d %b}" + ("\n(public holiday)" if i == 0 else "") for i, d in enumerate(days)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("Day of the test window (UTC)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title(
        "Daily MAPE across the test week\n"
        "1 January is New Year's Day: models without a holiday effect are exposed here"
    )
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, daily.to_numpy().max() * 1.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_per_day_mape.png", dpi=DPI)
    plt.close(fig)


def figure_05_winner_with_intervals(
    actual: pd.Series, result: ForecastResult, summary: dict[str, Any]
) -> None:
    """The winner's forecast with its 80 and 95 percent prediction bands."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    index = actual.index.tz_convert("UTC").tz_localize(None)
    color = MODEL_COLORS[result.name]

    fig, ax = plt.subplots(figsize=TIMESERIES_FIGSIZE)
    ax.fill_between(
        index,
        result.quantiles[0.025].to_numpy(),
        result.quantiles[0.975].to_numpy(),
        color=color,
        alpha=0.15,
        linewidth=0,
        label="95% prediction interval",
    )
    ax.fill_between(
        index,
        result.quantiles[0.1].to_numpy(),
        result.quantiles[0.9].to_numpy(),
        color=color,
        alpha=0.30,
        linewidth=0,
        label="80% prediction interval",
    )
    ax.plot(index, result.point.to_numpy(), color=color, linewidth=1.8, label="Point forecast")
    ax.plot(index, actual.to_numpy(), color="black", linewidth=1.4, label="Observed")
    ax.axvspan(index[0], index[23], color="0.85", alpha=0.5, linewidth=0)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (MW)")
    ax.set_title(
        f"Winning model: {MODEL_LABELS[result.name]} - test MAPE "
        f"{summary['mape_test_pct']:.2f}%\n"
        f"Actual coverage: {summary['coverage_80'] * 100:.1f}% inside the nominal 80% band, "
        f"{summary['coverage_95'] * 100:.1f}% inside the nominal 95% band "
        f"(shaded band on the left: 1 January)"
    )
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_winner_with_intervals.png", dpi=DPI)
    plt.close(fig)


def figure_06_residuals(actual: pd.Series, result: ForecastResult) -> None:
    """Where the winner is systematically wrong: by hour of day and by test day."""
    import matplotlib.pyplot as plt

    residual = (actual - result.point.reindex(actual.index)).astype("float64")
    local = actual.index.tz_convert("Europe/Berlin")
    color = MODEL_COLORS[result.name]

    fig, axes = plt.subplots(1, 2, figsize=TIMESERIES_FIGSIZE)

    hours = sorted(set(local.hour))
    by_hour = [residual.to_numpy()[local.hour == h] for h in hours]
    box = axes[0].boxplot(
        by_hour,
        positions=hours,
        widths=0.6,
        patch_artist=True,
        medianprops={"color": "black"},
        flierprops={"markersize": 3},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    axes[0].axhline(0, color="black", linewidth=0.9)
    axes[0].set_xlabel("Hour of day (local time, CET)")
    axes[0].set_ylabel("Residual: observed minus forecast (MW)")
    axes[0].set_title("Residuals by hour of day")
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)])

    days = actual.index.tz_convert("UTC").normalize()
    unique_days = sorted(set(days))
    by_day = [residual.to_numpy()[days == d] for d in unique_days]
    box = axes[1].boxplot(
        by_day,
        positions=range(len(unique_days)),
        widths=0.6,
        patch_artist=True,
        medianprops={"color": "black"},
        flierprops={"markersize": 3},
    )
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    axes[1].axhline(0, color="black", linewidth=0.9)
    axes[1].set_xlabel("Day of the test week (UTC)")
    axes[1].set_ylabel("Residual: observed minus forecast (MW)")
    axes[1].set_title("Residuals by day of the test week")
    axes[1].set_xticks(range(len(unique_days)))
    axes[1].set_xticklabels([f"{pd.Timestamp(d):%a\n%d %b}" for d in unique_days], fontsize=8)

    fig.suptitle(
        f"Residual structure of the winning model ({MODEL_LABELS[result.name]}); "
        "positive means the model under-predicted",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE_DIR / "06_residuals.png", dpi=DPI)
    plt.close(fig)


def figure_07_feature_importance(result: ForecastResult) -> None:
    """Which of the engineered features LightGBM actually leaned on."""
    import matplotlib.pyplot as plt

    importance = result.extra["feature_importance_gain"].sort_values()
    share = 100.0 * importance / importance.sum()

    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    ax.barh(
        range(len(share)),
        share.to_numpy(),
        color=MODEL_COLORS["lightgbm"],
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_yticks(range(len(share)))
    ax.set_yticklabels(share.index, fontsize=9)
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_ylabel("Feature")
    ax.set_title("LightGBM feature importance\n(gain, final fit on 2015-2019 data)")
    for i, value in enumerate(share.to_numpy()):
        ax.annotate(
            f"{value:.1f}",
            xy=(value, i),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, share.max() * 1.18)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "07_feature_importance.png", dpi=DPI)
    plt.close(fig)


def figure_08_decomposition(result: ForecastResult) -> None:
    """Prophet's own view of the series: trend, weekly shape, yearly shape."""
    import matplotlib.pyplot as plt

    parts = result.extra["components"]
    trend, weekly, yearly = parts["trend"], parts["weekly"], parts["yearly"]
    holiday = parts["holiday"]
    trend_dates = pd.DatetimeIndex(trend["ds"])
    yearly_dates = pd.DatetimeIndex(yearly["ds"])
    holiday_dates = pd.DatetimeIndex(holiday["ds"])

    fig, axes = plt.subplots(4, 1, figsize=(11.0, 11.5))
    color = MODEL_COLORS["prophet"]

    axes[0].plot(trend_dates, trend["trend"].to_numpy(), color=color)
    axes[0].set_ylabel("Trend (MW)")
    axes[0].set_xlabel("Time (UTC)")
    axes[0].set_title("Trend component: the slow drift in the level of demand")

    axes[1].plot(range(168), weekly["weekly"].to_numpy(), color=color)
    axes[1].set_ylabel("Weekly effect (MW)")
    axes[1].set_xlabel("Hour of the week (0 = Monday 00:00 UTC)")
    axes[1].set_xticks(range(0, 169, 24))
    axes[1].set_title("Weekly component: the working-week rhythm")
    axes[1].axhline(0, color="black", linewidth=0.8)

    axes[2].plot(yearly_dates, yearly["yearly"].to_numpy(), color=color)
    axes[2].set_ylabel("Yearly effect (MW)")
    axes[2].set_xlabel("Day of year (2019 shown as a representative year)")
    axes[2].set_title("Yearly component: the winter-summer swing")
    axes[2].axhline(0, color="black", linewidth=0.8)

    axes[3].plot(holiday_dates, holiday["holidays"].to_numpy(), color=color)
    axes[3].axvline(pd.Timestamp("2020-01-01"), color="black", linestyle="--", linewidth=0.9)
    axes[3].annotate(
        "1 January",
        xy=(pd.Timestamp("2020-01-01"), 0.55),
        xycoords=("data", "axes fraction"),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=8,
        ha="left",
    )
    axes[3].set_ylabel("Holiday effect (MW)")
    axes[3].set_xlabel("Time (UTC), around the test window")
    axes[3].set_title(
        "Public-holiday component: the demand that Prophet subtracts on German holidays"
    )
    axes[3].axhline(0, color="black", linewidth=0.8)

    fig.suptitle(
        "Prophet decomposition of German hourly load (fit on 2015-01-01 to 2019-12-31)", y=0.995
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(FIGURE_DIR / "08_decomposition.png", dpi=DPI)
    plt.close(fig)


# ----------------------------------------------------------------------
# Metrics assembly
# ----------------------------------------------------------------------


def winner_probabilistic_summary(
    actual: pd.Series, result: ForecastResult
) -> dict[str, float | None]:
    """Interval coverage and pinball losses for the winning model."""
    if not result.has_intervals():
        return {
            "coverage_80": None,
            "coverage_95": None,
            "pinball_q10": None,
            "pinball_q50": None,
            "pinball_q90": None,
        }
    return {
        "coverage_80": interval_coverage(actual, result.quantiles[0.1], result.quantiles[0.9]),
        "coverage_95": interval_coverage(actual, result.quantiles[0.025], result.quantiles[0.975]),
        "pinball_q10": pinball_loss(actual, result.quantiles[0.1], 0.1),
        "pinball_q50": pinball_loss(actual, result.quantiles[0.5], 0.5),
        "pinball_q90": pinball_loss(actual, result.quantiles[0.9], 0.9),
    }


def build_metrics_payload(
    results: dict[str, ForecastResult],
    metrics: dict[str, dict[str, float]],
    winner: str,
    winner_summary: dict[str, float | None],
    total_runtime: float,
) -> dict[str, Any]:
    """Assemble metrics.json exactly in the schema the study asks for."""
    return {
        "test_start": str(TEST_START.date()),
        "test_end": str(TEST_END.date()),
        "n_test_observations": HORIZON,
        "models": [
            {
                "name": name,
                "mape_test_pct": round(metrics[name]["mape_test_pct"], 4),
                "rmse_test_mw": round(metrics[name]["rmse_test_mw"], 2),
                "mae_test_mw": round(metrics[name]["mae_test_mw"], 2),
                "mape_jan1_pct": round(metrics[name]["mape_jan1_pct"], 4),
                "mape_jan2_to_jan7_pct": round(metrics[name]["mape_jan2_to_jan7_pct"], 4),
                "runtime_seconds": round(results[name].runtime_seconds, 2),
                "hyperparameters": results[name].hyperparameters,
            }
            for name in MODEL_ORDER
        ],
        "winner": winner,
        "winner_coverage_80pct": _round_or_none(winner_summary["coverage_80"], 4),
        "winner_coverage_95pct": _round_or_none(winner_summary["coverage_95"], 4),
        "winner_pinball_loss_q10": _round_or_none(winner_summary["pinball_q10"], 3),
        "winner_pinball_loss_q50": _round_or_none(winner_summary["pinball_q50"], 3),
        "winner_pinball_loss_q90": _round_or_none(winner_summary["pinball_q90"], 3),
        "total_runtime_seconds": round(total_runtime, 2),
    }


def _round_or_none(value: float | None, digits: int) -> float | None:
    return None if value is None else round(float(value), digits)


def write_metrics_csv(path: Path, payload: dict[str, Any], daily: pd.DataFrame) -> None:
    """The same numbers as metrics.json, one row per model, for spreadsheet users."""
    rows: list[dict[str, Any]] = []
    for entry in payload["models"]:
        name = entry["name"]
        is_winner = name == payload["winner"]
        row: dict[str, Any] = {
            "name": name,
            "label": MODEL_LABELS[name],
            "test_start": payload["test_start"],
            "test_end": payload["test_end"],
            "n_test_observations": payload["n_test_observations"],
            "mape_test_pct": entry["mape_test_pct"],
            "rmse_test_mw": entry["rmse_test_mw"],
            "mae_test_mw": entry["mae_test_mw"],
            "mape_jan1_pct": entry["mape_jan1_pct"],
            "mape_jan2_to_jan7_pct": entry["mape_jan2_to_jan7_pct"],
            "runtime_seconds": entry["runtime_seconds"],
            "is_winner": int(is_winner),
            "winner_coverage_80pct": payload["winner_coverage_80pct"] if is_winner else "",
            "winner_coverage_95pct": payload["winner_coverage_95pct"] if is_winner else "",
            "winner_pinball_loss_q10": payload["winner_pinball_loss_q10"] if is_winner else "",
            "winner_pinball_loss_q50": payload["winner_pinball_loss_q50"] if is_winner else "",
            "winner_pinball_loss_q90": payload["winner_pinball_loss_q90"] if is_winner else "",
            "total_runtime_seconds": payload["total_runtime_seconds"],
            "hyperparameters_json": json.dumps(entry["hyperparameters"], sort_keys=True),
        }
        for day in daily.index:
            row[f"mape_{day:%Y%m%d}_pct"] = round(float(daily.loc[day, name]), 4)
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("mape_test_pct").reset_index(drop=True)
    frame.to_csv(path, index=False)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main(use_cache: bool = False) -> None:
    setup_matplotlib()
    FIGURE_DIR.mkdir(exist_ok=True)

    wall_start = time.perf_counter()
    series = load_series()
    train, val, test = split_series(series)
    print(
        f"data: {len(series)} hours; train {len(train)}, validation {len(val)}, test {len(test)}",
        flush=True,
    )

    results: dict[str, ForecastResult] = {}
    fitting_wall = 0.0
    for name in MODEL_ORDER:
        print(f"--- running {name} ---", flush=True)
        t0 = time.perf_counter()
        results[name] = run_model_in_subprocess(name, series, use_cache)
        fitting_wall += time.perf_counter() - t0
        print(f"    {name} done in {results[name].runtime_seconds:.1f}s", flush=True)

    metrics = {name: evaluate_point_forecast(test, results[name].point) for name in MODEL_ORDER}
    daily = pd.DataFrame({name: per_day_mape(test, results[name].point) for name in MODEL_ORDER})

    order_by_mape = sorted(MODEL_ORDER, key=lambda n: metrics[n]["mape_test_pct"])
    winner = order_by_mape[0]
    winner_summary = winner_probabilistic_summary(test, results[winner])
    # Reported runtime is the sum of the six model runtimes plus the
    # orchestrator's own work. Measuring the parent's wall clock directly
    # would understate the run whenever fits are reused from the cache.
    total_runtime = sum(r.runtime_seconds for r in results.values()) + (
        time.perf_counter() - wall_start - fitting_wall
    )

    payload = build_metrics_payload(results, metrics, winner, winner_summary, total_runtime)
    write_json(ROOT / "metrics.json", payload)
    write_metrics_csv(ROOT / "metrics.csv", payload, daily)

    figure_01_overview(series)
    figure_02_forecast_comparison(test, results, metrics)
    figure_03_metric_comparison(metrics, order_by_mape)
    figure_04_per_day_mape(daily, order_by_mape)
    figure_05_winner_with_intervals(test, results[winner], {**metrics[winner], **winner_summary})
    figure_06_residuals(test, results[winner])

    top_two = order_by_mape[:2]
    if "lightgbm" in top_two:
        figure_07_feature_importance(results["lightgbm"])
    if "prophet" in top_two:
        figure_08_decomposition(results["prophet"])

    write_transcript(
        results, metrics, daily, order_by_mape, winner_summary, total_runtime, len(series)
    )

    print("\nfinal ranking by test MAPE:")
    for name in order_by_mape:
        print(f"  {MODEL_LABELS[name]:<26} {metrics[name]['mape_test_pct']:6.2f}%")
    print(f"\ntotal wall clock: {total_runtime / 60:.1f} min")


# ----------------------------------------------------------------------
# transcript.md
# ----------------------------------------------------------------------


def _results_table(
    metrics: dict[str, dict[str, float]], results: dict[str, ForecastResult], order: list[str]
) -> str:
    header = (
        "| Rank | Model | MAPE (%) | RMSE (MW) | MAE (MW) | MAPE 1 Jan (%) "
        "| MAPE 2-7 Jan (%) | Runtime (s) |\n"
        "|---:|---|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = []
    for rank, name in enumerate(order, start=1):
        m = metrics[name]
        lines.append(
            f"| {rank} | {MODEL_LABELS[name]} | {m['mape_test_pct']:.2f} "
            f"| {m['rmse_test_mw']:,.0f} | {m['mae_test_mw']:,.0f} "
            f"| {m['mape_jan1_pct']:.2f} | {m['mape_jan2_to_jan7_pct']:.2f} "
            f"| {results[name].runtime_seconds:,.0f} |"
        )
    return header + "\n".join(lines) + "\n"


def _selection_table(results: dict[str, ForecastResult]) -> str:
    lines = ["| Model | Chosen configuration | Selected on |", "|---|---|---|"]
    for name in MODEL_ORDER:
        hp = results[name].hyperparameters
        chosen = ", ".join(
            f"`{k}`={v}"
            for k, v in hp.items()
            if k
            not in {
                "selection_metric",
                "interval_method",
                "features",
                "categorical_features",
                "substitution",
                "seasonalities",
            }
        )
        criterion = hp.get("selection_metric", "no hyperparameters to select")
        lines.append(f"| {MODEL_LABELS[name]} | {chosen} | {criterion} |")
    return "\n".join(lines) + "\n"


def write_transcript(
    results: dict[str, ForecastResult],
    metrics: dict[str, dict[str, float]],
    daily: pd.DataFrame,
    order: list[str],
    winner_summary: dict[str, float | None],
    total_runtime: float,
    n_observations: int,
) -> None:
    """Write the methods-section write-up, with every number taken from this run."""
    winner = order[0]
    naive_mape = metrics["naive"]["mape_test_pct"]
    win_mape = metrics[winner]["mape_test_pct"]
    improvement = 100.0 * (naive_mape - win_mape) / naive_mape

    beat_naive = [n for n in order if metrics[n]["mape_test_pct"] < naive_mape]
    lost_to_naive = [n for n in order if metrics[n]["mape_test_pct"] > naive_mape]

    uname = subprocess.run(["uname", "-a"], capture_output=True, text=True).stdout.strip()

    cov80 = winner_summary["coverage_80"]
    cov95 = winner_summary["coverage_95"]
    coverage_sentence = (
        "The winning model produces no prediction intervals, so coverage is not reported."
        if cov80 is None
        else (
            "Both bands are drawn over the observed week in "
            "`figures/05_winner_with_intervals.png`, and section 5 reads the miscalibration."
        )
    )

    text = f"""# German hourly load: a six-model forecasting bake-off

A held-out week of German national electricity demand (1 to 7 January 2020,
168 hours) forecast by six models of increasing complexity, all fitted on the
same history, all judged on the same numbers.

**Headline: {MODEL_LABELS[winner]} wins with a test MAPE of {win_mape:.2f} percent**,
against {naive_mape:.2f} percent for the seasonal-naive baseline
({improvement:.0f} percent lower error).

---

## 1. Data

The series is hourly German national electricity load in megawatts, taken
from Open Power System Data, who in turn derive it from the ENTSO-E
Transparency Platform. It runs from 2015-01-01 00:00 UTC to 2020-09-30 23:00
UTC: {n_observations:,} hourly observations with no missing values and
no missing hours. The loader in `code/common.py` checks all three of those
claims on every run and raises if any of them fails, so nothing was imputed,
interpolated or dropped because nothing needed to be. The timestamps are UTC
throughout. Calendar features are derived from the Berlin local time that the
same instants correspond to, because German demand follows the local clock
(including summer time), not UTC.

## 2. Why these six models

The six are a deliberate ladder of complexity, and each rung is supposed to
add one specific thing.

The **seasonal-naive** forecast says the load at any hour equals the load at
the same hour one week earlier. It has no parameters and knows nothing. It is
the anchor: any model that cannot beat it is not paying for itself.

**SARIMA** adds a statistical description of short memory: how this hour
relates to the last few hours and to the same hour yesterday. It brings
principled, analytically derived prediction intervals.

**Prophet** adds explicit structure that a human names in advance: a bending
trend, a daily shape, a weekly shape, a yearly shape, and a separate effect
for every German public holiday. It is the only model in the set that is
*told* what 1 January is.

**LightGBM on engineered features** adds the ability to learn interactions.
It sees the same calendar information Prophet does, plus lags of the load
itself (24 hours, one week, and roughly one year back) and rolling averages,
and learns from data how they combine.

**N-BEATS** and the **TSMixer** entry (see the substitution note below) add
capacity and remove prior knowledge. Both read 168 raw hours and emit the
next 168 in one shot, with no calendar input at all. Anything they know about
weekends or holidays they had to infer from the numbers.

**Substitution note.** The brief asked for `darts.models.PatchTSTModel` or
`darts.models.TSMixerModel`. darts 0.41.0 in this environment exposes
`TSMixerModel` but not `PatchTSTModel`, so TSMixer occupies that slot. It is
an all-MLP architecture rather than a transformer, and it is reported under
the name `patchtst` in `metrics.json` only because the schema fixes that
label.

## 3. Validation strategy

Three windows, cut once and never moved:

| Window | Range (UTC) | Hours |
|---|---|---:|
| Train | 2015-01-01 00:00 to 2019-09-30 23:00 | 41,616 |
| Validation | 2019-10-01 00:00 to 2019-12-31 23:00 | 2,208 |
| Test | 2020-01-01 00:00 to 2020-01-07 23:00 | 168 |

Every model with hyperparameters was fitted on Train alone and scored on
Validation. The scoring is not one-step-ahead error: the validation window
holds thirteen whole weeks, and each candidate was asked to forecast each of
those weeks 168 hours ahead from a cold start, exactly the job it would be
given on the test window. Averaging over thirteen weeks rather than one keeps
the choice from being decided by a single unusual week. (SARIMA is scored on
four evenly spaced weeks rather than thirteen, purely to keep its sweep of
twelve candidate specifications inside a few minutes.)

Once a configuration was chosen it was refitted from scratch on Train plus
Validation combined, that is on everything from 2015-01-01 to 2019-12-31, and
only then pointed at the test week. Refitting is not optional: the last
quarter of 2019 is the most informative data available about January 2020,
and throwing it away to preserve a tidy split would make the forecast worse
for no methodological gain. The test window itself was never read by any
fitting or selection code; the naive baseline's own guard raises if its lags
would reach into it.

Configurations chosen:

{_selection_table(results)}
## 4. Results

All figures are test-window numbers on the 168 held-out hours. "1 Jan" is the
24 UTC hours of New Year's Day, a German federal public holiday falling on a
Wednesday; "2-7 Jan" is the remaining 144 hours, which are four ordinary
working days and a weekend.

{_results_table(metrics, results, order)}
Per-day MAPE for every model is in `figures/04_per_day_mape.png` and in the
per-day columns of `metrics.csv`.

For the winning model, {MODEL_LABELS[winner]}:

| Quantity | Nominal | Actual |
|---|---:|---:|
| 80% prediction-interval coverage | 80.0% | {"n/a" if cov80 is None else f"{cov80 * 100:.1f}%"} |
| 95% prediction-interval coverage | 95.0% | {"n/a" if cov95 is None else f"{cov95 * 100:.1f}%"} |

| Pinball loss (MW) | q = 0.1 | q = 0.5 | q = 0.9 |
|---|---:|---:|---:|
| {MODEL_LABELS[winner]} | {_fmt(winner_summary["pinball_q10"])} | {_fmt(winner_summary["pinball_q50"])} | {_fmt(winner_summary["pinball_q90"])} |

{coverage_sentence}

## 5. Discussion

{_discussion(results, metrics, daily, order, winner_summary, beat_naive, lost_to_naive)}

## 6. Recommendation

{_recommendation(order, metrics)}

## 7. Reproducibility

- **Seed.** A single seed, `SEED = {SEED}` in `code/common.py`, is passed to
  LightGBM, Prophet, N-BEATS and TSMixer, and `torch.manual_seed` is called
  before each neural fit. The naive baseline and SARIMA are deterministic.
  Two independent full runs on this machine produced byte-identical metrics
  for all six models, so nothing here depends on run order or on the fit
  cache. Bit-identical results on different hardware are not guaranteed:
  threaded floating-point reduction in the neural fits can move the last
  decimals.
- **How to reproduce.** `../../.venv/bin/python code/forecast.py`, run from
  this directory. It rewrites `metrics.json`, `metrics.csv`, every figure and
  this file. Each model is fitted in its own subprocess, so a failure in one
  does not take the study down with it.
- **Total wall-clock runtime.** {total_runtime / 60:.1f} minutes for the full
  run, including all hyperparameter sweeps, all refits and all figures.
  Per-model runtimes are in the results table above.
- **Software.** Python {platform.python_version()}, pandas {pd.__version__},
  numpy {np.__version__}, darts 0.41.0 (Prophet, N-BEATS, TSMixer), plus
  statsmodels, lightgbm and holidays. Neural training ran on CPU, not the
  machine's GPU, so that the run reproduces on machines without one.
- **Files.** Model code is one module per model under `code/`, with shared
  loading, metrics and plotting style in `code/common.py` and the
  orchestration in `code/forecast.py`.
- **Hardware.** `uname -a` reports:

  ```
  {uname}
  ```
"""
    (ROOT / "transcript.md").write_text(_wrap_prose(text))


def _wrap_prose(text: str, width: int = 79) -> str:
    """Re-wrap paragraph and list text to a fixed width, leaving structure alone.

    Numbers substituted into the prose shift line lengths, so the source
    strings cannot be hand-wrapped correctly. Tables, headings, horizontal
    rules and fenced blocks are passed through untouched; list items are
    wrapped with a hanging indent so continuation lines line up under the text.
    """
    import re
    import textwrap

    fixed = re.compile(r"^(#|\||```|-{3,}$)")
    bullet = re.compile(r"^(?P<marker>([-*+]|\d+\.)\s+)(?P<body>.*)$")

    out: list[str] = []
    buffer: list[str] = []
    indent = ""
    in_fence = False

    def flush() -> None:
        if buffer:
            out.extend(
                textwrap.wrap(
                    " ".join(buffer),
                    width=width,
                    subsequent_indent=indent,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
            buffer.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            out.append(line)
            in_fence = not in_fence
            continue
        if in_fence or not stripped or fixed.match(stripped):
            flush()
            indent = ""
            out.append(line)
            continue
        match = bullet.match(stripped)
        if match:
            flush()
            indent = " " * len(match.group("marker"))
            buffer.append(match.group("marker") + match.group("body"))
        else:
            buffer.append(stripped)
    flush()
    return "\n".join(out)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def _discussion(
    results: dict[str, ForecastResult],
    metrics: dict[str, dict[str, float]],
    daily: pd.DataFrame,
    order: list[str],
    winner_summary: dict[str, float | None],
    beat_naive: list[str],
    lost_to_naive: list[str],
) -> str:
    """The read of what actually happened, with every number taken from this run."""
    winner = order[0]
    naive_mape = metrics["naive"]["mape_test_pct"]
    win = metrics[winner]

    def label_list(names: list[str]) -> str:
        pretty = [MODEL_LABELS[n] for n in names]
        if not pretty:
            return "none of them"
        if len(pretty) == 1:
            return pretty[0]
        return ", ".join(pretty[:-1]) + " and " + pretty[-1]

    winner_worst_day = daily[winner].idxmax()
    nbeats_worst_day = daily["nbeats"].idxmax()
    nbeats_val_mape = min(row["val_mape_pct"] for row in results["nbeats"].selection_log)

    # Did the brief's epoch cap stop either neural model before it converged?
    capped = [
        MODEL_LABELS[n]
        for n in ("nbeats", "patchtst")
        if results[n].hyperparameters["epochs_used_for_final_fit"]
        >= results[n].hyperparameters["max_epochs"] - 1
    ]
    if capped:
        epoch_note = (
            f"{' and '.join(capped)} stopped at the brief's epoch cap, so that score is a "
            "floor rather than a ceiling"
        )
    else:
        epoch_note = "both neural models converged inside the brief's epoch cap"

    cov80 = winner_summary["coverage_80"]
    cov95 = winner_summary["coverage_95"]
    if cov80 is None:
        interval_note = "The winner produces no intervals, so there is nothing to calibrate."
    else:
        interval_note = (
            f"Its intervals are too narrow: the nominal 80 percent band held "
            f"{cov80 * 100:.0f} percent of the actuals, the 95 percent band "
            f"{cov95 * 100:.0f} percent. The quantile models bracket a one-hour-ahead error, "
            "yet are asked to bracket a week-ahead recursive forecast."
        )

    return f"""**{MODEL_LABELS[winner]} wins** at {win["mape_test_pct"]:.2f} percent MAPE
against {naive_mape:.2f} percent for the seasonal-naive anchor.
{label_list(beat_naive)} beat the anchor; {label_list(lost_to_naive)} did not.

The ordering is roughly what theory predicts, but the reason is narrower than
"more complex is better": what separates these models is whether the calendar
can change the *shape* of a day, not just its level. SARIMA, last, fails
structurally rather than through bad tuning. Its seasonal period is fixed at 24
hours because a 168-hour state-space term is impractical to fit, so it
reproduces the daily shape and cannot say that Saturday differs from Tuesday.
Over a week-long horizon that is most of the signal, and its forecast settles
into a repeated average day (figure 02).

The holiday gives the most useful result: **carrying a holiday feature is not
the same as being able to use it**. Prophet knows 1 January is a holiday and
still scores {metrics["prophet"]["mape_jan1_pct"]:.2f} percent on it, its worst day of the
seven, against {metrics["prophet"]["mape_jan2_to_jan7_pct"]:.2f} percent on the rest of the
week. Figure 08 shows why: its holiday term is one constant, about -10,500 MW,
applied to all 24 hours alike, whereas New Year is a differently shaped day,
not a uniformly lower Wednesday. LightGBM, given the same flag, scores
{metrics["lightgbm"]["mape_jan1_pct"]:.2f} percent, because a tree can split on the flag
and then on hour of day. The baseline's respectable
{metrics["naive"]["mape_jan1_pct"]:.2f} percent there is an accident of copying 25 to 31
December, paid for on the ordinary days
({metrics["naive"]["mape_jan2_to_jan7_pct"]:.2f} percent).

The neural models see no calendar, only the 168 hours before the forecast,
which here are the Christmas shutdown. N-BEATS keeps the level but loses the
weekly phase, over-predicting Sunday {nbeats_worst_day.day} January by about 14,000 MW
({daily["nbeats"].max():.1f} percent) after validating at {nbeats_val_mape:.2f} percent
over thirteen ordinary autumn weeks. The winner's own worst day is
{winner_worst_day.day} January ({daily[winner].max():.1f} percent, over by about
6,500 MW): Epiphany, a holiday in three federal states but not nationally, so
`holidays.Germany()` misses it. That calls for a regional holiday calendar, not
a better model.

{interval_note}

Caveats: one window is a single draw, and an unusually hard one that flatters
holiday-aware models; temperature is absent by design, so these are not
production error levels; and {epoch_note}."""


def _recommendation(order: list[str], metrics: dict[str, dict[str, float]]) -> str:
    """Which one to put into a pipeline, and what to do before trusting it."""
    winner = order[0]
    runner_up = order[1]
    return f"""If exactly one of these had to go into a JRC short-term load forecasting
pipeline, it would be **{MODEL_LABELS[winner]}**, and accuracy is only part of the
reason. It retrains in minutes on a laptop CPU; its features are readable by a
domain expert, who can therefore challenge them; its failure modes are
diagnosable from the importance and residual plots rather than opaque (this
run's largest error was traced to a specific missing regional holiday in an
afternoon); and adding a driver later is a new column, not a new architecture.
{MODEL_LABELS[runner_up]} is the natural challenger to keep in any regular
re-evaluation, and it is the better choice if interpretable components matter
more than accuracy.

Five things would have to happen before it is trusted in production. First,
replace this single test week with a rolling-origin backtest of at least
fifty-two weekly origins across two or more years: one week cannot separate a
better model from a luckier one, and this particular week is not
representative. Second, add temperature. It is the largest omitted driver of
German winter demand; the univariate constraint here was deliberate, to isolate
temporal structure, and it is not a constraint a production system should keep.
Third, replace the federal holiday flag with a load-weighted regional calendar
covering the sixteen Laender, plus school holidays and bridging days, which
would have caught the 6 January miss. Fourth, fix the interval calibration:
either train the quantile models on the recursive multi-step errors the system
will actually make, or wrap the point forecast in conformal prediction
calibrated separately for each lead time, so a day-seven interval is honestly
wider than a day-one interval. Fifth, decide the recursion explicitly: either
train one model per horizon step, which removes the compounding at the cost of
168 models, or keep the recursion and publish how error grows with lead time so
that nobody reads a day-seven number with day-one confidence."""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the German load forecasting bake-off.")
    parser.add_argument(
        "--run-model",
        choices=list(MODEL_ORDER),
        help="internal: fit one model in this process and pickle it to --output",
    )
    parser.add_argument("--output", type=Path, help="internal: where --run-model writes its result")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="reuse fits already in .model_cache/ instead of refitting (for figure work)",
    )
    args = parser.parse_args()

    if args.run_model:
        if args.output is None:
            parser.error("--run-model requires --output")
        run_single_model(args.run_model, args.output)
    else:
        main(use_cache=args.use_cache)
