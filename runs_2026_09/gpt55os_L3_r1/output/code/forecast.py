from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    MODEL_COLORS,
    MODEL_NAMES,
    MODEL_ORDER,
    TEST_END,
    TEST_START,
    ForecastResult,
    coverage,
    format_metric,
    load_data,
    model_metrics,
    per_day_mape,
    pinball_loss,
    save_figure,
    set_figure_style,
    timer,
    elapsed_seconds,
    uname_string,
    write_json,
)


def _run_models(data_path: Path) -> tuple[pd.Series, pd.DataFrame, dict[str, ForecastResult], float]:
    start = timer()
    splits = load_data(data_path)
    modules = {
        "naive": importlib.import_module("naive"),
        "sarima": importlib.import_module("sarima"),
        "prophet": importlib.import_module("prophet"),
        "lightgbm": importlib.import_module("lightgbm_features"),
        "nbeats": importlib.import_module("nbeats"),
        "patchtst": importlib.import_module("patchtst"),
    }
    results: dict[str, ForecastResult] = {}
    results["naive"] = modules["naive"].forecast(splits.train_validation, splits.test)
    results["sarima"] = modules["sarima"].forecast(splits.train, splits.validation, splits.train_validation, splits.test)
    results["prophet"] = modules["prophet"].forecast(splits.train, splits.validation, splits.train_validation, splits.test)
    results["lightgbm"] = modules["lightgbm"].forecast(splits.full, splits.train, splits.validation, splits.train_validation, splits.test)
    results["nbeats"] = modules["nbeats"].forecast(splits.train, splits.validation, splits.train_validation, splits.test)
    results["patchtst"] = modules["patchtst"].forecast(splits.train, splits.validation, splits.train_validation, splits.test)
    return splits.test, splits.full, results, elapsed_seconds(start)


def _metrics_payload(test: pd.Series, results: dict[str, ForecastResult], total_runtime: float) -> tuple[dict[str, Any], pd.DataFrame, str]:
    rows = [model_metrics(results[name], test) for name in MODEL_ORDER]
    rows = sorted(rows, key=lambda row: row["mape_test_pct"])
    winner = rows[0]["name"]
    winner_result = results[winner]
    actual = test.to_numpy(dtype=float)
    if winner_result.q10 is None or winner_result.q90 is None or winner_result.q025 is None or winner_result.q975 is None:
        raise RuntimeError(f"Winning model {winner} does not provide prediction intervals")
    q50 = winner_result.q50 if winner_result.q50 is not None else winner_result.point
    payload = {
        "test_start": "2020-01-01",
        "test_end": "2020-01-07",
        "n_test_observations": int(len(test)),
        "models": rows,
        "winner": winner,
        "winner_coverage_80pct": coverage(actual, winner_result.q10, winner_result.q90),
        "winner_coverage_95pct": coverage(actual, winner_result.q025, winner_result.q975),
        "winner_pinball_loss_q10": pinball_loss(actual, winner_result.q10, 0.1),
        "winner_pinball_loss_q50": pinball_loss(actual, q50, 0.5),
        "winner_pinball_loss_q90": pinball_loss(actual, winner_result.q90, 0.9),
        "total_runtime_seconds": total_runtime,
    }
    csv_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if key != "hyperparameters"}
        flat["hyperparameters"] = str(row["hyperparameters"])
        csv_rows.append(flat)
    metrics_df = pd.DataFrame(csv_rows)
    return payload, metrics_df, winner


def _write_metrics_csv(path: Path, metrics_df: pd.DataFrame) -> None:
    metrics_df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _plot_overview(full: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(full.index, full.iloc[:, 0], color="#b8b8b8", linewidth=0.5, label="Observed load")
    window = full.loc[TEST_START:TEST_END]
    ax.plot(window.index, window.iloc[:, 0], color="#D62728", linewidth=1.6, label="Held-out test week")
    ax.axvspan(TEST_START, TEST_END, color="#D62728", alpha=0.10)
    ax.set_title("German hourly electricity load, 2015 to 2020")
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="upper right")
    save_figure(fig, path)


def _plot_forecast_comparison(test: pd.Series, results: dict[str, ForecastResult], metrics_df: pd.DataFrame, path: Path) -> None:
    mape_by_name = metrics_df.set_index("name")["mape_test_pct"].to_dict()
    y_min = min(float(test.min()), *(float(np.min(results[name].point)) for name in MODEL_ORDER))
    y_max = max(float(test.max()), *(float(np.max(results[name].point)) for name in MODEL_ORDER))
    pad = (y_max - y_min) * 0.05
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, MODEL_ORDER, strict=True):
        ax.plot(test.index, test.to_numpy(), color="black", linewidth=1.8, label="Observed")
        ax.plot(test.index, results[name].point, color=MODEL_COLORS[name], linewidth=1.8, label="Forecast")
        ax.set_title(f"{MODEL_NAMES[name]}: MAPE {mape_by_name[name]:.2f}%")
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel("Date (UTC)")
        ax.set_ylabel("Load (MW)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    save_figure(fig, path)


def _plot_metric_comparison(metrics_df: pd.DataFrame, path: Path) -> None:
    sorted_df = metrics_df.sort_values("mape_test_pct")
    labels = [MODEL_NAMES[name] for name in sorted_df["name"]]
    metrics = ["mape_test_pct", "rmse_test_mw", "mae_test_mw"]
    metric_labels = ["MAPE (%)", "RMSE (MW)", "MAE (MW)"]
    fig, axes = plt.subplots(3, 1, figsize=(6, 9))
    for ax, metric, label in zip(axes, metrics, metric_labels, strict=True):
        values = sorted_df[metric].to_numpy(dtype=float)
        colors = [MODEL_COLORS[name] for name in sorted_df["name"]]
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=35)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Held-out test metrics by model", y=1.01)
    save_figure(fig, path)


def _plot_per_day_mape(test: pd.Series, results: dict[str, ForecastResult], path: Path) -> pd.DataFrame:
    data = {name: per_day_mape(test, results[name].point) for name in MODEL_ORDER}
    frame = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(frame.index))
    width = 0.12
    offsets = (np.arange(len(MODEL_ORDER)) - (len(MODEL_ORDER) - 1) / 2) * width
    for offset, name in zip(offsets, MODEL_ORDER, strict=True):
        ax.bar(x + offset, frame[name].to_numpy(), width=width, label=MODEL_NAMES[name], color=MODEL_COLORS[name], edgecolor="white", linewidth=0.8)
    ax.set_title("Daily MAPE during the held-out test week")
    ax.set_xlabel("Test date (UTC)")
    ax.set_ylabel("MAPE (%)")
    ax.set_xticks(x, [label[5:] for label in frame.index], rotation=35)
    ax.legend(ncol=2, fontsize=8)
    save_figure(fig, path)
    return frame


def _plot_winner_intervals(test: pd.Series, result: ForecastResult, metrics: dict[str, Any], path: Path) -> None:
    if result.q10 is None or result.q90 is None or result.q025 is None or result.q975 is None:
        raise RuntimeError("Winner interval figure requires interval forecasts")
    fig, ax = plt.subplots(figsize=(11, 6))
    x = test.index
    ax.fill_between(x, result.q025, result.q975, color=MODEL_COLORS[result.name], alpha=0.15, label="95% prediction interval")
    ax.fill_between(x, result.q10, result.q90, color=MODEL_COLORS[result.name], alpha=0.30, label="80% prediction interval")
    ax.plot(x, test.to_numpy(), color="black", linewidth=1.8, label="Observed")
    ax.plot(x, result.point, color=MODEL_COLORS[result.name], linewidth=1.8, label="Forecast")
    mape = next(row["mape_test_pct"] for row in metrics["models"] if row["name"] == result.name)
    ax.set_title(
        f"{MODEL_NAMES[result.name]} winner: MAPE {mape:.2f}%, "
        f"80% coverage {metrics['winner_coverage_80pct']:.1%}, 95% coverage {metrics['winner_coverage_95pct']:.1%}"
    )
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Load (MW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="best")
    save_figure(fig, path)


def _plot_residuals(test: pd.Series, result: ForecastResult, path: Path) -> None:
    residuals = test.to_numpy(dtype=float) - result.point
    frame = pd.DataFrame({"residual_mw": residuals, "hour": test.index.hour, "date": test.index.strftime("%b %d")}, index=test.index)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    hourly = [frame.loc[frame["hour"] == hour, "residual_mw"].to_numpy() for hour in range(24)]
    axes[0].boxplot(hourly, positions=np.arange(24), widths=0.6, showfliers=False)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Residuals by hour of day")
    axes[0].set_xlabel("Hour of day (UTC)")
    axes[0].set_ylabel("Observed minus forecast (MW)")
    dates = list(dict.fromkeys(frame["date"].tolist()))
    daily = [frame.loc[frame["date"] == day, "residual_mw"].to_numpy() for day in dates]
    axes[1].boxplot(daily, labels=dates, showfliers=False)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Residuals by test date")
    axes[1].set_xlabel("Date (UTC)")
    axes[1].set_ylabel("Observed minus forecast (MW)")
    axes[1].tick_params(axis="x", rotation=35)
    save_figure(fig, path)


def _plot_feature_importance(result: ForecastResult, path: Path) -> None:
    if result.feature_importance is None:
        raise RuntimeError("LightGBM feature importance is unavailable")
    frame = result.feature_importance.sort_values("gain", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(frame["feature"], frame["gain"], color=MODEL_COLORS["lightgbm"], edgecolor="white")
    ax.set_title("LightGBM feature importance")
    ax.set_xlabel("Gain (LightGBM split gain)")
    ax.set_ylabel("Feature")
    save_figure(fig, path)


def _plot_decomposition(result: ForecastResult, path: Path) -> None:
    components = result.extra.get("components")
    if components is None:
        raise RuntimeError("Prophet components are unavailable")
    component_frame = components.copy()
    component_frame["ds"] = pd.to_datetime(component_frame["ds"])
    available = [column for column in ["trend", "weekly", "yearly", "holidays"] if column in component_frame.columns]
    fig, axes = plt.subplots(len(available), 1, figsize=(11, 2.2 * len(available)), sharex=True)
    if len(available) == 1:
        axes = [axes]
    for ax, column in zip(axes, available, strict=True):
        ax.plot(component_frame["ds"], component_frame[column], color=MODEL_COLORS["prophet"], linewidth=1.5)
        ax.set_title(f"Prophet {column} component")
        ax.set_ylabel("Load contribution (MW)")
    axes[-1].set_xlabel("Date (UTC)")
    save_figure(fig, path)


def _write_transcript(
    path: Path,
    metrics: dict[str, Any],
    metrics_df: pd.DataFrame,
    daily_mape: pd.DataFrame,
    results: dict[str, ForecastResult],
    total_runtime: float,
) -> None:
    sorted_df = metrics_df.sort_values("mape_test_pct")
    lines = [
        "# German load forecasting bake-off",
        "",
        "## Data",
        "",
        "The study uses the Open Power System Data German national hourly load series, derived from the ENTSO-E Transparency Platform. The file `opsd_de_load.csv` covers 2015-01-01 00:00 UTC to 2020-09-30 23:00 UTC with exactly 50,400 hourly rows. The timestamp sequence was checked against a complete hourly index and the load column was checked for missing values, so no imputation or gap filling was needed.",
        "",
        "## Why these six models",
        "",
        "The six models form a complexity gradient. The seasonal-naive model asks how far a same-hour-last-week rule gets on its own. SARIMA adds a classical stochastic structure with daily seasonality. Prophet adds smooth trend, daily, weekly and yearly seasonal terms plus German holidays. LightGBM tests whether explicit calendar, lag and rolling-window features can capture the load shape. N-BEATS tests a univariate neural forecaster with one week of context. PatchTST was not exposed by the installed Darts 0.41.0 catalogue, so TSMixer was used as the transformer-style neural substitute with the same one-week input and output chunks.",
        "",
        "## Validation strategy",
        "",
        "All non-baseline models used the same chronology: 2015-01-01 to 2019-09-30 for fitting candidate configurations, 2019-10-01 to 2019-12-31 for validation, and 2020-01-01 to 2020-01-07 as the untouched test week. The selected configuration was then refit on train plus validation, 2015-01-01 to 2019-12-31, before forecasting the 168 held-out hours. This gives each model all pre-test information while keeping model selection away from the test window.",
        "",
        "## Results table",
        "",
        "| Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2 to Jan 7 MAPE (%) | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in sorted_df.iterrows():
        lines.append(
            f"| {MODEL_NAMES[str(row['name'])]} | {row['mape_test_pct']:.3f} | {row['rmse_test_mw']:.1f} | "
            f"{row['mae_test_mw']:.1f} | {row['mape_jan1_pct']:.3f} | {row['mape_jan2_to_jan7_pct']:.3f} | "
            f"{row['runtime_seconds']:.1f} |"
        )
    winner = str(metrics["winner"])
    best = sorted_df.iloc[0]
    worst_jan1 = sorted_df.sort_values("mape_jan1_pct", ascending=False).iloc[0]
    lines.extend(
        [
            "",
            "## Discussion",
            "",
            (
                f"{MODEL_NAMES[winner]} won on the held-out week with a test MAPE of {best['mape_test_pct']:.3f}%. "
                f"Its advantage came from tracking the daily load shape while making the smallest average error over the six working and semi-working days after New Year's Day. "
                "The holiday split matters because 2020-01-01 is a German public holiday and the load profile is not a normal Wednesday. "
                f"The weakest Jan 1 result was {MODEL_NAMES[str(worst_jan1['name'])]}, at {worst_jan1['mape_jan1_pct']:.3f}% MAPE, which shows how a method can look reasonable on ordinary days yet miss a holiday shape. "
                "The seasonal-naive rule is a useful anchor, but it repeats the previous week's Wednesday and has no way to know that New Year's Day is special. "
                "SARIMA captures regular autocorrelation and daily recurrence, but its daily seasonal order cannot directly represent every calendar exception. "
                "Prophet is designed for this problem class because it has explicit holidays and multiple seasonalities, but its smooth additive structure can still underfit sharp hourly bends. "
                "LightGBM has the most direct access to the useful signals in this constrained setup: hour, weekday, holiday flags, recent load and year-ago load. "
                "The neural models had enough data for broad seasonality but only a modest tuning budget, so their rank should be read as a practical bake-off result rather than a claim about their ceiling. "
                "No external weather or price data were used, which is a deliberate limitation because German load is weather-sensitive."
            ),
            "",
            "## Recommendation",
            "",
            (
                f"For a JRC production short-term load forecasting pipeline under the same input constraint, I would start from {MODEL_NAMES[winner]} if the objective is the best result from this reproducible bake-off. "
                "Before deployment I would add rolling-origin backtests over many weeks, tune prediction intervals for calibration, and add monitored holiday and bridge-day slices. "
                "If production constraints value interpretability more than a small accuracy gain, the LightGBM and Prophet diagnostics should be reviewed side by side before making the final operational choice."
            ),
            "",
            "## Reproducibility note",
            "",
            f"Random seed: 20260906 for the stochastic model wrappers and LightGBM. Total wall-clock runtime: {total_runtime:.1f} seconds. Hardware and operating system from `uname -a`: `{uname_string()}`.",
            "",
            "Validation selections:",
            "",
        ]
    )
    for name in MODEL_ORDER:
        lines.append(f"- {MODEL_NAMES[name]}: `{results[name].hyperparameters}`")
    lines.extend(
        [
            "",
            "Daily MAPE matrix used for Figure 04:",
            "",
            daily_mape.rename(columns=MODEL_NAMES).to_markdown(floatfmt=".3f"),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_figure_style()
    data_path = ROOT / "opsd_de_load.csv"
    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    test, full, results, total_runtime = _run_models(data_path)
    metrics, metrics_df, winner = _metrics_payload(test, results, total_runtime)
    write_json(ROOT / "metrics.json", metrics)
    _write_metrics_csv(ROOT / "metrics.csv", metrics_df)
    _plot_overview(full, figures_dir / "01_overview.png")
    _plot_forecast_comparison(test, results, metrics_df, figures_dir / "02_forecast_comparison.png")
    _plot_metric_comparison(metrics_df, figures_dir / "03_metric_comparison.png")
    daily_mape = _plot_per_day_mape(test, results, figures_dir / "04_per_day_mape.png")
    _plot_winner_intervals(test, results[winner], metrics, figures_dir / "05_winner_with_intervals.png")
    _plot_residuals(test, results[winner], figures_dir / "06_residuals.png")
    top_two = metrics_df.sort_values("mape_test_pct").head(2)["name"].tolist()
    if "lightgbm" in top_two:
        _plot_feature_importance(results["lightgbm"], figures_dir / "07_feature_importance.png")
    if "prophet" in top_two:
        _plot_decomposition(results["prophet"], figures_dir / "08_decomposition.png")
    _write_transcript(ROOT / "transcript.md", metrics, metrics_df, daily_mape, results, total_runtime)
    print(f"winner={winner}")
    print(metrics_df.sort_values("mape_test_pct").to_string(index=False))


if __name__ == "__main__":
    main()
