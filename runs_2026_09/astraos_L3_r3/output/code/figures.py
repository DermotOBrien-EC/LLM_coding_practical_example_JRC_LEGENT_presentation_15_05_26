from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter

from .common import COLORS, LABELS, NAMES, ROOT, TEST_INDEX, ForecastResult, mape

SHORT = {
    "naive": "Seasonal\nnaive",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "Transformer*",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#aaaaaa",
            "axes.linewidth": 0.6,
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "grid.color": "#e4e4e4",
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(figure: Any, name: str) -> None:
    figure.savefig(ROOT / "figures" / name, dpi=300)
    plt.close(figure)


def time_axis(axis: Any, full: bool = False) -> None:
    if full:
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        axis.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axis.set_xlim(TEST_INDEX[0], TEST_INDEX[-1])
    axis.set_xlabel("Date (UTC)")
    axis.set_ylabel("Electricity load (MW)")
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.grid(axis="y")


def make_figures(data: pd.Series, results: list[ForecastResult], summary: dict[str, Any]) -> None:
    style()
    by_name = {result.name: result for result in results}
    rows = {row["name"]: row for row in summary["models"]}
    order = [row["name"] for row in summary["models"]]
    actual = data.loc[TEST_INDEX].to_numpy()
    winner = by_name[summary["winner"]]
    color = COLORS[winner.name]

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    ax.plot(data.index, data.values, color="#c5c5c5", linewidth=0.45, label="Hourly observed load")
    ax.axvspan(TEST_INDEX[0], TEST_INDEX[-1] + pd.Timedelta(hours=1), color="#d62728", alpha=0.4)
    ax.plot(TEST_INDEX, actual, color="#d62728", linewidth=1, label="Held-out test: Jan 1-7, 2020")
    ax.annotate(
        "168-hour test",
        xy=(TEST_INDEX[0], 85000),
        xytext=(pd.Timestamp("2018-10-01", tz="UTC"), 96000),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        fontsize=10,
    )
    ax.set_ylim(25000, max(100000, float(data.max()) * 1.06))
    ax.set_title("German national hourly electricity load, 2015-2020")
    time_axis(ax, full=True)
    ax.legend(loc="lower left", ncol=2)
    save(fig, "01_overview.png")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True, layout="constrained")
    low = min(float(actual.min()), *(float(result.point.min()) for result in results))
    high = max(float(actual.max()), *(float(result.point.max()) for result in results))
    for ax, name in zip(axes.flat, NAMES):
        result = by_name[name]
        ax.axvspan(TEST_INDEX[0], TEST_INDEX[24], color="#eeeeee", zorder=0)
        ax.plot(TEST_INDEX, actual, color="black", linewidth=1.2, label="Observed")
        ax.plot(
            TEST_INDEX,
            result.point,
            color=COLORS[name],
            linewidth=1.5,
            label="Forecast",
            linestyle="--",
        )
        ax.set_title(f"{LABELS[name]}\nMAPE {rows[name]['mape_test_pct']:.2f}%", fontsize=10)
        ax.set_ylim(low - 1500, high + 1500)
        time_axis(ax)
        ax.tick_params(labelsize=8)
        ax.legend(loc="upper left", ncol=2, fontsize=7)
    fig.suptitle("One forecast origin, six models: 1-7 January 2020", fontsize=14)
    save(fig, "02_forecast_comparison.png")

    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    metrics = [
        ("mape_test_pct", "MAPE (%)", ""),
        ("rmse_test_mw", "RMSE (MW)", "///"),
        ("mae_test_mw", "MAE (MW)", "\\\\\\"),
    ]
    x = np.arange(6)
    maximum = 0.0
    for k, (key, label, hatch) in enumerate(metrics):
        values = np.array([rows[name][key] / rows["naive"][key] for name in order])
        positions = x + (k - 1) * 0.25
        bars = ax.bar(
            positions,
            values,
            width=0.21,
            color=[COLORS[name] for name in order],
            hatch=hatch,
            edgecolor="white",
            linewidth=0.3,
        )
        maximum = max(maximum, float(values.max()))
        for bar, name in zip(bars, order):
            raw = rows[name][key]
            label_text = f"{raw:.2f}%" if k == 0 else f"{raw:,.0f}"
            ax.annotate(
                label_text,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.axhline(1.0, color="#666666", linewidth=0.8)
    ax.set_ylim(0, maximum * 1.4)
    ax.set_xticks(x, [SHORT[name] for name in order], rotation=30, ha="right", fontsize=9)
    ax.set_xlabel("Model (sorted by test MAPE)")
    ax.set_ylabel("Error / seasonal-naive error (ratio)")
    ax.set_title(
        "Three error metrics, one consistent scale\nBar labels give raw % or MW", fontsize=12
    )
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    handles = [
        Patch(facecolor="#999999", edgecolor="white", hatch=hatch, label=label)
        for _, label, hatch in metrics
    ]
    ax.legend(handles=handles, loc="upper left", ncol=1)
    fig.supxlabel("*Transformer substitutes for unavailable PatchTST", fontsize=8)
    save(fig, "03_metric_comparison.png")

    daily = np.array(
        [
            [
                mape(
                    actual[day * 24 : (day + 1) * 24],
                    by_name[name].point[day * 24 : (day + 1) * 24],
                )
                for day in range(7)
            ]
            for name in order
        ]
    )
    pd.DataFrame(daily.T, index=pd.date_range("2020-01-01", periods=7), columns=order).to_csv(
        ROOT / "artifacts" / "per_day_mape.csv", index_label="date_utc"
    )
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    image = ax.imshow(daily, cmap="Blues", vmin=0, vmax=float(daily.max()), aspect="auto")
    labels = [
        "Jan 1*\nWed",
        "Jan 2\nThu",
        "Jan 3\nFri",
        "Jan 4\nSat",
        "Jan 5\nSun",
        "Jan 6\nMon",
        "Jan 7\nTue",
    ]
    ax.set_xticks(range(7), labels, fontsize=8)
    ax.set_yticks(
        range(6),
        [SHORT[name].replace("\n", " ").replace("*", " (sub.)") for name in order],
        fontsize=9,
    )
    for row in range(6):
        for column in range(7):
            ink = "white" if daily[row, column] > daily.max() * 0.56 else "#222222"
            ax.text(
                column,
                row,
                f"{daily[row, column]:.1f}",
                ha="center",
                va="center",
                color=ink,
                fontsize=9,
            )
    ax.set_xlabel("Day of test week (UTC); *federal public holiday")
    ax.set_ylabel("Model (sorted by whole-week MAPE)")
    ax.set_title("Where each model misses: daily MAPE", fontsize=12)
    fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        pad=0.08,
        shrink=0.8,
        label="Mean absolute percentage error (%)",
    )
    save(fig, "04_per_day_mape.png")

    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    if winner.quantiles is not None:
        q = winner.quantiles
        ax.fill_between(
            TEST_INDEX, q[:, 0], q[:, 4], color=color, alpha=0.12, label="95% prediction interval"
        )
        ax.fill_between(
            TEST_INDEX, q[:, 1], q[:, 3], color=color, alpha=0.27, label="80% prediction interval"
        )
        subtitle = f"Empirical coverage: 80% nominal = {summary['winner_coverage_80pct']:.1%}; 95% nominal = {summary['winner_coverage_95pct']:.1%}"
    else:
        subtitle = "Seasonal-naive baseline has no prediction intervals; coverage is not available"
    ax.plot(TEST_INDEX, actual, color="black", linewidth=1.6, label="Observed")
    ax.plot(
        TEST_INDEX, winner.point, color=color, linewidth=1.8, linestyle="--", label="Point forecast"
    )
    ax.set_title(
        f"{LABELS[winner.name]}: test MAPE {rows[winner.name]['mape_test_pct']:.2f}%\n{subtitle}",
        fontsize=12,
    )
    time_axis(ax)
    ax.legend(loc="upper left", ncol=2)
    save(fig, "05_winner_with_intervals.png")

    residual = actual - winner.point
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained"
    )
    groups = [
        [residual[TEST_INDEX.hour == hour] for hour in range(24)],
        [residual[day * 24 : (day + 1) * 24] for day in range(7)],
    ]
    for ax, values in zip(axes, groups):
        boxes = ax.boxplot(
            values,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1},
            flierprops={"marker": ".", "markersize": 3, "markeredgecolor": "#555555"},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.35)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y")
        ax.set_ylabel("Observed - forecast (MW)")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[0].set_xticks(np.arange(1, 25, 3), np.arange(0, 24, 3))
    axes[0].set_xlabel("Hour of day (UTC); 7 observations per box")
    axes[0].set_title("Residuals by hour of day")
    axes[1].set_xticks(
        range(1, 8),
        [
            "Wed\nJan 1",
            "Thu\nJan 2",
            "Fri\nJan 3",
            "Sat\nJan 4",
            "Sun\nJan 5",
            "Mon\nJan 6",
            "Tue\nJan 7",
        ],
        fontsize=8,
    )
    axes[1].set_xlabel("Day of test week (UTC); 24 observations per box")
    axes[1].set_title("Residuals by calendar day")
    fig.suptitle(f"{LABELS[winner.name]}: positive residuals mean underprediction")
    save(fig, "06_residuals.png")

    if "lightgbm" in order[:2]:
        gain = by_name["lightgbm"].diagnostics["gain_importance"]
        features = sorted(gain, key=gain.get)
        total = sum(gain.values())
        values = [100 * gain[feature] / total for feature in features]
        fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
        bars = ax.barh(features, values, height=0.55, color=COLORS["lightgbm"])
        ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=8)
        ax.set_xlim(0, max(values) * 1.18)
        ax.set_xlabel("Share of total split gain (%)")
        ax.set_ylabel("Engineered feature")
        ax.set_title("LightGBM: fitted median-model feature importance")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)
        save(fig, "07_feature_importance.png")

    if "prophet" in order[:2]:
        diagnostic = by_name["prophet"].diagnostics
        daily_components = diagnostic["daily_components"]
        dates = pd.to_datetime(daily_components["dates"])
        trend = np.asarray(daily_components["trend"])
        yearly = np.asarray(daily_components["yearly"])
        mask = dates.year == 2019
        fig, axes = plt.subplots(3, 1, figsize=(11, 6), layout="constrained")
        axes[0].plot(dates, trend, color=COLORS["prophet"])
        axes[0].set_title("Fitted trend")
        axes[0].set_xlabel("Date (UTC)")
        axes[0].set_ylabel("Trend load (MW)")
        weekly = diagnostic["weekly_components"]
        axes[1].plot(weekly["hours_since_monday"], weekly["weekly"], color=COLORS["prophet"])
        axes[1].set_xticks(np.arange(0, 168, 24), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        axes[1].set_title("Weekly component")
        axes[1].set_xlabel("Day of week (UTC)")
        axes[1].set_ylabel("Additive effect (MW)")
        axes[2].plot(dates[mask], yearly[mask], color=COLORS["prophet"])
        axes[2].set_title("Yearly component (2019 calendar)")
        axes[2].set_xlabel("Month (UTC)")
        axes[2].set_ylabel("Additive effect (MW)")
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        for ax in axes:
            ax.grid(axis="y")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        fig.suptitle("Prophet: fitted components, not causal explanations")
        save(fig, "08_decomposition.png")
