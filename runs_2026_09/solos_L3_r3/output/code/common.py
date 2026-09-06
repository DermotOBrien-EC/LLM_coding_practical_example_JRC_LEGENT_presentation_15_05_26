from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch

LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
TIMESTAMP_COLUMN = "utc_timestamp"
TRAIN_START = pd.Timestamp("2015-01-01 00:00:00")
VALIDATION_START = pd.Timestamp("2019-10-01 00:00:00")
TEST_START = pd.Timestamp("2020-01-01 00:00:00")
TEST_END_EXCLUSIVE = pd.Timestamp("2020-01-08 00:00:00")
EXPECTED_END_EXCLUSIVE = pd.Timestamp("2020-10-01 00:00:00")
SEED = 42
QUANTILES = (0.025, 0.1, 0.5, 0.9, 0.975)
MODEL_ORDER = ("naive", "sarima", "prophet", "lightgbm", "nbeats", "patchtst")
MODEL_LABELS = {
    "naive": "Seasonal naive",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "PatchTST / TSMixer",
}
MODEL_COLORS = {
    "naive": "#1f77b4",
    "sarima": "#ff7f0e",
    "prophet": "#9467bd",
    "lightgbm": "#2ca02c",
    "nbeats": "#e377c2",
    "patchtst": "#d62728",
}


@dataclass(frozen=True)
class DataSplits:
    full: pd.DataFrame
    train: pd.Series
    validation: pd.Series
    train_validation: pd.Series
    test: pd.Series


@dataclass
class ForecastResult:
    name: str
    point: np.ndarray
    runtime_seconds: float
    hyperparameters: dict[str, Any]
    quantiles: dict[float, np.ndarray] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def quantile(self, probability: float) -> np.ndarray:
        if probability not in self.quantiles:
            raise KeyError(f"{self.name} has no {probability:g} quantile")
        return np.asarray(self.quantiles[probability], dtype=float)


def load_and_split_data(csv_path: Path) -> DataSplits:
    frame = pd.read_csv(csv_path)
    required = {TIMESTAMP_COLUMN, LOAD_COLUMN}
    if set(frame.columns) != required:
        raise ValueError(f"Expected columns {sorted(required)}, found {frame.columns.tolist()}")

    timestamps = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True, errors="raise")
    if len(frame) != 50_400:
        raise ValueError(f"Expected 50,400 rows, found {len(frame):,}")
    if frame[LOAD_COLUMN].isna().any():
        raise ValueError("Load series contains missing values")
    if timestamps.duplicated().any():
        raise ValueError("Timestamp column contains duplicates")
    expected = pd.date_range(
        TRAIN_START.tz_localize("UTC"),
        EXPECTED_END_EXCLUSIVE.tz_localize("UTC"),
        freq="h",
        inclusive="left",
    )
    if not timestamps.equals(pd.Series(expected, name=TIMESTAMP_COLUMN)):
        raise ValueError("Timestamp sequence is not the expected complete hourly UTC grid")

    frame[TIMESTAMP_COLUMN] = timestamps.dt.tz_convert(None)
    frame = frame.set_index(TIMESTAMP_COLUMN, drop=False)
    series = frame[LOAD_COLUMN].astype(float)
    train = series.loc[TRAIN_START : VALIDATION_START - pd.Timedelta(hours=1)]
    validation = series.loc[VALIDATION_START : TEST_START - pd.Timedelta(hours=1)]
    train_validation = series.loc[TRAIN_START : TEST_START - pd.Timedelta(hours=1)]
    test = series.loc[TEST_START : TEST_END_EXCLUSIVE - pd.Timedelta(hours=1)]
    if (len(train), len(validation), len(test)) != (41_616, 2_208, 168):
        raise ValueError("Split sizes do not match the preregistered windows")
    return DataSplits(
        full=frame,
        train=train,
        validation=validation,
        train_validation=train_validation,
        test=test,
    )


def mape_pct(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=float)
    forecast_array = np.asarray(forecast, dtype=float)
    if actual_array.shape != forecast_array.shape:
        raise ValueError("Actual and forecast arrays must have equal shapes")
    if np.any(actual_array == 0.0):
        raise ValueError("MAPE is undefined when actual load is zero")
    return float(np.mean(np.abs((actual_array - forecast_array) / actual_array)) * 100.0)


def evaluate_point_forecast(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    errors = np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float)
    return {
        "mape_pct": mape_pct(actual, forecast),
        "rmse_mw": float(np.sqrt(np.mean(np.square(errors)))),
        "mae_mw": float(np.mean(np.abs(errors))),
    }


def pinball_loss(actual: np.ndarray, forecast_quantile: np.ndarray, quantile: float) -> float:
    residual = np.asarray(actual, dtype=float) - np.asarray(forecast_quantile, dtype=float)
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    actual_array = np.asarray(actual, dtype=float)
    return float(np.mean((actual_array >= lower) & (actual_array <= upper)))


def enforce_quantile_order(quantiles: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    probabilities = sorted(quantiles)
    matrix = np.column_stack([np.asarray(quantiles[q], dtype=float) for q in probabilities])
    ordered = np.sort(matrix, axis=1)
    return {q: ordered[:, index] for index, q in enumerate(probabilities)}


def model_metrics(result: ForecastResult, actual: pd.Series) -> dict[str, Any]:
    overall = evaluate_point_forecast(actual.to_numpy(), result.point)
    jan1_mask = actual.index.normalize() == TEST_START
    jan2_mask = ~jan1_mask
    return {
        "name": result.name,
        "mape_test_pct": overall["mape_pct"],
        "rmse_test_mw": overall["rmse_mw"],
        "mae_test_mw": overall["mae_mw"],
        "mape_jan1_pct": mape_pct(actual.to_numpy()[jan1_mask], result.point[jan1_mask]),
        "mape_jan2_to_jan7_pct": mape_pct(actual.to_numpy()[jan2_mask], result.point[jan2_mask]),
        "runtime_seconds": result.runtime_seconds,
        "hyperparameters": result.hyperparameters,
    }


def daily_mape(result: ForecastResult, actual: pd.Series) -> pd.Series:
    absolute_percentage_error = (
        np.abs((actual.to_numpy() - result.point) / actual.to_numpy()) * 100.0
    )
    return (
        pd.Series(absolute_percentage_error, index=actual.index)
        .groupby(actual.index.normalize())
        .mean()
    )


def winner_probabilistic_metrics(result: ForecastResult, actual: pd.Series) -> dict[str, float]:
    actual_values = actual.to_numpy()
    return {
        "winner_coverage_80pct": interval_coverage(
            actual_values, result.quantile(0.1), result.quantile(0.9)
        ),
        "winner_coverage_95pct": interval_coverage(
            actual_values, result.quantile(0.025), result.quantile(0.975)
        ),
        "winner_pinball_loss_q10": pinball_loss(actual_values, result.quantile(0.1), 0.1),
        "winner_pinball_loss_q50": pinball_loss(actual_values, result.quantile(0.5), 0.5),
        "winner_pinball_loss_q90": pinball_loss(actual_values, result.quantile(0.9), 0.9),
    }


def configure_figure_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#c3c2b7",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#e1e0d9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(figure: Figure, path: Path) -> None:
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def plot_overview(full: pd.DataFrame, output_path: Path) -> None:
    configure_figure_style()
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    axis.plot(
        full.index,
        full[LOAD_COLUMN],
        color="#b8b8b8",
        linewidth=0.45,
        label="Hourly load",
        rasterized=True,
    )
    axis.axvspan(
        TEST_START,
        TEST_END_EXCLUSIVE,
        color="#d03b3b",
        alpha=0.32,
        label="Held-out test week",
    )
    axis.set_title("German national hourly electricity load and held-out test window")
    axis.set_xlabel("Time (UTC)")
    axis.set_ylabel("Electricity load (MW)")
    axis.legend(loc="upper right")
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save_figure(figure, output_path)


def plot_forecast_comparison(
    results: dict[str, ForecastResult],
    metrics_by_name: dict[str, dict[str, Any]],
    actual: pd.Series,
    output_path: Path,
) -> None:
    configure_figure_style()
    figure, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    values = [actual.to_numpy()] + [results[name].point for name in MODEL_ORDER]
    y_min = min(float(np.min(value)) for value in values)
    y_max = max(float(np.max(value)) for value in values)
    padding = 0.06 * (y_max - y_min)
    for axis, name in zip(axes.flat, MODEL_ORDER, strict=True):
        axis.plot(actual.index, actual, color="#0b0b0b", linewidth=1.6, label="Observed")
        axis.plot(
            actual.index,
            results[name].point,
            color=MODEL_COLORS[name],
            linewidth=1.55,
            label="Forecast",
        )
        axis.set_title(f"{MODEL_LABELS[name]}\nMAPE {metrics_by_name[name]['mape_test_pct']:.2f}%")
        axis.set_ylim(y_min - padding, y_max + padding)
        axis.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        axis.legend(loc="lower right", fontsize=7.5)
    for axis in axes[:, 0]:
        axis.set_ylabel("Load (MW)")
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (UTC)")
    figure.suptitle("Observed and forecast load during the held-out test week", fontsize=13)
    figure.autofmt_xdate(rotation=0)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(figure, output_path)


def plot_metric_comparison(metrics: list[dict[str, Any]], output_path: Path) -> None:
    configure_figure_style()
    ordered = sorted(metrics, key=lambda row: row["mape_test_pct"])
    names = [row["name"] for row in ordered]
    labels = [MODEL_LABELS[name] for name in names]
    specifications = (
        ("mape_test_pct", "MAPE (%)", ""),
        ("rmse_test_mw", "RMSE (MW)", "///"),
        ("mae_test_mw", "MAE (MW)", "..."),
    )
    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    positions = np.arange(len(names), dtype=float)
    width = 0.24
    for metric_position, (key, label, hatch) in enumerate(specifications):
        raw_values = np.asarray([float(row[key]) for row in ordered])
        relative_values = raw_values / raw_values.max() * 100.0
        bars = axis.bar(
            positions + (metric_position - 1) * width,
            relative_values,
            width=width,
            color=[MODEL_COLORS[name] for name in names],
            edgecolor="#52514e",
            linewidth=0.55,
            hatch=hatch,
        )
        axis.bar_label(
            bars,
            labels=[
                f"{value:.2f}%" if key == "mape_test_pct" else f"{value:,.0f}"
                for value in raw_values
            ],
            padding=2,
            rotation=90,
            fontsize=6.2,
        )
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set_ylim(0.0, 122.0)
    axis.set_xlabel("Model")
    axis.set_ylabel("Relative error within metric (% of worst model)")
    axis.set_title(
        "Test-week errors, sorted by MAPE\n"
        "Bar heights are normalized within each metric; labels show raw values"
    )
    axis.legend(
        handles=[
            Patch(facecolor="white", edgecolor="#52514e", hatch=hatch, label=label)
            for _, label, hatch in specifications
        ],
        loc="upper left",
        ncol=3,
    )
    _save_figure(figure, output_path)


def plot_daily_mape(
    results: dict[str, ForecastResult],
    metrics: list[dict[str, Any]],
    actual: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    configure_figure_style()
    names = [row["name"] for row in sorted(metrics, key=lambda row: row["mape_test_pct"])]
    daily = pd.DataFrame({name: daily_mape(results[name], actual) for name in names}).T
    colors = ["#fff7ec", "#eda100", "#d03b3b", "#6b1d35"]
    colormap = LinearSegmentedColormap.from_list("mape", colors)
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    image = axis.imshow(daily.to_numpy(), aspect="auto", cmap=colormap, vmin=0.0)
    axis.grid(False)
    axis.set_xticks(
        np.arange(daily.shape[1]),
        [timestamp.strftime("%b %d\n%a") for timestamp in daily.columns],
    )
    axis.set_yticks(np.arange(daily.shape[0]), [MODEL_LABELS[name] for name in names])
    axis.set_xlabel("Test date (UTC)")
    axis.set_ylabel("Model")
    axis.set_title("Daily MAPE across the held-out week")
    threshold = float(np.nanmax(daily.to_numpy())) * 0.55
    for row in range(daily.shape[0]):
        for column in range(daily.shape[1]):
            value = float(daily.iloc[row, column])
            axis.text(
                column,
                row,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color="white" if value > threshold else "#0b0b0b",
                fontsize=8,
            )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.8, pad=0.02)
    colorbar.set_label("MAPE (%)")
    _save_figure(figure, output_path)
    return daily


def plot_winner_with_intervals(
    result: ForecastResult,
    metrics: dict[str, Any],
    probabilistic_metrics: dict[str, float],
    actual: pd.Series,
    output_path: Path,
) -> None:
    configure_figure_style()
    figure, axis = plt.subplots(figsize=(11, 6), constrained_layout=True)
    color = MODEL_COLORS[result.name]
    band95 = axis.fill_between(
        actual.index,
        result.quantile(0.025),
        result.quantile(0.975),
        color=color,
        alpha=0.12,
        label="95% prediction interval",
    )
    band80 = axis.fill_between(
        actual.index,
        result.quantile(0.1),
        result.quantile(0.9),
        color=color,
        alpha=0.24,
        label="80% prediction interval",
    )
    forecast_line = axis.plot(
        actual.index,
        result.point,
        color=color,
        linewidth=1.8,
        label="Point forecast",
    )[0]
    observed_line = axis.plot(
        actual.index,
        actual,
        color="#0b0b0b",
        linewidth=1.8,
        label="Observed",
    )[0]
    axis.set_title(
        f"{MODEL_LABELS[result.name]}: test MAPE {metrics['mape_test_pct']:.2f}%\n"
        f"Actual coverage: 80% interval {probabilistic_metrics['winner_coverage_80pct']:.1%}; "
        f"95% interval {probabilistic_metrics['winner_coverage_95pct']:.1%}"
    )
    axis.set_xlabel("Time (UTC)")
    axis.set_ylabel("Electricity load (MW)")
    axis.xaxis.set_major_locator(mdates.DayLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axis.legend(
        [observed_line, forecast_line, band80, band95],
        ["Observed", "Point forecast", "80% prediction interval", "95% prediction interval"],
        loc="lower right",
        ncol=2,
    )
    _save_figure(figure, output_path)


def plot_residuals(result: ForecastResult, actual: pd.Series, output_path: Path) -> None:
    configure_figure_style()
    residuals = actual.to_numpy() - result.point
    residual_frame = pd.DataFrame(
        {
            "residual": residuals,
            "hour": actual.index.hour,
            "day": actual.index.strftime("%a %b %d"),
        },
        index=actual.index,
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 6), constrained_layout=True)
    hour_groups = [
        residual_frame.loc[residual_frame["hour"] == hour, "residual"].to_numpy()
        for hour in range(24)
    ]
    hour_plot = axes[0].boxplot(
        hour_groups,
        positions=np.arange(24),
        widths=0.65,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#0b0b0b", "linewidth": 1.0},
    )
    for patch in hour_plot["boxes"]:
        patch.set_facecolor(MODEL_COLORS[result.name])
        patch.set_alpha(0.65)
    axes[0].axhline(0.0, color="#52514e", linewidth=1.0)
    axes[0].set_xticks(
        np.arange(0, 24, 3),
        [str(hour) for hour in range(0, 24, 3)],
    )
    axes[0].set_xlabel("Hour of day (UTC)")
    axes[0].set_ylabel("Residual: observed minus forecast (MW)")
    axes[0].set_title("Residuals by hour of day")

    days = list(dict.fromkeys(residual_frame["day"].tolist()))
    day_groups = [
        residual_frame.loc[residual_frame["day"] == day, "residual"].to_numpy() for day in days
    ]
    day_plot = axes[1].boxplot(
        day_groups,
        positions=np.arange(len(days)),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#0b0b0b", "linewidth": 1.0},
    )
    for patch in day_plot["boxes"]:
        patch.set_facecolor(MODEL_COLORS[result.name])
        patch.set_alpha(0.65)
    axes[1].axhline(0.0, color="#52514e", linewidth=1.0)
    axes[1].set_xticks(np.arange(len(days)), days, rotation=35, ha="right")
    axes[1].set_xlabel("Day in test week (UTC)")
    axes[1].set_ylabel("Residual: observed minus forecast (MW)")
    axes[1].set_title("Residuals by day")
    figure.suptitle(f"Where {MODEL_LABELS[result.name]} over- and under-predicts", fontsize=13)
    _save_figure(figure, output_path)


def plot_feature_importance(result: ForecastResult, output_path: Path) -> None:
    importance = result.extras.get("feature_importance")
    if not isinstance(importance, pd.DataFrame):
        raise ValueError("LightGBM result has no feature-importance table")
    ordered = importance.sort_values("gain_pct", ascending=True)
    configure_figure_style()
    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    bars = axis.barh(
        ordered["feature"],
        ordered["gain_pct"],
        color=MODEL_COLORS["lightgbm"],
        edgecolor="white",
    )
    axis.bar_label(bars, labels=[f"{value:.1f}%" for value in ordered["gain_pct"]], padding=2)
    axis.set_xlim(0.0, max(float(ordered["gain_pct"].max()) * 1.2, 1.0))
    axis.set_xlabel("Share of total gain (%)")
    axis.set_ylabel("Feature")
    axis.set_title("LightGBM feature importance after pre-test refit")
    _save_figure(figure, output_path)


def plot_prophet_decomposition(result: ForecastResult, output_path: Path) -> None:
    components = result.extras.get("decomposition")
    if not isinstance(components, pd.DataFrame):
        raise ValueError("Prophet result has no decomposition table")
    configure_figure_style()
    figure, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    for axis, column, title in zip(
        axes,
        ("trend", "weekly", "yearly"),
        ("Trend", "Weekly component", "Yearly component"),
        strict=True,
    ):
        axis.plot(
            components.index,
            components[column],
            color=MODEL_COLORS["prophet"],
            linewidth=1.8,
        )
        axis.set_ylabel("Load contribution (MW)")
        axis.set_title(title, loc="left", fontsize=10)
    axes[-1].set_xlabel("Time (UTC)")
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    figure.suptitle("Prophet components during the held-out forecast week", fontsize=13)
    _save_figure(figure, output_path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_metrics_files(
    metrics: list[dict[str, Any]],
    winner_name: str,
    probabilistic_metrics: dict[str, float],
    total_runtime_seconds: float,
    json_path: Path,
    csv_path: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "test_start": "2020-01-01",
        "test_end": "2020-01-07",
        "n_test_observations": 168,
        "models": metrics,
        "winner": winner_name,
        **probabilistic_metrics,
        "total_runtime_seconds": total_runtime_seconds,
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    rows: list[dict[str, Any]] = []
    global_columns = {
        "test_start": payload["test_start"],
        "test_end": payload["test_end"],
        "n_test_observations": payload["n_test_observations"],
        "winner": winner_name,
        **probabilistic_metrics,
        "total_runtime_seconds": total_runtime_seconds,
    }
    for row in metrics:
        flat = {**global_columns, **row}
        flat["is_winner"] = row["name"] == winner_name
        flat["hyperparameters"] = json.dumps(_json_safe(row["hyperparameters"]), sort_keys=True)
        rows.append(flat)
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6f")
    return payload


def get_hardware_description() -> str:
    try:
        return subprocess.run(
            ["uname", "-a"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.platform()


def _format_results_table(metrics: list[dict[str, Any]]) -> str:
    ordered = sorted(metrics, key=lambda row: row["mape_test_pct"])
    lines = [
        "| Rank | Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2-7 MAPE (%) | Runtime (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ordered, start=1):
        lines.append(
            f"| {rank} | {MODEL_LABELS[row['name']]} | {row['mape_test_pct']:.3f} | "
            f"{row['rmse_test_mw']:.1f} | {row['mae_test_mw']:.1f} | "
            f"{row['mape_jan1_pct']:.3f} | {row['mape_jan2_to_jan7_pct']:.3f} | "
            f"{row['runtime_seconds']:.1f} |"
        )
    return "\n".join(lines)


def _ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def _discussion(
    metrics: list[dict[str, Any]],
    winner_name: str,
    probabilistic_metrics: dict[str, float],
    daily: pd.DataFrame,
) -> str:
    ordered = sorted(metrics, key=lambda row: row["mape_test_pct"])
    rank = {row["name"]: index for index, row in enumerate(ordered, start=1)}
    by_name = {row["name"]: row for row in metrics}
    winner = by_name[winner_name]
    second = ordered[1]
    rank_text = ", ".join(MODEL_LABELS[row["name"]] for row in ordered)
    winner_worst_day = daily.loc[winner_name].idxmax()
    lightgbm_worst_day = daily.loc["lightgbm"].idxmax()
    nbeats_worst_day = daily.loc["nbeats"].idxmax()
    tsmixer_worst_day = daily.loc["patchtst"].idxmax()

    sentences = [
        f"{MODEL_LABELS[winner_name]} won on the preregistered primary metric with a test MAPE of "
        f"{winner['mape_test_pct']:.3f}%, ahead of {MODEL_LABELS[second['name']]} at "
        f"{second['mape_test_pct']:.3f}%. Its largest daily error was "
        f"{daily.loc[winner_name, winner_worst_day]:.1f}% on {winner_worst_day.strftime('%b %d')}, "
        "so the weekly average does not describe uniformly good performance.",
        f"The complete rank order was {rank_text}.",
        f"The seasonal-naive baseline ranked {_ordinal(rank['naive'])}. Its one-week copy failed when "
        f"Christmas-week levels were carried into the first working days: MAPE rose from "
        f"{by_name['naive']['mape_jan1_pct']:.3f}% on Jan 1 to "
        f"{by_name['naive']['mape_jan2_to_jan7_pct']:.3f}% thereafter.",
        f"SARIMA ranked {_ordinal(rank['sarima'])}. Weekly differencing improved only slightly on naive, "
        f"and its Jan 2-7 MAPE remained {by_name['sarima']['mape_jan2_to_jan7_pct']:.3f}% because the "
        "linear model had no explicit holiday or working-day regime.",
        f"Prophet ranked {_ordinal(rank['prophet'])}, but its explicit German-holiday term did not solve "
        f"the holiday itself: Jan 1 MAPE was {by_name['prophet']['mape_jan1_pct']:.3f}%, about twice its "
        f"{by_name['prophet']['mape_jan2_to_jan7_pct']:.3f}% error on Jan 2-7. A smooth additive holiday "
        "effect was not flexible enough for the observed hourly shape.",
        f"LightGBM ranked {_ordinal(rank['lightgbm'])}. Calendar, holiday, lag, and rolling features "
        f"supported its nonlinear forecast, but its daily MAPE reached "
        f"{daily.loc['lightgbm', lightgbm_worst_day]:.1f}% on "
        f"{lightgbm_worst_day.strftime('%b %d')} as recursive errors accumulated.",
        f"The winning {MODEL_LABELS[winner_name]} uncertainty estimates were also imperfect: the nominal "
        f"80% and 95% intervals covered {probabilistic_metrics['winner_coverage_80pct']:.1%} and "
        f"{probabilistic_metrics['winner_coverage_95pct']:.1%} of observations.",
        f"N-BEATS and the TSMixer substitute ranked {_ordinal(rank['nbeats'])} and "
        f"{_ordinal(rank['patchtst'])}. Their worst daily MAPEs were "
        f"{daily.loc['nbeats', nbeats_worst_day]:.1f}% and "
        f"{daily.loc['patchtst', tsmixer_worst_day]:.1f}% on "
        f"{nbeats_worst_day.strftime('%b %d')} and {tsmixer_worst_day.strftime('%b %d')}. Both received "
        "only past load, so they had to infer future holiday regimes indirectly.",
        "The ordering is only partly predicted by model complexity: flexible models help when their inputs "
        "identify the coming regime. This one winter holiday week is not an annual skill estimate. Weather "
        "was deliberately excluded, deep training remains stochastic despite fixed seeds, and no intervals "
        "were recalibrated over multiple forecast origins.",
    ]
    return " ".join(sentences)


def write_transcript(
    path: Path,
    metrics: list[dict[str, Any]],
    winner_name: str,
    probabilistic_metrics: dict[str, float],
    total_runtime_seconds: float,
    hardware: str,
    daily: pd.DataFrame,
) -> None:
    by_name = {row["name"]: row for row in metrics}
    recommendation_name = winner_name
    discussion = _discussion(metrics, winner_name, probabilistic_metrics, daily)
    text = f"""# German hourly load forecasting bake-off

## Data

The study uses the German national load series published by Open Power System Data and derived from the ENTSO-E Transparency Platform. The file contains 50,400 consecutive hourly UTC observations from 2015-01-01 00:00 through 2020-09-30 23:00, measured in megawatts. The timestamp grid was checked for duplicates and one-hour spacing, and the load column was checked for missing values. No gaps or missing values were found, so no imputation or row removal was needed.

## Why these six models

The six models form a deliberate complexity gradient. Seasonal naive copies the same hour from one week earlier and supplies the reference skill level. SARIMA removes last week's level, then uses a linear probability model to extend recurring daily changes. Prophet adds separate curves for long-run movement, daily, weekly, yearly, and German-holiday effects. LightGBM uses decision trees to combine calendar fields, federal holidays, past observations, and strictly past rolling summaries. N-BEATS uses stacked neural-network blocks to map the previous 168 hours into the next 168. The requested PatchTST slot uses Darts TSMixerModel because Darts 0.41.0 does not expose PatchTSTModel in this environment. TSMixer mixes information across the hours in the input window and keeps the same 168-hour input and output lengths, but it is not a literal PatchTST implementation.

## Validation strategy

The initial training window contains 41,616 hours from 2015-01-01 through 2019-09-30. The validation window contains 2,208 hours from 2019-10-01 through 2019-12-31, and the untouched test window contains 168 hours from 2020-01-01 through 2020-01-07. SARIMA order, Prophet prior scale, LightGBM tree settings, and the deep models' stopping epoch were selected only from validation MAPE. For each deep model, independent fits at 6, 12, 18, 24, and at most 30 epochs were compared using the median forecast from one fixed origin; the search stopped after two successive candidates failed to improve MAPE. Every selected configuration was then fit again on all 43,824 pre-test hours. Test observations were passed only to the common scoring functions after every forecast had been produced.

Mean absolute percentage error (MAPE) is the average absolute miss as a percentage of observed load. Mean absolute error (MAE) is the average miss in megawatts, while root mean squared error (RMSE) gives extra weight to large misses. A q=0.1 forecast is a level that observations should exceed about 90% of the time. Pinball loss scores this quantile accuracy, with lower values better.

## Results table

{_format_results_table(metrics)}

The winning {MODEL_LABELS[winner_name]} intervals achieved {probabilistic_metrics["winner_coverage_80pct"]:.1%} coverage at the nominal 80% level and {probabilistic_metrics["winner_coverage_95pct"]:.1%} coverage at the nominal 95% level. Pinball losses were {probabilistic_metrics["winner_pinball_loss_q10"]:.1f} MW at q=0.1, {probabilistic_metrics["winner_pinball_loss_q50"]:.1f} MW at q=0.5, and {probabilistic_metrics["winner_pinball_loss_q90"]:.1f} MW at q=0.9.

## Discussion

{discussion}

## Recommendation

If exactly one of these implementations had to enter a JRC short-term load forecasting pipeline, I would choose {MODEL_LABELS[recommendation_name]}. It had the lowest held-out MAPE in the common test and its error profile is directly visible in the daily and residual figures. Before production use, I would run rolling-origin backtests across at least one full year, recalibrate its prediction intervals on those origins, test forecast latency and failure recovery, and add weather only in a separate study so its incremental value is measured rather than assumed. The seasonal-naive forecast should remain deployed beside it as a live fallback and drift alarm, not as an unreported comparator.

## Reproducibility note

All stochastic code used seed {SEED}. The complete run took {total_runtime_seconds:.1f} seconds ({total_runtime_seconds / 60.0:.1f} minutes) on `{hardware}`. Reproduce from this directory with `../../.venv/bin/python code/forecast.py`; no package installation or external data download is required. Calendar and holiday indicators use the UTC date carried by the input, matching the evaluation labels rather than converting to German civil time. LightGBM point forecasts append their own prediction before computing later lag and rolling features. Its uncertainty bands add empirical validation residual quantiles for the matching UTC hour to that recursive point path, so they carry the error distribution observed across the fixed-origin validation forecast without using test observations. No observed value from a forecast window enters a later feature row. N-BEATS and TSMixer training windows used a stride of {by_name["nbeats"]["hyperparameters"].get("training_stride", "unknown")} hours to keep the complete six-model run within the thirty-minute reproducibility target.
"""
    path.write_text(text, encoding="utf-8")
