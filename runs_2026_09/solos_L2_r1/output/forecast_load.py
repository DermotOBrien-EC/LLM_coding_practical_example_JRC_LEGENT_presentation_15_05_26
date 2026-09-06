from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.axes import Axes
from matplotlib.ticker import StrMethodFormatter
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TIMESTAMP_COLUMN = "utc_timestamp"
LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
FORECAST_START = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
HORIZON_HOURS = 168
LAGS_HOURS = (168, 336, 504, 672, 8760)
MAX_FORECAST_HOURS = min(LAGS_HOURS)
RANDOM_SEED = 42
MODEL_NAME = "ExtraTreesRegressor"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
ACTUAL_COLOR = "#2a78d6"
FORECAST_COLOR = "#eb6834"


@dataclass(frozen=True)
class ForecastMetrics:
    forecast_start_utc: str
    forecast_end_utc: str
    forecast_hours: int
    training_start_utc: str
    training_end_utc: str
    training_rows: int
    model: str
    python_version: str
    numpy_version: str
    pandas_version: str
    matplotlib_version: str
    scikit_learn_version: str
    mae_mw: float
    rmse_mw: float
    mape_percent: float
    r_squared: float
    mean_error_mw: float
    max_absolute_error_mw: float
    weekly_naive_mae_mw: float
    weekly_naive_mape_percent: float
    mae_improvement_over_weekly_naive_percent: float
    annual_naive_mae_mw: float
    annual_naive_mape_percent: float
    mae_improvement_over_annual_naive_percent: float
    daily_mae_mw: dict[str, float]


def parse_utc_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def forecast_index(start: pd.Timestamp, hours: int) -> pd.DatetimeIndex:
    if hours <= 0:
        raise ValueError("Forecast horizon must be positive.")
    if hours > MAX_FORECAST_HOURS:
        raise ValueError(
            f"Forecast horizon cannot exceed {MAX_FORECAST_HOURS} hours because "
            "longer horizons would use observed target-period loads as lag features."
        )
    normalized_start = parse_utc_timestamp(str(start))
    if normalized_start != normalized_start.floor("h"):
        raise ValueError("Forecast start must be aligned to an exact UTC hour.")
    return pd.date_range(normalized_start, periods=hours, freq="h", tz="UTC")


def load_hourly_series(csv_path: Path) -> pd.Series:
    frame = pd.read_csv(csv_path)
    required_columns = {TIMESTAMP_COLUMN, LOAD_COLUMN}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required CSV columns: {names}")

    timestamps = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True, errors="raise")
    values = pd.to_numeric(frame[LOAD_COLUMN], errors="raise")
    series = pd.Series(values.to_numpy(dtype=float), index=timestamps, name="actual_mw")
    series = series.sort_index()
    series_index = pd.DatetimeIndex(series.index)

    if series_index.has_duplicates:
        duplicate_count = int(series_index.duplicated().sum())
        raise ValueError(f"Found {duplicate_count} duplicate timestamps.")
    if series.isna().any():
        raise ValueError(f"Found {int(series.isna().sum())} missing load values.")
    if (series <= 0.0).any():
        raise ValueError("Load values must be positive.")

    expected_index = pd.date_range(series_index[0], series_index[-1], freq="h", tz="UTC")
    if not series_index.equals(expected_index):
        missing_count = len(expected_index.difference(series_index))
        raise ValueError(
            f"The input is not a complete hourly series; {missing_count} hours are missing."
        )

    return series


def build_features(load: pd.Series) -> pd.DataFrame:
    index = pd.DatetimeIndex(load.index)
    features = pd.DataFrame(index=index)

    for lag_hours in LAGS_HOURS:
        features[f"lag_{lag_hours}"] = load.shift(lag_hours)

    weekly_columns = [f"lag_{lag_hours}" for lag_hours in (168, 336, 504, 672)]
    weekly_lags = features[weekly_columns]
    features["weekly_mean_4"] = weekly_lags.mean(axis=1)
    features["weekly_median_4"] = weekly_lags.median(axis=1)
    features["weekly_trend_1_4"] = features["lag_168"] - features["lag_672"]

    hour = index.hour.to_numpy(dtype=float)
    day_of_week = index.dayofweek.to_numpy(dtype=float)
    day_of_year = index.dayofyear.to_numpy(dtype=float)
    features["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    features["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    features["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    features["day_of_year_sin"] = np.sin(2.0 * np.pi * (day_of_year - 1.0) / 365.2425)
    features["day_of_year_cos"] = np.cos(2.0 * np.pi * (day_of_year - 1.0) / 365.2425)
    features["is_weekend"] = (day_of_week >= 5.0).astype(int)
    features["year_trend"] = index.year.to_numpy() - int(index.year.min())
    features["is_new_year"] = ((index.month.to_numpy() == 1) & (index.day.to_numpy() == 1)).astype(
        int
    )
    features["days_from_new_year"] = np.minimum(np.abs(day_of_year - 1.0), 10.0)

    return features


def fit_and_forecast(
    load: pd.Series,
    start: pd.Timestamp,
    hours: int,
    n_estimators: int,
) -> tuple[pd.DataFrame, int, pd.Timestamp, pd.Timestamp]:
    target_index = forecast_index(start, hours)
    load_index = pd.DatetimeIndex(load.index)
    missing_target_hours = target_index.difference(load_index)
    if len(missing_target_hours) > 0:
        raise ValueError(f"The data does not contain {len(missing_target_hours)} target hours.")

    features = build_features(load)
    train_mask = (features.index < target_index[0]) & features.notna().all(axis=1)
    if not bool(train_mask.any()):
        raise ValueError("No complete historical feature rows are available for training.")

    target_features = features.loc[target_index]
    if target_features.isna().any().any():
        raise ValueError("Target features contain missing lag values.")

    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_features=0.4,
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    training_features = features.loc[train_mask]
    training_target = load.loc[train_mask]
    model.fit(training_features, training_target)
    forecast = model.predict(target_features)

    results = pd.DataFrame(
        {
            "actual_mw": load.loc[target_index].to_numpy(dtype=float),
            "forecast_mw": forecast,
            "weekly_naive_mw": target_features["lag_168"].to_numpy(dtype=float),
            "annual_naive_mw": target_features["lag_8760"].to_numpy(dtype=float),
        },
        index=target_index,
    )
    results.index.name = TIMESTAMP_COLUMN
    results["error_mw"] = results["forecast_mw"] - results["actual_mw"]
    results["absolute_error_mw"] = results["error_mw"].abs()
    results["absolute_percentage_error_percent"] = (
        results["absolute_error_mw"] / results["actual_mw"] * 100.0
    )

    return (
        results,
        len(training_features),
        pd.Timestamp(training_features.index[0]),
        pd.Timestamp(training_features.index[-1]),
    )


def calculate_metrics(
    results: pd.DataFrame,
    training_rows: int,
    training_start: pd.Timestamp,
    training_end: pd.Timestamp,
) -> ForecastMetrics:
    actual = results["actual_mw"]
    forecast = results["forecast_mw"]
    weekly_naive = results["weekly_naive_mw"]
    annual_naive = results["annual_naive_mw"]

    mae = float(mean_absolute_error(actual, forecast))
    weekly_naive_mae = float(mean_absolute_error(actual, weekly_naive))
    annual_naive_mae = float(mean_absolute_error(actual, annual_naive))
    date_labels = pd.DatetimeIndex(results.index).date
    daily_mae = results["absolute_error_mw"].groupby(date_labels).mean()
    daily_mae_mw = {str(date): round(float(value), 3) for date, value in daily_mae.items()}

    return ForecastMetrics(
        forecast_start_utc=results.index[0].isoformat(),
        forecast_end_utc=results.index[-1].isoformat(),
        forecast_hours=len(results),
        training_start_utc=training_start.isoformat(),
        training_end_utc=training_end.isoformat(),
        training_rows=training_rows,
        model=MODEL_NAME,
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        pandas_version=pd.__version__,
        matplotlib_version=matplotlib.__version__,
        scikit_learn_version=sklearn.__version__,
        mae_mw=mae,
        rmse_mw=float(np.sqrt(mean_squared_error(actual, forecast))),
        mape_percent=float(
            np.mean(np.abs((actual.to_numpy() - forecast.to_numpy()) / actual.to_numpy())) * 100.0
        ),
        r_squared=float(r2_score(actual, forecast)),
        mean_error_mw=float((forecast - actual).mean()),
        max_absolute_error_mw=float(results["absolute_error_mw"].max()),
        weekly_naive_mae_mw=weekly_naive_mae,
        weekly_naive_mape_percent=float(
            np.mean(np.abs((actual.to_numpy() - weekly_naive.to_numpy()) / actual.to_numpy()))
            * 100.0
        ),
        mae_improvement_over_weekly_naive_percent=float(
            (weekly_naive_mae - mae) / weekly_naive_mae * 100.0
        ),
        annual_naive_mae_mw=annual_naive_mae,
        annual_naive_mape_percent=float(
            np.mean(np.abs((actual.to_numpy() - annual_naive.to_numpy()) / actual.to_numpy()))
            * 100.0
        ),
        mae_improvement_over_annual_naive_percent=float(
            (annual_naive_mae - mae) / annual_naive_mae * 100.0
        ),
        daily_mae_mw=daily_mae_mw,
    )


def add_endpoint_label(
    axis: Axes,
    timestamp: pd.Timestamp,
    value: float,
    label: str,
    line_color: str,
    vertical_offset: float,
) -> None:
    axis.annotate(
        f"{label} {value / 1000.0:.1f} GW",
        xy=(timestamp, value),
        xytext=(-10, vertical_offset),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8.5,
        color=TEXT_SECONDARY,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": SURFACE, "edgecolor": "none"},
        arrowprops={"arrowstyle": "-", "color": line_color, "linewidth": 1.2},
    )


def format_forecast_period(index: pd.Index) -> str:
    timestamps = pd.DatetimeIndex(index)
    if timestamps.empty:
        raise ValueError("Cannot format an empty forecast period.")

    start = pd.Timestamp(timestamps[0])
    end = pd.Timestamp(timestamps[-1])
    if start.date() == end.date():
        period = f"{start.day} {start.strftime('%B %Y')}"
    elif start.year == end.year and start.month == end.month:
        period = f"{start.day}–{end.day} {start.strftime('%B %Y')}"
    elif start.year == end.year:
        period = f"{start.day} {start.strftime('%B')}–{end.day} {end.strftime('%B %Y')}"
    else:
        period = f"{start.day} {start.strftime('%B %Y')}–{end.day} {end.strftime('%B %Y')}"
    return f"{period} (UTC)"


def plot_forecast(
    results: pd.DataFrame,
    metrics: ForecastMetrics,
    output_path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": TEXT_SECONDARY,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "text.color": TEXT_PRIMARY,
        }
    )
    figure, axis = plt.subplots(figsize=(13.0, 6.5), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)

    axis.plot(
        results.index,
        results["actual_mw"],
        color=ACTUAL_COLOR,
        linewidth=2.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        label="Actual load",
        zorder=3,
    )
    axis.plot(
        results.index,
        results["forecast_mw"],
        color=FORECAST_COLOR,
        linewidth=2.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        label="Forecast",
        zorder=2,
    )

    axis.set_title(
        "German hourly electricity load: forecast vs actual",
        loc="left",
        fontsize=17,
        color=TEXT_PRIMARY,
        pad=32,
    )
    axis.text(
        0.0,
        1.025,
        (
            f"{format_forecast_period(results.index)}  |  MAE {metrics.mae_mw:,.0f} MW  |  "
            f"MAPE {metrics.mape_percent:.2f}%  |  R² {metrics.r_squared:.3f}"
        ),
        transform=axis.transAxes,
        color=TEXT_SECONDARY,
        fontsize=10,
        va="bottom",
    )
    axis.set_ylabel("Load (MW)")
    axis.set_xlabel("Hour (UTC)", labelpad=10)
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    day_locator = mdates.DayLocator(interval=1, tz=mdates.UTC)  # type: ignore[no-untyped-call]
    date_formatter = mdates.DateFormatter(  # type: ignore[no-untyped-call]
        "%a\n%d %b", tz=mdates.UTC
    )
    axis.xaxis.set_major_locator(day_locator)
    axis.xaxis.set_major_formatter(date_formatter)
    axis.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0.0, 1.0))

    for spine_name in ("top", "right", "left"):
        axis.spines[spine_name].set_visible(False)
    axis.spines["bottom"].set_color(BASELINE)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(axis="both", length=0)
    axis.margins(y=0.12)
    axis.set_xlim(results.index[0], results.index[-1])

    actual_end = float(results["actual_mw"].iloc[-1])
    forecast_end = float(results["forecast_mw"].iloc[-1])
    close_endpoints = abs(actual_end - forecast_end) < 3_000.0
    if close_endpoints:
        actual_offset = 15.0 if actual_end >= forecast_end else -17.0
        forecast_offset = -17.0 if actual_end >= forecast_end else 15.0
    else:
        actual_offset = 0.0
        forecast_offset = 0.0
    add_endpoint_label(
        axis,
        pd.Timestamp(results.index[-1]),
        actual_end,
        "Actual",
        ACTUAL_COLOR,
        actual_offset,
    )
    add_endpoint_label(
        axis,
        pd.Timestamp(results.index[-1]),
        forecast_end,
        "Forecast",
        FORECAST_COLOR,
        forecast_offset,
    )

    figure.text(
        0.99,
        0.012,
        "Source: Open Power System Data, German ENTSO-E load",
        ha="right",
        va="bottom",
        fontsize=8,
        color=TEXT_MUTED,
    )
    figure.tight_layout(rect=(0.02, 0.035, 0.99, 0.98))
    figure.savefig(output_path, dpi=180, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def write_outputs(
    results: pd.DataFrame,
    metrics: ForecastMetrics,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = output_dir / "forecast_results.csv"
    metrics_path = output_dir / "forecast_metrics.json"
    plot_path = output_dir / "forecast_vs_actual.png"

    results.reset_index().to_csv(forecast_path, index=False, float_format="%.3f")
    metrics_path.write_text(
        json.dumps(asdict(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_forecast(results, metrics, plot_path)
    return forecast_path, metrics_path, plot_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Forecast German hourly electricity load for a seven-day interval and "
            "compare the forecast with actual values."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("opsd_de_load.csv"),
        help="Input CSV path (default: opsd_de_load.csv).",
    )
    parser.add_argument(
        "--start",
        type=parse_utc_timestamp,
        default=FORECAST_START,
        help="Forecast start timestamp in UTC (default: 2020-01-01 00:00:00).",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=HORIZON_HOURS,
        help="Forecast horizon from 1 to 168 hours (default: 168).",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=500,
        help="Number of Extra Trees estimators (default: 500).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV, JSON, and PNG outputs (default: outputs).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.estimators <= 0:
        raise ValueError("Estimator count must be positive.")

    load = load_hourly_series(arguments.data)
    results, training_rows, training_start, training_end = fit_and_forecast(
        load=load,
        start=arguments.start,
        hours=arguments.hours,
        n_estimators=arguments.estimators,
    )
    metrics = calculate_metrics(results, training_rows, training_start, training_end)
    forecast_path, metrics_path, plot_path = write_outputs(results, metrics, arguments.output_dir)

    print(f"Forecast interval: {metrics.forecast_start_utc} to {metrics.forecast_end_utc}")
    print(f"Training rows: {metrics.training_rows:,}")
    print(f"MAE: {metrics.mae_mw:,.1f} MW")
    print(f"RMSE: {metrics.rmse_mw:,.1f} MW")
    print(f"MAPE: {metrics.mape_percent:.3f}%")
    print(f"R-squared: {metrics.r_squared:.4f}")
    print(f"Mean error: {metrics.mean_error_mw:+,.1f} MW (positive means the model overforecast).")
    print(f"Same-hour previous-year baseline MAE: {metrics.annual_naive_mae_mw:,.1f} MW")
    print(
        "MAE improvement over the previous-year baseline: "
        f"{metrics.mae_improvement_over_annual_naive_percent:.2f}%"
    )
    print(f"Forecast data: {forecast_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
