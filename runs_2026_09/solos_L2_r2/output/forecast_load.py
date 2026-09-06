from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import holidays
import lightgbm as lgb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from matplotlib.axes import Axes
from matplotlib.figure import Figure

TIMESTAMP_COLUMN = "utc_timestamp"
LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
TARGET_START = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
TARGET_HOURS = 7 * 24
TARGET_END = TARGET_START + pd.Timedelta(hours=TARGET_HOURS)
TRAINING_YEARS = 2
BACKTEST_YEARS = (2017, 2018, 2019)
RANDOM_SEED = 42
REFERENCE_TIME = pd.Timestamp("2015-01-01 00:00:00", tz="UTC")

ACTUAL_COLOR = "#2a78d6"
FORECAST_COLOR = "#eb6834"
SURFACE_COLOR = "#fcfcfb"
TEXT_COLOR = "#0b0b0b"
SECONDARY_TEXT_COLOR = "#52514e"
MUTED_COLOR = "#898781"
GRID_COLOR = "#e1e0d9"


@dataclass(frozen=True)
class ForecastMetrics:
    mae_mw: float
    rmse_mw: float
    mape_percent: float
    smape_percent: float
    r_squared: float


@dataclass(frozen=True)
class ModelRun:
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    forecast: pd.Series


@dataclass(frozen=True)
class BacktestReport:
    year: int
    training_start_utc: str
    training_end_utc: str
    metrics: ForecastMetrics


@dataclass(frozen=True)
class ForecastPeriodReport:
    start: str
    end_exclusive: str
    hours: int


@dataclass(frozen=True)
class ModelReport:
    type: str
    features: str
    training_window_years: int
    random_seed: int


@dataclass(frozen=True)
class TrainingPeriodReport:
    start: str
    end: str


@dataclass(frozen=True)
class ArtifactReport:
    forecast_csv: str
    forecast_csv_sha256: str
    plot_png: str
    plot_png_sha256: str


@dataclass(frozen=True)
class ForecastReport:
    forecast_period_utc: ForecastPeriodReport
    model: ModelReport
    final_training_period_utc: TrainingPeriodReport
    target_metrics: ForecastMetrics
    historical_new_year_backtests: tuple[BacktestReport, ...]
    mean_backtest_metrics: ForecastMetrics | None
    artifacts: ArtifactReport


def make_forecast_index(start: pd.Timestamp) -> pd.DatetimeIndex:
    if start.tz is None:
        raise ValueError("The forecast start must be timezone-aware.")
    return pd.date_range(start.tz_convert("UTC"), periods=TARGET_HOURS, freq="h")


def make_target_index() -> pd.DatetimeIndex:
    return make_forecast_index(TARGET_START)


def validate_hourly_index(index: pd.DatetimeIndex, label: str) -> None:
    if index.empty:
        raise ValueError(f"The {label} index is empty.")
    if index.tz is None:
        raise ValueError(f"The {label} timestamps must be timezone-aware.")
    if index.has_duplicates:
        raise ValueError(f"The {label} timestamps contain duplicates.")
    if not index.is_monotonic_increasing:
        raise ValueError(f"The {label} timestamps are not sorted.")

    expected = pd.date_range(index[0], periods=len(index), freq="h")
    if not index.equals(expected):
        raise ValueError(f"The {label} timestamps are not continuous at hourly frequency.")


def load_hourly_load(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    required_columns = {TIMESTAMP_COLUMN, LOAD_COLUMN}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    timestamps = pd.DatetimeIndex(pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True, errors="raise"))
    load = pd.Series(
        pd.to_numeric(frame[LOAD_COLUMN], errors="raise").to_numpy(dtype=float),
        index=timestamps,
        name="load_mw",
    )
    relevant_load = load.loc[load.index < TARGET_END]
    validate_hourly_load(relevant_load)
    return relevant_load


def validate_hourly_load(load: pd.Series) -> None:
    if load.empty:
        raise ValueError("The load series is empty.")
    if load.index.tz is None:
        raise ValueError("The load timestamps must be timezone-aware.")
    if load.index.has_duplicates:
        raise ValueError("The load timestamps contain duplicates.")
    if not load.index.is_monotonic_increasing:
        raise ValueError("The load timestamps are not sorted.")
    if load.isna().any():
        raise ValueError("The load series contains missing values.")
    if not np.isfinite(load.to_numpy(dtype=float)).all():
        raise ValueError("The load series contains non-finite values.")


def split_training_and_target(load: pd.Series) -> tuple[pd.Series, pd.Series]:
    target_index = make_target_index()
    missing_target_hours = target_index.difference(load.index)
    if not missing_target_hours.empty:
        raise ValueError(f"The dataset is missing {len(missing_target_hours)} target hours.")

    training = load.loc[load.index < TARGET_START]
    actual = load.loc[target_index]
    if training.empty:
        raise ValueError("No observations are available before the forecast period.")
    return training, actual


def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    if index.empty:
        raise ValueError("The feature index is empty.")
    if index.tz is None:
        raise ValueError("The feature timestamps must be timezone-aware.")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("The feature timestamps must be unique and sorted.")

    utc_index = index.tz_convert("UTC")
    local_index = utc_index.tz_convert("Europe/Berlin")
    local_dates = pd.Index(local_index.date)
    holiday_calendar = holidays.country_holidays(
        "DE", years=range(int(local_index.year.min()) - 1, int(local_index.year.max()) + 2)
    )

    features = pd.DataFrame(index=utc_index)
    features["elapsed_days"] = (utc_index - REFERENCE_TIME).total_seconds() / 86_400.0
    features["local_hour"] = local_index.hour
    features["weekday"] = local_index.dayofweek
    features["day_of_year"] = local_index.dayofyear
    features["month"] = local_index.month
    features["day_of_month"] = local_index.day

    features["hour_sin"] = np.sin(2.0 * np.pi * local_index.hour / 24.0)
    features["hour_cos"] = np.cos(2.0 * np.pi * local_index.hour / 24.0)
    features["weekday_sin"] = np.sin(2.0 * np.pi * local_index.dayofweek / 7.0)
    features["weekday_cos"] = np.cos(2.0 * np.pi * local_index.dayofweek / 7.0)
    features["year_sin"] = np.sin(2.0 * np.pi * local_index.dayofyear / 365.25)
    features["year_cos"] = np.cos(2.0 * np.pi * local_index.dayofyear / 365.25)

    features["is_holiday"] = local_dates.isin(holiday_calendar).astype(int)
    features["is_new_year"] = ((local_index.month == 1) & (local_index.day == 1)).astype(int)
    features["is_christmas_eve"] = ((local_index.month == 12) & (local_index.day == 24)).astype(int)
    features["is_christmas"] = ((local_index.month == 12) & local_index.day.isin([25, 26])).astype(
        int
    )
    features["is_new_year_eve"] = ((local_index.month == 12) & (local_index.day == 31)).astype(int)
    return features


def build_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=400,
        learning_rate=0.04,
        num_leaves=16,
        max_depth=6,
        min_child_samples=50,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbosity=-1,
    )


def required_training_index(forecast_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    validate_hourly_index(forecast_index, "forecast")
    training_start = forecast_index[0] - pd.DateOffset(years=TRAINING_YEARS)
    return pd.date_range(training_start, forecast_index[0], freq="h", inclusive="left")


def has_complete_forecast_data(load: pd.Series, forecast_index: pd.DatetimeIndex) -> bool:
    try:
        training_index = required_training_index(forecast_index)
    except ValueError:
        return False
    required_index = training_index.append(forecast_index)
    return bool(required_index.difference(load.index).empty)


def fit_and_forecast(load: pd.Series, forecast_index: pd.DatetimeIndex) -> ModelRun:
    training_index = required_training_index(forecast_index)
    missing_training_hours = training_index.difference(load.index)
    if not missing_training_hours.empty:
        raise ValueError(
            f"The dataset is missing {len(missing_training_hours)} required training hours."
        )

    training = load.loc[training_index]
    if training.isna().any() or not np.isfinite(training.to_numpy(dtype=float)).all():
        raise ValueError("The training series contains missing or non-finite values.")

    training_features = build_calendar_features(training.index)
    forecast_features = build_calendar_features(forecast_index)
    model = build_model()
    model.fit(training_features, training.to_numpy(dtype=float))
    forecast_values = model.predict(forecast_features)

    forecast = pd.Series(forecast_values, index=forecast_index, name="forecast_load_mw")
    return ModelRun(
        training_start=training.index[0],
        training_end=training.index[-1],
        forecast=forecast,
    )


def validate_actual_and_forecast(actual: pd.Series, forecast: pd.Series) -> None:
    if not actual.index.equals(forecast.index):
        raise ValueError("Actual and forecast series must have identical timestamp indexes.")
    if actual.empty:
        raise ValueError("Metrics require at least one observation.")

    actual_values = actual.to_numpy(dtype=float)
    forecast_values = forecast.to_numpy(dtype=float)
    if not np.isfinite(actual_values).all() or not np.isfinite(forecast_values).all():
        raise ValueError("Actual and forecast series must contain finite values.")
    if np.any(actual_values == 0.0):
        raise ValueError("MAPE is undefined when an actual value is zero.")


def calculate_metrics(actual: pd.Series, forecast: pd.Series) -> ForecastMetrics:
    validate_actual_and_forecast(actual, forecast)
    actual_values = actual.to_numpy(dtype=float)
    forecast_values = forecast.to_numpy(dtype=float)
    errors = forecast_values - actual_values
    absolute_errors = np.abs(errors)
    squared_errors = errors**2
    denominator = np.abs(actual_values) + np.abs(forecast_values)
    squared_error_sum = float(np.sum(squared_errors))
    total_variation = float(np.sum((actual_values - actual_values.mean()) ** 2))
    if total_variation == 0.0:
        r_squared = 1.0 if squared_error_sum == 0.0 else 0.0
    else:
        r_squared = 1.0 - squared_error_sum / total_variation

    return ForecastMetrics(
        mae_mw=float(absolute_errors.mean()),
        rmse_mw=float(np.sqrt(squared_errors.mean())),
        mape_percent=float(np.mean(absolute_errors / np.abs(actual_values)) * 100.0),
        smape_percent=float(np.mean(2.0 * absolute_errors / denominator) * 100.0),
        r_squared=float(r_squared),
    )


def run_backtests(load: pd.Series) -> list[BacktestReport]:
    reports: list[BacktestReport] = []
    for year in BACKTEST_YEARS:
        forecast_index = make_forecast_index(pd.Timestamp(f"{year}-01-01", tz="UTC"))
        if not has_complete_forecast_data(load, forecast_index):
            continue

        run = fit_and_forecast(load, forecast_index)
        actual = load.loc[forecast_index]
        reports.append(
            BacktestReport(
                year=year,
                training_start_utc=run.training_start.isoformat(),
                training_end_utc=run.training_end.isoformat(),
                metrics=calculate_metrics(actual, run.forecast),
            )
        )
    return reports


def average_backtest_metrics(backtests: list[BacktestReport]) -> ForecastMetrics | None:
    if not backtests:
        return None

    return ForecastMetrics(
        mae_mw=float(np.mean([report.metrics.mae_mw for report in backtests])),
        rmse_mw=float(np.mean([report.metrics.rmse_mw for report in backtests])),
        mape_percent=float(np.mean([report.metrics.mape_percent for report in backtests])),
        smape_percent=float(np.mean([report.metrics.smape_percent for report in backtests])),
        r_squared=float(np.mean([report.metrics.r_squared for report in backtests])),
    )


def build_forecast_table(actual: pd.Series, forecast: pd.Series) -> pd.DataFrame:
    validate_actual_and_forecast(actual, forecast)
    table = pd.DataFrame(
        {
            "actual_load_mw": actual,
            "forecast_load_mw": forecast,
        }
    )
    table["error_mw"] = table["forecast_load_mw"] - table["actual_load_mw"]
    table["absolute_percentage_error"] = (
        table["error_mw"].abs() / table["actual_load_mw"].abs() * 100.0
    )
    table.index.name = TIMESTAMP_COLUMN
    return table


def add_endpoint_label(
    axes: Axes,
    timestamp: pd.Timestamp,
    value: float,
    label: str,
    color: str,
    y_offset: int,
) -> None:
    axes.scatter(
        [timestamp],
        [value],
        s=42,
        color=color,
        edgecolor=SURFACE_COLOR,
        linewidth=2,
        zorder=4,
    )
    axes.annotate(
        label,
        xy=(timestamp, value),
        xytext=(12, y_offset),
        textcoords="offset points",
        color=TEXT_COLOR,
        fontsize=8.5,
        va="center",
        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 1.0},
    )


def format_period(index: pd.DatetimeIndex) -> str:
    return f"{index[0]:%d %b %Y} to {index[-1]:%d %b %Y}, UTC"


def plot_forecast(
    table: pd.DataFrame,
    metrics: ForecastMetrics,
    output_path: Path,
) -> None:
    figure: Figure
    axes: Axes
    figure, axes = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    figure.patch.set_facecolor(SURFACE_COLOR)
    axes.set_facecolor(SURFACE_COLOR)

    axes.plot(
        table.index,
        table["actual_load_mw"],
        color=ACTUAL_COLOR,
        linewidth=2.0,
        solid_capstyle="round",
        label="Actual load",
    )
    axes.plot(
        table.index,
        table["forecast_load_mw"],
        color=FORECAST_COLOR,
        linewidth=2.0,
        solid_capstyle="round",
        label="Forecast",
    )

    axes.set_title(
        "German hourly electricity load: actual vs forecast",
        loc="left",
        color=TEXT_COLOR,
        fontsize=16,
        fontweight="semibold",
        pad=24,
    )
    axes.text(
        0.0,
        1.015,
        (
            f"{format_period(table.index)}  |  "
            f"MAE {metrics.mae_mw:,.0f} MW  |  "
            f"MAPE {metrics.mape_percent:.2f}%  |  "
            f"R² {metrics.r_squared:.3f}"
        ),
        transform=axes.transAxes,
        color=SECONDARY_TEXT_COLOR,
        fontsize=9.5,
        va="bottom",
    )
    axes.set_ylabel("Load (MW)", color=SECONDARY_TEXT_COLOR)
    axes.set_xlabel("Timestamp (UTC)", color=SECONDARY_TEXT_COLOR, labelpad=12)
    axes.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    axes.xaxis.set_major_locator(mdates.DayLocator(tz=mdates.UTC))  # type: ignore[no-untyped-call]
    axes.xaxis.set_major_formatter(
        mdates.DateFormatter("%b %d", tz=mdates.UTC)  # type: ignore[no-untyped-call]
    )
    axes.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    axes.grid(axis="x", visible=False)
    axes.tick_params(colors=MUTED_COLOR, labelsize=9)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color(GRID_COLOR)
    axes.spines["bottom"].set_color(GRID_COLOR)

    legend = axes.legend(
        loc="upper left",
        frameon=False,
        ncols=2,
        bbox_to_anchor=(0.0, 0.99),
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    final_timestamp = table.index[-1]
    final_actual = float(table["actual_load_mw"].iloc[-1])
    final_forecast = float(table["forecast_load_mw"].iloc[-1])
    if final_actual >= final_forecast:
        actual_offset, forecast_offset = 10, -14
    else:
        actual_offset, forecast_offset = -14, 10
    add_endpoint_label(
        axes,
        final_timestamp,
        final_actual,
        f"Actual {final_actual:,.0f}",
        ACTUAL_COLOR,
        actual_offset,
    )
    add_endpoint_label(
        axes,
        final_timestamp,
        final_forecast,
        f"Forecast {final_forecast:,.0f}",
        FORECAST_COLOR,
        forecast_offset,
    )
    axes.set_xlim(table.index[0], table.index[-1] + pd.Timedelta(hours=12))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="png", dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report(
    run: ModelRun,
    target_metrics: ForecastMetrics,
    backtests: list[BacktestReport],
    artifacts: ArtifactReport,
) -> ForecastReport:
    forecast_end = run.forecast.index[-1] + pd.Timedelta(hours=1)
    return ForecastReport(
        forecast_period_utc=ForecastPeriodReport(
            start=run.forecast.index[0].isoformat(),
            end_exclusive=forecast_end.isoformat(),
            hours=len(run.forecast),
        ),
        model=ModelReport(
            type="LightGBM gradient-boosted decision trees",
            features="German local calendar, cyclic time, trend, and federal public holidays",
            training_window_years=TRAINING_YEARS,
            random_seed=RANDOM_SEED,
        ),
        final_training_period_utc=TrainingPeriodReport(
            start=run.training_start.isoformat(),
            end=run.training_end.isoformat(),
        ),
        target_metrics=target_metrics,
        historical_new_year_backtests=tuple(backtests),
        mean_backtest_metrics=average_backtest_metrics(backtests),
        artifacts=artifacts,
    )


def write_metrics_report(path: Path, report: ForecastReport) -> None:
    payload = json.dumps(asdict(report), indent=2, allow_nan=False) + "\n"
    path.write_text(payload, encoding="utf-8")


def publish_outputs(
    output_dir: Path,
    table: pd.DataFrame,
    metrics: ForecastMetrics,
    run: ModelRun,
    backtests: list[BacktestReport],
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "german_load_forecast_2020_week1.csv"
    plot_path = output_dir / "german_load_forecast_2020_week1.png"
    metrics_path = output_dir / "german_load_forecast_2020_week1_metrics.json"
    token = uuid4().hex
    temporary_csv = output_dir / f".{csv_path.name}.{token}.tmp"
    temporary_plot = output_dir / f".{plot_path.name}.{token}.tmp"
    temporary_metrics = output_dir / f".{metrics_path.name}.{token}.tmp"

    temporary_paths = (temporary_csv, temporary_plot, temporary_metrics)
    try:
        table.to_csv(temporary_csv, float_format="%.3f")
        plot_forecast(table, metrics, temporary_plot)
        artifacts = ArtifactReport(
            forecast_csv=csv_path.name,
            forecast_csv_sha256=sha256_file(temporary_csv),
            plot_png=plot_path.name,
            plot_png_sha256=sha256_file(temporary_plot),
        )
        write_metrics_report(
            temporary_metrics,
            build_report(run, metrics, backtests, artifacts),
        )
        temporary_csv.replace(csv_path)
        temporary_plot.replace(plot_path)
        temporary_metrics.replace(metrics_path)
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)

    return csv_path, plot_path, metrics_path


def print_summary(
    run: ModelRun,
    metrics: ForecastMetrics,
    backtests: list[BacktestReport],
    csv_path: Path,
    plot_path: Path,
    metrics_path: Path,
) -> None:
    mean_backtest = average_backtest_metrics(backtests)
    print("German electricity load forecast complete")
    print(f"Forecast window: {format_period(run.forecast.index)}")
    print(f"Training window: {run.training_start.isoformat()} to {run.training_end.isoformat()}")
    print(f"Hours forecast: {len(run.forecast)}")
    print(f"MAE:  {metrics.mae_mw:,.1f} MW")
    print(f"RMSE: {metrics.rmse_mw:,.1f} MW")
    print(f"MAPE: {metrics.mape_percent:.3f}%")
    print(f"sMAPE: {metrics.smape_percent:.3f}%")
    print(f"R²:   {metrics.r_squared:.4f}")
    if mean_backtest is not None:
        years = ", ".join(str(backtest.year) for backtest in backtests)
        print(
            f"Mean New Year backtest ({years}): "
            f"MAE {mean_backtest.mae_mw:,.1f} MW, "
            f"MAPE {mean_backtest.mape_percent:.3f}%"
        )
    print(f"Forecast table: {csv_path}")
    print(f"Plot: {plot_path}")
    print(f"Metrics: {metrics_path}")


def parse_args() -> argparse.Namespace:
    default_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Forecast German hourly electricity load for 1 to 7 January 2020."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=default_directory / "opsd_de_load.csv",
        help="Path to the OPSD German load CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_directory,
        help="Directory for the forecast CSV, plot, and metrics JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load = load_hourly_load(args.data)
    training, actual = split_training_and_target(load)
    backtests = run_backtests(training)
    run = fit_and_forecast(training, actual.index)
    metrics = calculate_metrics(actual, run.forecast)
    table = build_forecast_table(actual, run.forecast)

    csv_path, plot_path, metrics_path = publish_outputs(
        args.output_dir.resolve(), table, metrics, run, backtests
    )
    print_summary(run, metrics, backtests, csv_path, plot_path, metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
