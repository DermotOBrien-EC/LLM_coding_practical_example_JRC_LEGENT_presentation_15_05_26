from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import holidays
import lightgbm
import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from lightgbm import LGBMRegressor
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

TIMESTAMP_COLUMN = "utc_timestamp"
LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
FORECAST_START = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
FORECAST_HOURS = 168
FORECAST_END = FORECAST_START + pd.Timedelta(hours=FORECAST_HOURS)
FORECAST_LAST = FORECAST_END - pd.Timedelta(hours=1)
WEEK_HOURS = 168
WEEKLY_LAGS = tuple(WEEK_HOURS * week for week in range(1, 5))
ROLLING_FEATURE_WINDOW_HOURS = 4 * WEEK_HOURS
FEATURE_WARMUP_HOURS = max(
    max(WEEKLY_LAGS),
    WEEK_HOURS + ROLLING_FEATURE_WINDOW_HOURS,
)
MIN_TRAINING_ROWS = 365 * 24
REQUIRED_HISTORY_HOURS = FEATURE_WARMUP_HOURS + MIN_TRAINING_ROWS - 1
VALIDATION_STARTS = (
    pd.Timestamp("2017-01-01 00:00:00", tz="UTC"),
    pd.Timestamp("2018-01-01 00:00:00", tz="UTC"),
    pd.Timestamp("2019-01-01 00:00:00", tz="UTC"),
)
REQUIRED_FORECAST_STARTS = (*VALIDATION_STARTS, FORECAST_START)
REQUIRED_DATA_START = min(REQUIRED_FORECAST_STARTS) - pd.Timedelta(hours=REQUIRED_HISTORY_HOURS)
REQUIRED_DATA_END = max(REQUIRED_FORECAST_STARTS) + pd.Timedelta(hours=FORECAST_HOURS - 1)

SURFACE = "#fcfcfb"
PRIMARY_TEXT = "#0b0b0b"
SECONDARY_TEXT = "#52514e"
MUTED_TEXT = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACTUAL_COLOR = "#2a78d6"
FORECAST_COLOR = "#eb6834"
MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
WEEKDAY_ABBREVIATIONS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@dataclass(frozen=True)
class ModelParameters:
    n_estimators: int = 400
    learning_rate: float = 0.04
    num_leaves: int = 31
    min_child_samples: int = 60
    reg_lambda: float = 1.0
    random_state: int = 42


@dataclass(frozen=True)
class ForecastMetrics:
    mae_mw: float
    rmse_mw: float
    mape_percent: float
    mean_error_mw: float
    r_squared: float | None


@dataclass(frozen=True)
class ModelForecast:
    predictions: npt.NDArray[np.float64]
    training_index: pd.DatetimeIndex


@dataclass(frozen=True)
class ArtifactDigests:
    forecast_csv_sha256: str
    plot_png_sha256: str


@dataclass(frozen=True)
class ForecastReport:
    model: str
    model_library: str
    model_library_version: str
    model_parameters: ModelParameters
    feature_timezone: str
    target_start_utc: str
    target_end_utc_exclusive: str
    forecast_hours: int
    training_start_utc: str
    training_end_utc: str
    training_hours: int
    historical_validation_mape_percent_by_start_utc: dict[str, float]
    historical_validation_mean_mape_percent: float
    artifact_digests: ArtifactDigests
    accuracy: ForecastMetrics
    weekly_naive_baseline: ForecastMetrics
    mape_improvement_over_weekly_naive_percent: float | None


def format_date_range(start: pd.Timestamp, end_inclusive: pd.Timestamp) -> str:
    start_date = start.date()
    end_date = end_inclusive.date()
    start_month = MONTH_NAMES[start_date.month - 1]
    end_month = MONTH_NAMES[end_date.month - 1]
    if start_date == end_date:
        return f"{start_date.day} {start_month} {start_date.year}"
    if (start_date.year, start_date.month) == (end_date.year, end_date.month):
        return f"{start_date.day}-{end_date.day} {start_month} {start_date.year}"
    if start_date.year == end_date.year:
        return f"{start_date.day} {start_month}-{end_date.day} {end_month} {start_date.year}"
    return (
        f"{start_date.day} {start_month} {start_date.year}-"
        f"{end_date.day} {end_month} {end_date.year}"
    )


FORECAST_DATE_LABEL = format_date_range(FORECAST_START, FORECAST_LAST)
OUTPUT_STEM = (
    f"german_load_forecast_{FORECAST_START.strftime('%Y-%m-%d')}"
    f"_to_{FORECAST_LAST.strftime('%Y-%m-%d')}"
)
MODEL_PARAMETERS = ModelParameters()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Forecast German hourly electricity load for {FORECAST_DATE_LABEL} "
            "from earlier observations in the OPSD load dataset."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("opsd_de_load.csv"),
        help="Path to opsd_de_load.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory for the forecast CSV, metrics JSON, and PNG plot.",
    )
    return parser.parse_args(argv)


def forecast_index(start: pd.Timestamp = FORECAST_START) -> pd.DatetimeIndex:
    if start.tzinfo is None:
        raise ValueError("Forecast start must be timezone-aware")
    start_utc = start.tz_convert("UTC")
    if any(
        (
            start_utc.minute,
            start_utc.second,
            start_utc.microsecond,
            start_utc.nanosecond,
        )
    ):
        raise ValueError("Forecast start must fall exactly on a UTC hour boundary")
    return pd.date_range(start_utc, periods=FORECAST_HOURS, freq="h")


def load_hourly_series(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    missing_columns = {TIMESTAMP_COLUMN, LOAD_COLUMN}.difference(frame.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {names}")
    if frame.empty:
        raise ValueError("Dataset is empty")

    timestamps = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True, errors="raise")
    missing_timestamps = int(timestamps.isna().sum())
    if missing_timestamps:
        raise ValueError(f"Dataset contains {missing_timestamps} missing timestamps")

    relevant_rows = timestamps <= REQUIRED_DATA_END
    if not relevant_rows.any():
        raise ValueError("Dataset contains no observations in the required time range")
    timestamps = timestamps.loc[relevant_rows]
    values = pd.to_numeric(frame.loc[relevant_rows, LOAD_COLUMN], errors="raise")
    load = pd.Series(values.to_numpy(dtype=float), index=timestamps, name=LOAD_COLUMN)
    load = load.sort_index()

    if load.index.has_duplicates:
        duplicates = int(load.index.duplicated().sum())
        raise ValueError(f"Dataset contains {duplicates} duplicate timestamps")
    missing_values = int(load.isna().sum())
    if missing_values:
        raise ValueError(f"Dataset contains {missing_values} missing load values")
    if not np.isfinite(load.to_numpy()).all() or (load <= 0).any():
        raise ValueError("Load values must be finite and strictly positive")

    validate_hourly_index(load.index)
    if load.index[0] > REQUIRED_DATA_START:
        raise ValueError(
            "Dataset does not contain enough history for validation and training; "
            f"it must begin by {REQUIRED_DATA_START.isoformat()}"
        )
    if REQUIRED_DATA_END not in load.index:
        raise ValueError(
            "Dataset does not cover every validation and forecast horizon; "
            f"it must contain {REQUIRED_DATA_END.isoformat()}"
        )
    return load


def validate_hourly_index(index: pd.DatetimeIndex) -> None:
    if index.empty:
        raise ValueError("Hourly index must not be empty")
    if index.tz is None or str(index.tz) != "UTC":
        raise ValueError("Hourly index must use UTC")
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("Hourly index must be unique and sorted")
    if any(
        (
            bool((index.minute != 0).any()),
            bool((index.second != 0).any()),
            bool((index.microsecond != 0).any()),
            bool((index.nanosecond != 0).any()),
        )
    ):
        raise ValueError("Timestamps must fall exactly on UTC hour boundaries")
    if len(index) > 1 and not (index[1:] - index[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("Hourly index must be continuous at one-hour intervals")


def date_membership(
    local_dates: npt.NDArray[np.object_],
    members: set[date],
) -> npt.NDArray[np.float64]:
    return np.asarray(pd.Index(local_dates).isin(members), dtype=np.float64)


def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert("Europe/Berlin")
    local_dates = np.asarray(local.date, dtype=object)
    holiday_years = list(range(int(local.year.min()) - 1, int(local.year.max()) + 2))
    public_holidays = set(holidays.country_holidays("DE", years=holiday_years).keys())
    days_before_holidays = {
        public_holiday - timedelta(days=1) for public_holiday in public_holidays
    }
    days_after_holidays = {public_holiday + timedelta(days=1) for public_holiday in public_holidays}
    reduced_load_days = {
        date(year, month, day) for year in holiday_years for month, day in ((12, 24), (12, 31))
    }

    hour = local.hour.to_numpy(dtype=float)
    day_of_week = local.dayofweek.to_numpy(dtype=float)
    day_of_year = local.dayofyear.to_numpy(dtype=float)
    features = pd.DataFrame(index=index)
    features["hour"] = hour
    features["day_of_week"] = day_of_week
    features["month"] = local.month.to_numpy(dtype=float)
    features["day_of_month"] = local.day.to_numpy(dtype=float)
    features["day_of_year"] = day_of_year
    features["trend_years"] = (index - index[0]).total_seconds().to_numpy() / (
        365.25 * 24 * 60 * 60
    )
    features["is_weekend"] = (day_of_week >= 5).astype(float)
    features["is_public_holiday"] = date_membership(local_dates, public_holidays)
    features["is_day_before_holiday"] = date_membership(
        local_dates,
        days_before_holidays,
    )
    features["is_day_after_holiday"] = date_membership(
        local_dates,
        days_after_holidays,
    )
    features["is_reduced_load_day"] = date_membership(local_dates, reduced_load_days)

    harmonic_groups = (
        ("hour", hour, 24.0, 3),
        ("week", day_of_week, 7.0, 2),
        ("year", day_of_year, 365.25, 4),
    )
    for prefix, values, period, harmonics in harmonic_groups:
        for harmonic in range(1, harmonics + 1):
            angle = 2 * np.pi * harmonic * values / period
            features[f"{prefix}_sin_{harmonic}"] = np.sin(angle)
            features[f"{prefix}_cos_{harmonic}"] = np.cos(angle)

    return features


def build_feature_frame(load: pd.Series) -> pd.DataFrame:
    validate_hourly_index(load.index)
    features = build_calendar_features(load.index)
    lag_names: list[str] = []
    for lag in WEEKLY_LAGS:
        name = f"lag_{lag}"
        features[name] = load.shift(lag)
        lag_names.append(name)

    features["weekly_lag_mean"] = features[lag_names].mean(axis=1)
    features["weekly_lag_median"] = features[lag_names].median(axis=1)
    features["weekly_lag_std"] = features[lag_names].std(axis=1)
    previous_week = features[f"lag_{WEEK_HOURS}"]
    features["previous_week_mean"] = previous_week.rolling(WEEK_HOURS).mean()
    features["recent_four_week_mean"] = previous_week.rolling(ROLLING_FEATURE_WINDOW_HOURS).mean()
    features["lag_change_one_week"] = (
        features[f"lag_{WEEK_HOURS}"] - features[f"lag_{2 * WEEK_HOURS}"]
    )
    return features


def make_model() -> LGBMRegressor:
    return LGBMRegressor(
        **asdict(MODEL_PARAMETERS),
        verbosity=-1,
        n_jobs=-1,
    )


def complete_training_rows(
    features: pd.DataFrame,
    before: pd.Timestamp,
) -> pd.Series:
    return (features.index < before) & features.notna().all(axis=1)


def train_and_predict(
    load: pd.Series,
    features: pd.DataFrame,
    start: pd.Timestamp,
) -> ModelForecast:
    if not load.index.equals(features.index):
        raise ValueError("Load and feature indexes must match exactly")
    target = forecast_index(start)
    if not target.isin(features.index).all():
        raise ValueError(f"Dataset does not cover forecast horizon beginning {start}")

    train_rows = complete_training_rows(features, start)
    training_count = int(train_rows.sum())
    if training_count < MIN_TRAINING_ROWS:
        raise ValueError(
            f"At least {MIN_TRAINING_ROWS} complete training rows are required before "
            f"{start.isoformat()}; found {training_count}"
        )

    history_only = load.mask(load.index >= start)
    target_features = build_feature_frame(history_only).loc[target]
    if target_features.isna().any().any():
        raise ValueError(f"Target features contain missing values for horizon {start}")

    model = make_model()
    model.fit(features.loc[train_rows], load.loc[train_rows])
    predictions = np.asarray(model.predict(target_features), dtype=np.float64)
    predictions = np.round(predictions, decimals=3)
    if not np.isfinite(predictions).all() or (predictions <= 0).any():
        raise ValueError("Model predictions must be finite and strictly positive")
    return ModelForecast(
        predictions=predictions,
        training_index=features.index[train_rows],
    )


def calculate_metrics(
    actual: npt.ArrayLike,
    predicted: npt.ArrayLike,
) -> ForecastMetrics:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted arrays must have the same shape")
    if actual_values.size == 0:
        raise ValueError("Actual and predicted arrays must not be empty")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Actual and predicted values must be finite")
    if np.any(actual_values == 0):
        raise ValueError("MAPE is undefined when an actual value is zero")

    errors = predicted_values - actual_values
    percentage_errors = np.abs(errors / actual_values)
    if not np.isfinite(percentage_errors).all():
        raise ValueError("Percentage errors must be finite")
    residual_sum = float(np.sum(np.square(errors)))
    total_sum = float(np.sum(np.square(actual_values - actual_values.mean())))
    r_squared = 1.0 - residual_sum / total_sum if total_sum else None
    return ForecastMetrics(
        mae_mw=float(np.mean(np.abs(errors))),
        rmse_mw=float(np.sqrt(np.mean(np.square(errors)))),
        mape_percent=float(np.mean(percentage_errors) * 100),
        mean_error_mw=float(np.mean(errors)),
        r_squared=r_squared,
    )


def relative_mape_improvement(
    model_mape: float,
    baseline_mape: float,
) -> float | None:
    if baseline_mape == 0:
        return None
    return (baseline_mape - model_mape) / baseline_mape * 100


def run_historical_validation(
    load: pd.Series,
    features: pd.DataFrame,
) -> dict[str, float]:
    fold_scores: dict[str, float] = {}
    for start in VALIDATION_STARTS:
        forecast = train_and_predict(load, features, start)
        actual = load.loc[forecast_index(start)].to_numpy()
        metrics = calculate_metrics(actual, forecast.predictions)
        fold_scores[start.isoformat()] = metrics.mape_percent
    return fold_scores


def build_forecast_table(
    target: pd.DatetimeIndex,
    actual: npt.NDArray[np.float64],
    predicted: npt.NDArray[np.float64],
) -> pd.DataFrame:
    errors = predicted - actual
    table = pd.DataFrame(
        {
            "actual_load_mw": actual,
            "forecast_load_mw": predicted,
            "error_mw": errors,
            "absolute_percentage_error_percent": np.abs(errors / actual) * 100,
        },
        index=target,
    )
    table.index.name = TIMESTAMP_COLUMN
    return table


def format_megawatt_tick(value: float, _: float) -> str:
    if abs(value) >= 10_000:
        return f"{value / 1_000:.0f}k"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:,.0f}"


def style_axis(axis: Axes) -> None:
    axis.set_facecolor(SURFACE)
    axis.tick_params(colors=MUTED_TEXT, labelsize=9, length=0)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(AXIS)
    axis.spines["bottom"].set_color(AXIS)
    axis.yaxis.set_major_formatter(FuncFormatter(format_megawatt_tick))


def plot_forecast(
    table: pd.DataFrame,
    output_path: Path,
) -> None:
    plot_style: dict[str, object] = {
        "font.family": "DejaVu Sans",
        "text.color": PRIMARY_TEXT,
        "axes.labelcolor": SECONDARY_TEXT,
        "axes.titlecolor": PRIMARY_TEXT,
    }
    with matplotlib.rc_context(plot_style):
        figure = Figure(figsize=(14, 8), facecolor=SURFACE)
        FigureCanvasAgg(figure)
        load_axis, error_axis = figure.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": (3.2, 1), "hspace": 0.08},
        )
        index = table.index
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("Forecast table must use a DatetimeIndex")
        actual = table["actual_load_mw"]
        predicted = table["forecast_load_mw"]
        errors = table["error_mw"]
        metrics = calculate_metrics(actual.to_numpy(), predicted.to_numpy())
        period_label = format_date_range(index[0], index[-1])
        timezone_label = str(index.tz) if index.tz is not None else "local time"

        load_axis.plot(
            index,
            actual,
            color=ACTUAL_COLOR,
            linewidth=2,
            solid_capstyle="round",
            label="Actual load",
        )
        load_axis.plot(
            index,
            predicted,
            color=FORECAST_COLOR,
            linewidth=2,
            linestyle=(0, (4, 2)),
            dash_capstyle="round",
            label="Forecast load",
        )
        load_axis.fill_between(
            index,
            actual,
            predicted,
            color=FORECAST_COLOR,
            alpha=0.08,
            linewidth=0,
        )
        load_axis.set_ylabel("Load (MW)")
        load_axis.legend(
            loc="upper left",
            ncols=2,
            frameon=False,
            fontsize=10,
            labelcolor=SECONDARY_TEXT,
        )

        error_axis.plot(
            index,
            errors,
            color=SECONDARY_TEXT,
            linewidth=1.5,
            solid_capstyle="round",
        )
        error_axis.fill_between(
            index,
            0,
            errors,
            color=SECONDARY_TEXT,
            alpha=0.12,
            linewidth=0,
        )
        error_axis.axhline(0, color=AXIS, linewidth=1)
        error_axis.set_ylabel("Error\n(MW)")
        error_limit = max(1_000.0, float(np.max(np.abs(errors))) * 1.15)
        error_axis.set_ylim(-error_limit, error_limit)
        error_axis.set_xlim(index[0], index[-1])
        daily_ticks = pd.date_range(index[0].floor("D"), index[-1], freq="D")
        error_axis.set_xticks(daily_ticks.to_pydatetime())
        error_axis.set_xticklabels(
            [
                (
                    f"{WEEKDAY_ABBREVIATIONS[timestamp.dayofweek]}\n"
                    f"{timestamp.day:02d} {MONTH_NAMES[timestamp.month - 1][:3]}"
                )
                for timestamp in daily_ticks
            ]
        )
        error_axis.set_xlabel(f"Timestamp ({timezone_label})")

        for axis in (load_axis, error_axis):
            style_axis(axis)

        figure.suptitle(
            "German electricity load: actual vs forecast",
            x=0.075,
            y=0.975,
            ha="left",
            fontsize=18,
            fontweight="bold",
            color=PRIMARY_TEXT,
        )
        figure.text(
            0.975,
            0.967,
            (
                f"MAPE {metrics.mape_percent:.2f}%   "
                f"MAE {metrics.mae_mw:,.0f} MW   "
                f"RMSE {metrics.rmse_mw:,.0f} MW"
            ),
            ha="right",
            va="top",
            fontsize=10,
            color=SECONDARY_TEXT,
        )
        figure.text(
            0.075,
            0.935,
            (
                f"{len(table)} hourly observations, {period_label} | "
                "Shaded area shows forecast error"
            ),
            ha="left",
            va="top",
            fontsize=10,
            color=SECONDARY_TEXT,
        )
        figure.subplots_adjust(left=0.075, right=0.975, bottom=0.105, top=0.89)
        figure.savefig(output_path, dpi=180, facecolor=SURFACE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(
    load: pd.Series,
    features: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path, Path, ForecastMetrics]:
    validation_scores = run_historical_validation(load, features)
    forecast = train_and_predict(load, features, FORECAST_START)
    target = forecast_index()
    actual = np.asarray(load.loc[target].to_numpy(), dtype=np.float64)
    metrics = calculate_metrics(actual, forecast.predictions)
    weekly_naive = np.asarray(
        features.loc[target, f"lag_{WEEK_HOURS}"].to_numpy(),
        dtype=np.float64,
    )
    weekly_naive_metrics = calculate_metrics(actual, weekly_naive)
    mape_improvement = relative_mape_improvement(
        model_mape=metrics.mape_percent,
        baseline_mape=weekly_naive_metrics.mape_percent,
    )
    training_index = forecast.training_index

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{OUTPUT_STEM}.csv"
    metrics_path = output_dir / f"{OUTPUT_STEM}_metrics.json"
    plot_path = output_dir / f"{OUTPUT_STEM}.png"
    table = build_forecast_table(target, actual, forecast.predictions)

    lock_path = output_dir / f".{OUTPUT_STEM}.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        with TemporaryDirectory(dir=output_dir, prefix=f".{OUTPUT_STEM}-") as temporary:
            temporary_dir = Path(temporary)
            temporary_csv = temporary_dir / csv_path.name
            temporary_metrics = temporary_dir / metrics_path.name
            temporary_plot = temporary_dir / plot_path.name

            table.to_csv(temporary_csv, float_format="%.3f")
            plot_forecast(table, temporary_plot)
            artifact_digests = ArtifactDigests(
                forecast_csv_sha256=sha256_file(temporary_csv),
                plot_png_sha256=sha256_file(temporary_plot),
            )
            report = ForecastReport(
                model=LGBMRegressor.__name__,
                model_library="lightgbm",
                model_library_version=lightgbm.__version__,
                model_parameters=MODEL_PARAMETERS,
                feature_timezone="Europe/Berlin",
                target_start_utc=FORECAST_START.isoformat(),
                target_end_utc_exclusive=FORECAST_END.isoformat(),
                forecast_hours=FORECAST_HOURS,
                training_start_utc=training_index[0].isoformat(),
                training_end_utc=training_index[-1].isoformat(),
                training_hours=len(training_index),
                historical_validation_mape_percent_by_start_utc=validation_scores,
                historical_validation_mean_mape_percent=float(
                    np.mean(list(validation_scores.values()))
                ),
                artifact_digests=artifact_digests,
                accuracy=metrics,
                weekly_naive_baseline=weekly_naive_metrics,
                mape_improvement_over_weekly_naive_percent=mape_improvement,
            )
            report_json = json.dumps(asdict(report), indent=2, allow_nan=False) + "\n"
            temporary_metrics.write_text(report_json, encoding="utf-8")

            artifact_pairs = (
                (temporary_csv, csv_path),
                (temporary_plot, plot_path),
                (temporary_metrics, metrics_path),
            )
            for temporary_path, final_path in artifact_pairs:
                os.replace(temporary_path, final_path)

    return csv_path, metrics_path, plot_path, metrics


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    load = load_hourly_series(args.data)
    features = build_feature_frame(load)
    csv_path, metrics_path, plot_path, metrics = write_outputs(
        load,
        features,
        args.output_dir,
    )

    print(f"Forecast complete: {FORECAST_HOURS} hourly values")
    print(f"MAPE: {metrics.mape_percent:.3f}%")
    print(f"MAE: {metrics.mae_mw:,.1f} MW")
    print(f"RMSE: {metrics.rmse_mw:,.1f} MW")
    r_squared = "undefined" if metrics.r_squared is None else f"{metrics.r_squared:.4f}"
    print(f"R-squared: {r_squared}")
    print(f"Mean error: {metrics.mean_error_mw:,.1f} MW")
    print(f"Forecast table: {csv_path}")
    print(f"Metrics report: {metrics_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
