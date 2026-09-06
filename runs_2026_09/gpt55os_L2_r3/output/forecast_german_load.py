from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
TIME_COLUMN = "utc_timestamp"
LOCAL_TZ = "Europe/Berlin"
LAG_HOURS = (24, 48, 72, 168, 336)
TARGET_START_UTC = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
TARGET_HOURS = 168
VALIDATION_YEARS = (2017, 2018, 2019)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


@dataclass(frozen=True)
class Metrics:
    mae_mw: float
    rmse_mw: float
    mape_percent: float
    bias_mw: float


@dataclass(frozen=True)
class RidgeModel:
    alpha: float
    feature_names: tuple[str, ...]
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    coefficients: np.ndarray
    reference_start_utc: pd.Timestamp

    def predict(self, x: np.ndarray) -> np.ndarray:
        scaled = (x - self.x_mean) / self.x_scale
        return self.y_mean + scaled @ self.coefficients


def easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def german_national_holidays(years: Iterable[int]) -> set[date]:
    holidays: set[date] = set()
    for year in years:
        easter = easter_sunday(year)
        holidays.update(
            {
                date(year, 1, 1),
                easter - timedelta(days=2),
                easter + timedelta(days=1),
                date(year, 5, 1),
                easter + timedelta(days=39),
                easter + timedelta(days=50),
                date(year, 10, 3),
                date(year, 12, 25),
                date(year, 12, 26),
            }
        )
    return holidays


def load_data(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    expected = {TIME_COLUMN, LOAD_COLUMN}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], utc=True)
    frame[LOAD_COLUMN] = pd.to_numeric(frame[LOAD_COLUMN], errors="raise")
    frame = frame[[TIME_COLUMN, LOAD_COLUMN]].dropna().sort_values(TIME_COLUMN)
    frame = frame.set_index(TIME_COLUMN)

    if frame.index.has_duplicates:
        raise ValueError("The input has duplicate timestamps")

    expected_index = pd.date_range(frame.index.min(), frame.index.max(), freq="h")
    if not frame.index.equals(expected_index):
        missing_count = len(expected_index.difference(frame.index))
        extra_count = len(frame.index.difference(expected_index))
        raise ValueError(
            "The input must be a continuous hourly UTC series "
            f"(missing={missing_count}, extra={extra_count})"
        )

    return frame


def one_hot(values: np.ndarray, size: int, prefix: str) -> tuple[list[np.ndarray], list[str]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    for value in range(size):
        columns.append((values == value).astype(float))
        names.append(f"{prefix}_{value:02d}")
    return columns, names


def calendar_features(
    index: pd.DatetimeIndex, reference_start_utc: pd.Timestamp
) -> tuple[np.ndarray, tuple[str, ...]]:
    local = index.tz_convert(LOCAL_TZ)
    years = range(int(local.year.min()) - 1, int(local.year.max()) + 2)
    holidays = german_national_holidays(years)

    hour = local.hour.to_numpy(dtype=int)
    day_of_week = local.dayofweek.to_numpy(dtype=int)
    month = (local.month.to_numpy(dtype=int) - 1).astype(int)
    local_dates = np.array([ts.date() for ts in local], dtype=object)
    is_holiday = np.array([day in holidays for day in local_dates], dtype=float)
    is_weekend = (day_of_week >= 5).astype(float)
    is_workday = ((day_of_week < 5) & (is_holiday == 0)).astype(float)

    elapsed_years = (
        (index - reference_start_utc).total_seconds().to_numpy(dtype=float)
        / (365.2425 * 24.0 * 3600.0)
    )
    daily_phase = 2.0 * np.pi * hour / 24.0
    weekly_phase = 2.0 * np.pi * (day_of_week * 24.0 + hour) / 168.0
    annual_phase = 2.0 * np.pi * ((local.dayofyear.to_numpy(dtype=float) - 1.0) * 24.0 + hour) / (
        365.2425 * 24.0
    )

    columns: list[np.ndarray] = [
        elapsed_years,
        np.sin(daily_phase),
        np.cos(daily_phase),
        np.sin(2.0 * daily_phase),
        np.cos(2.0 * daily_phase),
        np.sin(3.0 * daily_phase),
        np.cos(3.0 * daily_phase),
        np.sin(weekly_phase),
        np.cos(weekly_phase),
        np.sin(2.0 * weekly_phase),
        np.cos(2.0 * weekly_phase),
        np.sin(annual_phase),
        np.cos(annual_phase),
        np.sin(2.0 * annual_phase),
        np.cos(2.0 * annual_phase),
        is_holiday,
        is_weekend,
        is_workday,
        is_holiday * np.sin(daily_phase),
        is_holiday * np.cos(daily_phase),
        is_weekend * np.sin(daily_phase),
        is_weekend * np.cos(daily_phase),
    ]
    names = [
        "elapsed_years",
        "daily_sin_1",
        "daily_cos_1",
        "daily_sin_2",
        "daily_cos_2",
        "daily_sin_3",
        "daily_cos_3",
        "weekly_sin_1",
        "weekly_cos_1",
        "weekly_sin_2",
        "weekly_cos_2",
        "annual_sin_1",
        "annual_cos_1",
        "annual_sin_2",
        "annual_cos_2",
        "is_holiday",
        "is_weekend",
        "is_workday",
        "holiday_daily_sin",
        "holiday_daily_cos",
        "weekend_daily_sin",
        "weekend_daily_cos",
    ]

    extra_columns, extra_names = one_hot(hour, 24, "hour")
    columns.extend(extra_columns)
    names.extend(extra_names)
    extra_columns, extra_names = one_hot(day_of_week, 7, "dow")
    columns.extend(extra_columns)
    names.extend(extra_names)
    extra_columns, extra_names = one_hot(month, 12, "month")
    columns.extend(extra_columns)
    names.extend(extra_names)

    matrix = np.column_stack(columns).astype(float)
    return matrix, tuple(names)


def lag_features(index: pd.DatetimeIndex, known_load: pd.Series) -> tuple[np.ndarray, tuple[str, ...]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    for lag in LAG_HOURS:
        lagged_index = index - pd.Timedelta(hours=lag)
        columns.append(known_load.reindex(lagged_index).to_numpy(dtype=float))
        names.append(f"lag_{lag}h")
    return np.column_stack(columns).astype(float), tuple(names)


def design_matrix(
    index: pd.DatetimeIndex,
    known_load: pd.Series,
    reference_start_utc: pd.Timestamp,
) -> tuple[np.ndarray, tuple[str, ...]]:
    calendar_matrix, calendar_names = calendar_features(index, reference_start_utc)
    lag_matrix, lag_names = lag_features(index, known_load)
    return np.column_stack([calendar_matrix, lag_matrix]), calendar_names + lag_names


def fit_ridge(
    frame: pd.DataFrame,
    cutoff_utc: pd.Timestamp,
    alpha: float,
    reference_start_utc: pd.Timestamp,
) -> RidgeModel:
    train = frame.loc[frame.index < cutoff_utc, LOAD_COLUMN]
    x, feature_names = design_matrix(train.index, train, reference_start_utc)
    y = train.to_numpy(dtype=float)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(y) < 24 * 365:
        raise ValueError(f"Not enough training rows before {cutoff_utc}")

    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale == 0.0] = 1.0
    y_mean = float(y.mean())
    x_scaled = (x - x_mean) / x_scale
    y_centered = y - y_mean

    penalty = alpha * np.eye(x_scaled.shape[1])
    coefficients = np.linalg.solve(x_scaled.T @ x_scaled + penalty, x_scaled.T @ y_centered)
    return RidgeModel(
        alpha=alpha,
        feature_names=feature_names,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        coefficients=coefficients,
        reference_start_utc=reference_start_utc,
    )


def forecast_recursive(
    model: RidgeModel,
    frame: pd.DataFrame,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    known = frame.loc[frame.index < target_index[0], LOAD_COLUMN].copy()
    predictions: list[float] = []
    for timestamp in target_index:
        one_index = pd.DatetimeIndex([timestamp])
        x, feature_names = design_matrix(one_index, known, model.reference_start_utc)
        if feature_names != model.feature_names:
            raise RuntimeError("Feature names changed between training and forecasting")
        if not np.isfinite(x).all():
            missing_lags = [lag for lag in LAG_HOURS if timestamp - pd.Timedelta(hours=lag) not in known.index]
            raise ValueError(f"Missing lag values for {timestamp}: {missing_lags}")
        prediction = float(model.predict(x)[0])
        predictions.append(prediction)
        known.loc[timestamp] = prediction
    return pd.Series(predictions, index=target_index, name="forecast_mw")


def metrics(actual: pd.Series, forecast: pd.Series) -> Metrics:
    aligned_actual, aligned_forecast = actual.align(forecast, join="inner")
    error = aligned_forecast.to_numpy(dtype=float) - aligned_actual.to_numpy(dtype=float)
    actual_values = aligned_actual.to_numpy(dtype=float)
    return Metrics(
        mae_mw=float(np.mean(np.abs(error))),
        rmse_mw=float(np.sqrt(np.mean(error**2))),
        mape_percent=float(np.mean(np.abs(error) / actual_values) * 100.0),
        bias_mw=float(np.mean(error)),
    )


def choose_alpha(frame: pd.DataFrame, reference_start_utc: pd.Timestamp) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for alpha in RIDGE_ALPHAS:
        fold_mae: list[float] = []
        fold_rmse: list[float] = []
        fold_mape: list[float] = []
        for year in VALIDATION_YEARS:
            validation_start = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
            validation_index = pd.date_range(validation_start, periods=TARGET_HOURS, freq="h")
            model = fit_ridge(frame, validation_start, alpha, reference_start_utc)
            forecast = forecast_recursive(model, frame, validation_index)
            actual = frame.loc[validation_index, LOAD_COLUMN]
            fold_metrics = metrics(actual, forecast)
            fold_mae.append(fold_metrics.mae_mw)
            fold_rmse.append(fold_metrics.rmse_mw)
            fold_mape.append(fold_metrics.mape_percent)
        rows.append(
            {
                "alpha": alpha,
                "validation_mae_mw": float(np.mean(fold_mae)),
                "validation_rmse_mw": float(np.mean(fold_rmse)),
                "validation_mape_percent": float(np.mean(fold_mape)),
            }
        )
    best = min(rows, key=lambda row: row["validation_mae_mw"])
    return float(best["alpha"]), rows


def write_forecast_csv(path: Path, actual: pd.Series, forecast: pd.Series) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utc_timestamp", "actual_mw", "forecast_mw", "error_mw", "abs_percentage_error"])
        for timestamp in forecast.index:
            actual_value = float(actual.loc[timestamp])
            forecast_value = float(forecast.loc[timestamp])
            error = forecast_value - actual_value
            writer.writerow(
                [
                    timestamp.isoformat(),
                    f"{actual_value:.3f}",
                    f"{forecast_value:.3f}",
                    f"{error:.3f}",
                    f"{abs(error) / actual_value:.6f}",
                ]
            )


def plot_forecast(path: Path, actual: pd.Series, forecast: pd.Series) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfd")

    ax.plot(
        actual.index,
        actual.to_numpy(dtype=float),
        color="#2f6f9f",
        linewidth=2.0,
        label="Actual load",
    )
    ax.plot(
        forecast.index,
        forecast.to_numpy(dtype=float),
        color="#c44e52",
        linewidth=2.0,
        linestyle="--",
        label="Forecast",
    )

    ax.set_title("German hourly electricity load forecast, 2020-01-01 to 2020-01-07")
    ax.set_ylabel("Load (MW)")
    ax.set_xlabel("UTC timestamp")
    ax.grid(True, axis="y", color="#e6e8ef", linewidth=0.8)
    ax.grid(True, axis="x", color="#f0f1f5", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_validation_table(rows: list[dict[str, float]]) -> None:
    print("Validation on first-week-of-January backtests for 2017, 2018, and 2019")
    print("alpha, validation_MAE_MW, validation_RMSE_MW, validation_MAPE_percent")
    for row in rows:
        print(
            f"{row['alpha']:>7g}, "
            f"{row['validation_mae_mw']:>9.1f}, "
            f"{row['validation_rmse_mw']:>9.1f}, "
            f"{row['validation_mape_percent']:>7.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast German hourly electricity load for 2020-01-01 through 2020-01-07."
    )
    parser.add_argument("--data", type=Path, default=Path("opsd_de_load.csv"))
    parser.add_argument("--forecast-csv", type=Path, default=Path("german_load_forecast_jan2020.csv"))
    parser.add_argument("--plot", type=Path, default=Path("german_load_forecast_jan2020.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_data(args.data)
    reference_start_utc = frame.index.min()
    target_index = pd.date_range(TARGET_START_UTC, periods=TARGET_HOURS, freq="h")
    missing_target_hours = target_index.difference(frame.index)
    if len(missing_target_hours) > 0:
        raise ValueError(f"Actual target values are missing for {len(missing_target_hours)} hours")

    best_alpha, validation_rows = choose_alpha(frame, reference_start_utc)
    model = fit_ridge(frame, TARGET_START_UTC, best_alpha, reference_start_utc)
    forecast = forecast_recursive(model, frame, target_index)
    actual = frame.loc[target_index, LOAD_COLUMN].rename("actual_mw")
    result_metrics = metrics(actual, forecast)

    write_forecast_csv(args.forecast_csv, actual, forecast)
    plot_forecast(args.plot, actual, forecast)

    print_validation_table(validation_rows)
    print()
    print(f"Selected ridge alpha: {best_alpha:g}")
    print(f"Training rows used: {int((frame.index < TARGET_START_UTC).sum())}")
    print(f"Forecast horizon: {TARGET_HOURS} hours")
    print(f"Forecast CSV: {args.forecast_csv}")
    print(f"Plot: {args.plot}")
    print()
    print("Accuracy for 2020-01-01 00:00 UTC through 2020-01-07 23:00 UTC")
    print(f"MAE:  {result_metrics.mae_mw:,.1f} MW")
    print(f"RMSE: {result_metrics.rmse_mw:,.1f} MW")
    print(f"MAPE: {result_metrics.mape_percent:.2f}%")
    print(f"Bias: {result_metrics.bias_mw:,.1f} MW")


if __name__ == "__main__":
    main()
