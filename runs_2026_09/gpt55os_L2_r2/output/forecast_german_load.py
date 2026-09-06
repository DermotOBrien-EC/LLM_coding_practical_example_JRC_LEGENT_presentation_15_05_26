from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LoadPoint:
    timestamp: datetime
    load_mw: float


@dataclass(frozen=True)
class ForecastRow:
    timestamp: datetime
    actual_mw: float
    forecast_mw: float
    seasonal_naive_mw: float
    error_mw: float
    absolute_error_mw: float
    absolute_percentage_error: float


@dataclass(frozen=True)
class TrainedModel:
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    feature_names: list[str]


GERMAN_LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
TIMESTAMP_COLUMN = "utc_timestamp"


def read_load_points(path: Path) -> list[LoadPoint]:
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError:
        return read_load_points_with_csv(path)

    frame = pd.read_csv(path, parse_dates=[TIMESTAMP_COLUMN])
    required = {TIMESTAMP_COLUMN, GERMAN_LOAD_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    frame = frame[[TIMESTAMP_COLUMN, GERMAN_LOAD_COLUMN]].dropna()
    points: list[LoadPoint] = []
    for row in frame.itertuples(index=False):
        timestamp = row[0].to_pydatetime()
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        points.append(LoadPoint(timestamp=timestamp, load_mw=float(row[1])))
    return sorted(points, key=lambda point: point.timestamp)


def read_load_points_with_csv(path: Path) -> list[LoadPoint]:
    points: list[LoadPoint] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {TIMESTAMP_COLUMN, GERMAN_LOAD_COLUMN}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        for row in reader:
            value = row[GERMAN_LOAD_COLUMN]
            if not value:
                continue
            timestamp = datetime.fromisoformat(row[TIMESTAMP_COLUMN])
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            points.append(LoadPoint(timestamp=timestamp, load_mw=float(value)))
    return sorted(points, key=lambda point: point.timestamp)


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


def is_german_holiday_or_low_load_day(day: date) -> bool:
    easter = easter_sunday(day.year)
    movable = {
        easter - timedelta(days=2),
        easter + timedelta(days=1),
        easter + timedelta(days=39),
        easter + timedelta(days=50),
        easter + timedelta(days=60),
    }
    fixed = {
        (1, 1),
        (1, 6),
        (5, 1),
        (10, 3),
        (12, 24),
        (12, 25),
        (12, 26),
        (12, 31),
    }
    return day in movable or (day.month, day.day) in fixed


def feature_vector(timestamp: datetime, load_by_time: dict[datetime, float]) -> tuple[list[float], list[str]]:
    lag_time = timestamp - timedelta(hours=168)
    if lag_time not in load_by_time:
        raise ValueError(f"Missing 168 hour lag for {timestamp.isoformat()}")

    values: list[float] = [load_by_time[lag_time]]
    names: list[str] = ["lag_168h_load_mw"]

    hour = timestamp.hour
    for harmonic in (1, 2, 3):
        angle = 2.0 * math.pi * harmonic * hour / 24.0
        values.extend([math.sin(angle), math.cos(angle)])
        names.extend([f"hour_sin_{harmonic}", f"hour_cos_{harmonic}"])

    day_of_week = timestamp.weekday()
    for candidate in range(6):
        values.append(1.0 if day_of_week == candidate else 0.0)
        names.append(f"dow_{candidate}")

    for candidate in range(1, 12):
        values.append(1.0 if timestamp.month == candidate else 0.0)
        names.append(f"month_{candidate}")

    day_of_year = timestamp.timetuple().tm_yday
    year_length = 366.0 if is_leap_year(timestamp.year) else 365.0
    for harmonic in (1, 2, 3):
        angle = 2.0 * math.pi * harmonic * day_of_year / year_length
        values.extend([math.sin(angle), math.cos(angle)])
        names.extend([f"annual_sin_{harmonic}", f"annual_cos_{harmonic}"])

    current_day = timestamp.date()
    previous_day = current_day - timedelta(days=1)
    next_day = current_day + timedelta(days=1)
    values.extend(
        [
            1.0 if day_of_week >= 5 else 0.0,
            1.0 if is_german_holiday_or_low_load_day(current_day) else 0.0,
            1.0 if is_german_holiday_or_low_load_day(previous_day) else 0.0,
            1.0 if is_german_holiday_or_low_load_day(next_day) else 0.0,
            1.0 if timestamp.month == 1 and timestamp.day <= 7 else 0.0,
            1.0 if timestamp.month == 12 and timestamp.day >= 24 else 0.0,
        ]
    )
    names.extend(
        [
            "is_weekend",
            "is_holiday_or_low_load_day",
            "previous_day_holiday_or_low_load",
            "next_day_holiday_or_low_load",
            "first_week_of_january",
            "christmas_new_year_period",
        ]
    )
    return values, names


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def build_training_matrix(
    points: Iterable[LoadPoint],
    load_by_time: dict[datetime, float],
    train_end: datetime,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows: list[list[float]] = []
    targets: list[float] = []
    feature_names: list[str] | None = None
    min_time = min(load_by_time) + timedelta(hours=168)
    for point in points:
        if point.timestamp < min_time or point.timestamp >= train_end:
            continue
        features, names = feature_vector(point.timestamp, load_by_time)
        if feature_names is None:
            feature_names = names
        rows.append(features)
        targets.append(point.load_mw)

    if not rows or feature_names is None:
        raise ValueError("No usable training rows")
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float), feature_names


def fit_ridge_regression(features: np.ndarray, target: np.ndarray, names: list[str], alpha: float) -> TrainedModel:
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale == 0.0] = 1.0
    standardized = (features - feature_mean) / feature_scale
    design = np.column_stack([np.ones(standardized.shape[0]), standardized])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    return TrainedModel(
        coefficients=coefficients,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        feature_names=["intercept", *names],
    )


def predict(model: TrainedModel, features: list[float]) -> float:
    raw = np.asarray(features, dtype=float)
    standardized = (raw - model.feature_mean) / model.feature_scale
    design_row = np.concatenate([[1.0], standardized])
    return float(design_row @ model.coefficients)


def forecast_period(
    model: TrainedModel,
    start: datetime,
    hours: int,
    load_by_time: dict[datetime, float],
) -> list[ForecastRow]:
    rows: list[ForecastRow] = []
    for offset in range(hours):
        timestamp = start + timedelta(hours=offset)
        if timestamp not in load_by_time:
            raise ValueError(f"Missing actual load for {timestamp.isoformat()}")
        features, _ = feature_vector(timestamp, load_by_time)
        actual = load_by_time[timestamp]
        forecast = predict(model, features)
        seasonal_naive = load_by_time[timestamp - timedelta(hours=168)]
        error = forecast - actual
        rows.append(
            ForecastRow(
                timestamp=timestamp,
                actual_mw=actual,
                forecast_mw=forecast,
                seasonal_naive_mw=seasonal_naive,
                error_mw=error,
                absolute_error_mw=abs(error),
                absolute_percentage_error=abs(error) / actual * 100.0,
            )
        )
    return rows


def compute_metrics(rows: list[ForecastRow]) -> dict[str, float]:
    actual = np.asarray([row.actual_mw for row in rows], dtype=float)
    forecast = np.asarray([row.forecast_mw for row in rows], dtype=float)
    naive = np.asarray([row.seasonal_naive_mw for row in rows], dtype=float)
    error = forecast - actual
    naive_error = naive - actual
    return {
        "hours": float(len(rows)),
        "mean_actual_mw": float(actual.mean()),
        "mean_forecast_mw": float(forecast.mean()),
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error * error))),
        "mape_percent": float(np.mean(np.abs(error) / actual) * 100.0),
        "smape_percent": float(np.mean(2.0 * np.abs(error) / (np.abs(actual) + np.abs(forecast))) * 100.0),
        "wape_percent": float(np.sum(np.abs(error)) / np.sum(actual) * 100.0),
        "mean_bias_mw": float(np.mean(error)),
        "seasonal_naive_mae_mw": float(np.mean(np.abs(naive_error))),
        "seasonal_naive_rmse_mw": float(np.sqrt(np.mean(naive_error * naive_error))),
        "mae_improvement_vs_seasonal_naive_percent": float(
            (1.0 - np.mean(np.abs(error)) / np.mean(np.abs(naive_error))) * 100.0
        ),
    }


def write_forecast_csv(path: Path, rows: list[ForecastRow]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "utc_timestamp",
                "actual_mw",
                "forecast_mw",
                "seasonal_naive_mw",
                "error_mw",
                "absolute_error_mw",
                "absolute_percentage_error",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.timestamp.isoformat(),
                    f"{row.actual_mw:.3f}",
                    f"{row.forecast_mw:.3f}",
                    f"{row.seasonal_naive_mw:.3f}",
                    f"{row.error_mw:.3f}",
                    f"{row.absolute_error_mw:.3f}",
                    f"{row.absolute_percentage_error:.6f}",
                ]
            )


def write_plot(output_stem: Path, rows: list[ForecastRow]) -> Path:
    try:
        import matplotlib.dates as mdates  # type: ignore[import-not-found]
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        svg_path = output_stem.with_suffix(".svg")
        write_svg_plot(svg_path, rows)
        return svg_path

    timestamps = [row.timestamp for row in rows]
    actual = [row.actual_mw for row in rows]
    forecast = [row.forecast_mw for row in rows]

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(timestamps, actual, label="Actual load", color="#2a78d6", linewidth=2.0)
    ax.plot(timestamps, forecast, label="Forecast", color="#eb6834", linewidth=2.0)
    ax.set_title("German hourly electricity load, first week of January 2020")
    ax.set_ylabel("Load (MW)")
    ax.set_xlabel("UTC time")
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    png_path = output_stem.with_suffix(".png")
    fig.savefig(png_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png_path


def write_svg_plot(path: Path, rows: list[ForecastRow]) -> None:
    width = 1200
    height = 560
    left = 80
    right = 30
    top = 64
    bottom = 72
    plot_width = width - left - right
    plot_height = height - top - bottom

    actual = [row.actual_mw for row in rows]
    forecast = [row.forecast_mw for row in rows]
    all_values = actual + forecast
    minimum = math.floor((min(all_values) - 1000.0) / 1000.0) * 1000.0
    maximum = math.ceil((max(all_values) + 1000.0) / 1000.0) * 1000.0

    def x_position(index: int) -> float:
        return left + plot_width * index / (len(rows) - 1)

    def y_position(value: float) -> float:
        return top + plot_height * (maximum - value) / (maximum - minimum)

    def polyline(values: list[float]) -> str:
        return " ".join(f"{x_position(index):.2f},{y_position(value):.2f}" for index, value in enumerate(values))

    y_ticks = np.linspace(minimum, maximum, 6)
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">German hourly electricity load forecast</title>',
        '<desc id="desc">Line chart comparing forecast German electricity load with actual load for 168 UTC hours from 2020-01-01.</desc>',
        '<rect width="100%" height="100%" fill="#fcfcfb"/>',
        '<style>text{font-family:system-ui,-apple-system,Segoe UI,sans-serif;fill:#0b0b0b}.muted{fill:#52514e}.axis{stroke:#c3c2b7;stroke-width:1}.grid{stroke:#e1e0d9;stroke-width:1}.line{fill:none;stroke-width:2.4;stroke-linejoin:round;stroke-linecap:round}</style>',
        '<text x="80" y="34" font-size="22" font-weight="700">German hourly electricity load forecast</text>',
        '<text x="80" y="56" font-size="13" class="muted">First 168 UTC hours of January 2020</text>',
    ]

    for tick in y_ticks:
        y = y_position(float(tick))
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="12" class="muted">{tick:,.0f}</text>')

    for index, row in enumerate(rows):
        if row.timestamp.hour == 0:
            x = x_position(index)
            parts.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{height - bottom}" class="grid"/>')
            parts.append(
                f'<text x="{x:.2f}" y="{height - 36}" text-anchor="middle" font-size="12" class="muted">{row.timestamp.strftime("%b %d")}</text>'
            )

    parts.extend(
        [
            f'<line x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}" class="axis"/>',
            f'<line x1="{left}" x2="{left}" y1="{top}" y2="{height - bottom}" class="axis"/>',
            f'<polyline points="{polyline(actual)}" class="line" stroke="#2a78d6"/>',
            f'<polyline points="{polyline(forecast)}" class="line" stroke="#eb6834"/>',
            '<circle cx="895" cy="31" r="5" fill="#2a78d6"/><text x="908" y="36" font-size="13">Actual load</text>',
            '<circle cx="1005" cy="31" r="5" fill="#eb6834"/><text x="1018" y="36" font-size="13">Forecast</text>',
            f'<text x="{left - 54}" y="{top + plot_height / 2:.2f}" transform="rotate(-90 {left - 54} {top + plot_height / 2:.2f})" text-anchor="middle" font-size="13" class="muted">Load (MW)</text>',
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 12}" text-anchor="middle" font-size="13" class="muted">UTC time</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def write_metrics(path: Path, metrics: dict[str, float]) -> None:
    with path.open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast German hourly electricity load for early January 2020.")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--input", type=Path, default=script_dir / "opsd_de_load.csv")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "outputs")
    parser.add_argument("--target-start", default="2020-01-01T00:00:00+00:00")
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--ridge-alpha", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = read_load_points(args.input)
    if not points:
        raise ValueError("No load points found")

    load_by_time = {point.timestamp: point.load_mw for point in points}
    target_start = datetime.fromisoformat(args.target_start)
    if target_start.tzinfo is None:
        target_start = target_start.replace(tzinfo=timezone.utc)

    features, target, names = build_training_matrix(points, load_by_time, target_start)
    model = fit_ridge_regression(features, target, names, alpha=args.ridge_alpha)
    rows = forecast_period(model, target_start, args.hours, load_by_time)
    metrics = compute_metrics(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    forecast_csv = args.output_dir / "german_load_forecast_2020_01_01_to_2020_01_07.csv"
    metrics_json = args.output_dir / "german_load_forecast_metrics.json"
    plot_path = write_plot(args.output_dir / "german_load_forecast_2020_01_01_to_2020_01_07", rows)
    write_forecast_csv(forecast_csv, rows)
    write_metrics(metrics_json, metrics)

    print(f"Training rows: {features.shape[0]}")
    print(f"Forecast hours: {len(rows)}")
    print(f"Forecast CSV: {forecast_csv}")
    print(f"Metrics JSON: {metrics_json}")
    print(f"Plot: {plot_path}")
    print(f"MAE: {metrics['mae_mw']:.1f} MW")
    print(f"RMSE: {metrics['rmse_mw']:.1f} MW")
    print(f"MAPE: {metrics['mape_percent']:.2f}%")
    print(f"WAPE: {metrics['wape_percent']:.2f}%")
    print(f"Mean bias: {metrics['mean_bias_mw']:.1f} MW")
    print(f"Seasonal naive MAE: {metrics['seasonal_naive_mae_mw']:.1f} MW")
    print(f"MAE improvement vs seasonal naive: {metrics['mae_improvement_vs_seasonal_naive_percent']:.1f}%")


if __name__ == "__main__":
    main()
