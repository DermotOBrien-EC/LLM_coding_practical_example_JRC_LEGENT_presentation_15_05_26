from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

LOCAL_PACKAGES = Path(__file__).with_name(".python_packages")
if LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

TIMESTAMP_COLUMN = "utc_timestamp"
LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
TARGET_START = pd.Timestamp(datetime(2020, 1, 1, tzinfo=timezone.utc))
TARGET_HOURS = 168
RIDGE_ALPHA = 100.0


@dataclass(frozen=True)
class Metrics:
    mae_mw: float
    rmse_mw: float
    mape_percent: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Forecast German hourly electricity load for the first week of January 2020."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).with_name("opsd_de_load.csv"),
        help="CSV containing utc_timestamp and DE_load_actual_entsoe_transparency columns.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).with_name("forecast_de_load_jan2020.csv"),
        help="Path to write hourly forecast and actual values.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path(__file__).with_name("forecast_de_load_jan2020.png"),
        help="Path to write the forecast-vs-actual plot.",
    )
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=[TIMESTAMP_COLUMN])
    missing = {TIMESTAMP_COLUMN, LOAD_COLUMN} - set(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"missing required column(s): {names}")

    frame = frame[[TIMESTAMP_COLUMN, LOAD_COLUMN]].dropna().copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True)
    frame = frame.sort_values(TIMESTAMP_COLUMN).set_index(TIMESTAMP_COLUMN)
    if frame.empty:
        raise ValueError(f"no usable load records found in {path}")
    return frame


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    index = frame.index
    day_of_year = index.dayofyear.astype(float)
    days_since_start = (index - index[0]).total_seconds() / 86_400.0

    features = pd.DataFrame(index=index)
    features["trend_years"] = days_since_start / 365.25
    features["daily_sin_1"] = np.sin(2.0 * np.pi * day_of_year / 366.0)
    features["daily_cos_1"] = np.cos(2.0 * np.pi * day_of_year / 366.0)
    features["daily_sin_2"] = np.sin(4.0 * np.pi * day_of_year / 366.0)
    features["daily_cos_2"] = np.cos(4.0 * np.pi * day_of_year / 366.0)
    features["is_weekend"] = (index.weekday >= 5).astype(float)

    for hour in range(1, 24):
        features[f"hour_{hour:02d}"] = (index.hour == hour).astype(float)
    for weekday in range(1, 7):
        features[f"weekday_{weekday}"] = (index.weekday == weekday).astype(float)
    for month in range(2, 13):
        features[f"month_{month:02d}"] = (index.month == month).astype(float)
    for day in range(1, 8):
        features[f"jan_day_{day}"] = ((index.month == 1) & (index.day == day)).astype(float)

    load = frame[LOAD_COLUMN]
    features["lag_1_week"] = load.shift(168) / 10_000.0
    features["lag_2_weeks"] = load.shift(336) / 10_000.0
    features["lag_52_weeks"] = load.shift(24 * 364) / 10_000.0
    features["lag_1_year"] = load.shift(24 * 365) / 10_000.0
    features["weekly_change"] = features["lag_1_week"] - features["lag_2_weeks"]
    return features


def fit_ridge_regression(features: pd.DataFrame, target: pd.Series, alpha: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    x = features.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])

    means = design[:, 1:].mean(axis=0)
    stds = design[:, 1:].std(axis=0)
    stds[stds == 0.0] = 1.0
    standardized = design.copy()
    standardized[:, 1:] = (standardized[:, 1:] - means) / stds

    penalty = alpha * np.eye(standardized.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(standardized.T @ standardized + penalty, standardized.T @ y)
    return coefficients, means, stds


def predict(features: pd.DataFrame, coefficients: NDArray[np.float64], means: NDArray[np.float64], stds: NDArray[np.float64]) -> NDArray[np.float64]:
    x = features.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    design[:, 1:] = (design[:, 1:] - means) / stds
    return design @ coefficients


def forecast(frame: pd.DataFrame) -> pd.DataFrame:
    target_end = TARGET_START + pd.Timedelta(hours=TARGET_HOURS)
    features = build_features(frame)
    modeled = pd.concat([frame[LOAD_COLUMN], features], axis=1).dropna()

    train = modeled.loc[modeled.index < TARGET_START]
    target = modeled.loc[(modeled.index >= TARGET_START) & (modeled.index < target_end)]
    if len(target) != TARGET_HOURS:
        raise ValueError(f"expected {TARGET_HOURS} target rows, found {len(target)}")

    train_x = train.drop(columns=[LOAD_COLUMN])
    train_y = train[LOAD_COLUMN]
    target_x = target.drop(columns=[LOAD_COLUMN])
    coefficients, means, stds = fit_ridge_regression(train_x, train_y, RIDGE_ALPHA)

    result = pd.DataFrame(index=target.index)
    result["actual_mw"] = target[LOAD_COLUMN]
    result["forecast_mw"] = predict(target_x, coefficients, means, stds)
    result["error_mw"] = result["forecast_mw"] - result["actual_mw"]
    result["absolute_percentage_error"] = (result["error_mw"].abs() / result["actual_mw"])
    return result


def calculate_metrics(result: pd.DataFrame) -> Metrics:
    errors = result["error_mw"]
    return Metrics(
        mae_mw=float(errors.abs().mean()),
        rmse_mw=float(np.sqrt(np.mean(np.square(errors)))),
        mape_percent=float(result["absolute_percentage_error"].mean() * 100.0),
    )


def write_forecast_csv(path: Path, result: pd.DataFrame) -> None:
    output = result.copy()
    output.index.name = "utc_timestamp"
    output.to_csv(path, float_format="%.3f")


def write_plot(path: Path, result: pd.DataFrame, metrics: Metrics) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(13.5, 6.8), constrained_layout=True)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.plot(result.index, result["actual_mw"], label="Actual load", color="#2a78d6", linewidth=2.2)
    ax.plot(result.index, result["forecast_mw"], label="Forecast", color="#eb6834", linewidth=2.2, linestyle=(0, (6, 4)))

    ax.set_title("German load forecast: first week of January 2020", loc="left", fontsize=17, fontweight="bold", color="#0b0b0b")
    ax.text(
        0.0,
        1.02,
        f"Ridge regression trained on pre-2020 hourly history. MAE {metrics.mae_mw:,.0f} MW, RMSE {metrics.rmse_mw:,.0f} MW, MAPE {metrics.mape_percent:.2f}%.",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#52514e",
    )
    ax.set_ylabel("Load, MW", color="#52514e")
    ax.set_xlabel("Hour, UTC", color="#52514e")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value / 1000:.0f}k")
    ax.grid(axis="y", color="#e5e2dc", linewidth=0.9)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8f8a82")
    ax.spines["bottom"].set_color("#8f8a82")
    ax.tick_params(colors="#696761")
    ax.legend(frameon=True, facecolor="#f3f1ed", edgecolor="#dedbd3", loc="upper right")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = load_data(args.data)
    result = forecast(frame)
    metrics = calculate_metrics(result)
    write_forecast_csv(args.output_csv, result)
    write_plot(args.plot, result, metrics)

    target_end = TARGET_START + pd.Timedelta(hours=TARGET_HOURS)
    training_rows = int((frame.index < TARGET_START).sum())
    print("German hourly electricity load forecast")
    print(f"Target period: {TARGET_START.isoformat()} to {target_end.isoformat()} exclusive")
    print(f"Training rows: {training_rows}")
    print(f"Forecast rows: {len(result)}")
    print(f"MAE: {metrics.mae_mw:,.1f} MW")
    print(f"RMSE: {metrics.rmse_mw:,.1f} MW")
    print(f"MAPE: {metrics.mape_percent:.2f}%")
    print(f"Forecast CSV: {args.output_csv}")
    print(f"Plot: {args.plot}")


if __name__ == "__main__":
    main()
