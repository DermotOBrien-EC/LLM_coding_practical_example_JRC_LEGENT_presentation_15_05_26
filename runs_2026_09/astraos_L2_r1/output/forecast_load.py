#!/usr/bin/env python3
"""Fixed-origin, seven-day German load forecast with historical-only selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path

# Bound numerical-library parallelism for repeatable, lightweight CLI runs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

FORECAST_START = pd.Timestamp("2020-01-01T00:00:00Z")
HORIZON = 168
LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
LAGS = (168, 336, 8736, 8760)
VALIDATION_STARTS = (
    "2018-01-01",
    "2019-01-01",
    "2019-11-25",
    "2019-12-09",
    "2019-12-23",
)


@dataclass(frozen=True)
class ModelConfig:
    max_leaf_nodes: int
    max_iter: int = 200
    learning_rate: float = 0.07
    min_samples_leaf: int = 30
    l2_regularization: float = 10.0
    random_state: int = 42

    @property
    def name(self) -> str:
        return f"hist_gradient_boosting_{self.max_leaf_nodes}_leaves"


CANDIDATES = (ModelConfig(15), ModelConfig(31))


@dataclass(frozen=True)
class ErrorMetrics:
    mae_mw: float
    rmse_mw: float
    mape_percent: float
    wape_percent: float
    mean_error_mw: float


@dataclass(frozen=True)
class ForecastResult:
    predictions: pd.Series
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    train_rows: int
    iterations: int


def target_hours(origin: pd.Timestamp) -> pd.DatetimeIndex:
    if origin.tzinfo is None or origin != origin.floor("h"):
        raise ValueError("Forecast origin must be a timezone-aware whole hour.")
    return pd.date_range(origin.tz_convert("UTC"), periods=HORIZON, freq="h")


def validate_series(data: pd.Series) -> None:
    if data.empty or not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Load must be a nonempty time series.")
    if data.index.tz is None or data.index.hasnans:
        raise ValueError("Timestamps must be valid and timezone-aware.")
    if data.index.has_duplicates:
        raise ValueError("Load contains duplicate timestamps.")
    expected = pd.date_range(data.index[0], data.index[-1], freq="h")
    if not data.index.equals(expected) or not data.index.equals(data.index.floor("h")):
        raise ValueError("Load must be sorted, continuous, and hourly; no hours are imputed.")
    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("All load values must be finite and positive; no loads are imputed.")


def read_load(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    required = {"utc_timestamp", LOAD_COLUMN}
    if not required.issubset(frame.columns):
        raise ValueError(f"CSV must contain {sorted(required)}.")
    index = pd.DatetimeIndex(pd.to_datetime(frame["utc_timestamp"], utc=True, errors="raise"))
    data = pd.Series(
        pd.to_numeric(frame[LOAD_COLUMN], errors="raise").to_numpy(dtype=float),
        index=index,
        name="load_mw",
    ).sort_index()
    validate_series(data)
    return data


def split_data(data: pd.Series) -> tuple[pd.Series, pd.Series]:
    history = data.loc[data.index < FORECAST_START].copy()
    actual = data.reindex(target_hours(FORECAST_START)).copy()
    if actual.isna().any() or len(actual) != HORIZON:
        raise ValueError("All 168 actual target hours must be present; no actuals are imputed.")
    validate_series(history)
    validate_series(actual)
    return history, actual


def build_features(history: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert("Europe/Berlin")
    dates = local.date
    calendar = holidays.country_holidays(
        "DE", years=range(int(local.year.min()) - 1, int(local.year.max()) + 2)
    )
    phase = 2 * np.pi * (local.dayofyear.to_numpy() - 1) / 365.25
    features = pd.DataFrame(
        {
            "hour": local.hour,
            "weekday": local.dayofweek,
            "weekend": (local.dayofweek >= 5).astype(int),
            "month": local.month,
            "day": local.day,
            "annual_sin": np.sin(phase),
            "annual_cos": np.cos(phase),
            "national_holiday": [int(day in calendar) for day in dates],
            "holiday_previous_day": [int(day - timedelta(days=1) in calendar) for day in dates],
            "holiday_two_days_ago": [int(day - timedelta(days=2) in calendar) for day in dates],
            "holiday_tomorrow": [int(day + timedelta(days=1) in calendar) for day in dates],
            "epiphany": ((local.month == 1) & (local.day == 6)).astype(int),
            "christmas_or_new_year_eve": (
                (local.month == 12) & np.isin(local.day, [24, 31])
            ).astype(int),
            "year_end_break": (
                ((local.month == 12) & (local.day >= 24)) | ((local.month == 1) & (local.day <= 6))
            ).astype(int),
        },
        index=index,
        dtype=float,
    )
    for lag in LAGS:
        source_hours = index - pd.Timedelta(hours=lag)
        features[f"lag_{lag}"] = history.reindex(source_hours).to_numpy(dtype=float)
    source_hours = index - pd.Timedelta(hours=HORIZON)
    for window in (24, 168):
        trailing = history.rolling(window, min_periods=window).mean()
        features[f"mean_{window}_lag_168"] = trailing.reindex(source_hours).to_numpy(dtype=float)
    return features


def seasonal_baseline(history: pd.Series, index: pd.DatetimeIndex, lag: int) -> pd.Series:
    source_hours = index - pd.Timedelta(hours=lag)
    if lag < HORIZON or source_hours.max() > history.index.max():
        raise ValueError("Baseline must use only available historical hours.")
    prediction = pd.Series(history.reindex(source_hours).to_numpy(), index=index, dtype=float)
    if prediction.isna().any():
        raise ValueError("Insufficient history for seasonal baseline.")
    return prediction


def forecast_week(history: pd.Series, origin: pd.Timestamp, config: ModelConfig) -> ForecastResult:
    index = target_hours(origin)
    validate_series(history)
    if history.index.max() >= index[0]:
        raise ValueError("All training data must be before the forecast origin.")
    if history.index[-1] != index[0] - pd.Timedelta(hours=1):
        raise ValueError("History must end exactly one hour before the forecast origin.")
    if len(history) < max(LAGS) + HORIZON:
        raise ValueError("Need at least one year plus one week of continuous history.")
    train_x: pd.DataFrame = build_features(history, pd.DatetimeIndex(history.index)).dropna()
    train_y = history.loc[train_x.index]
    future_x = build_features(history, index)
    if future_x.isna().any().any():
        raise ValueError("Forecast features contain unavailable historical loads.")
    model = HistGradientBoostingRegressor(**asdict(config), early_stopping=False)
    with threadpool_limits(limits=1):
        model.fit(train_x, train_y)
        values = model.predict(future_x)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Model produced invalid load forecasts.")
    return ForecastResult(
        predictions=pd.Series(values, index=index, name="forecast_mw"),
        train_start=train_x.index[0],
        train_end=train_x.index[-1],
        train_rows=len(train_x),
        iterations=int(model.n_iter_),
    )


def error_metrics(actual: pd.Series, forecast: pd.Series) -> ErrorMetrics:
    if actual.empty or not actual.index.equals(forecast.index):
        raise ValueError("Actual and forecast timestamps must align exactly.")
    observed = actual.to_numpy(dtype=float)
    predicted = forecast.to_numpy(dtype=float)
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all() or (observed <= 0).any():
        raise ValueError("Metrics need finite values and positive actual load.")
    error = predicted - observed
    absolute = np.abs(error)
    return ErrorMetrics(
        mae_mw=float(absolute.mean()),
        rmse_mw=float(np.sqrt(np.mean(error**2))),
        mape_percent=float(100 * np.mean(absolute / observed)),
        wape_percent=float(100 * absolute.sum() / observed.sum()),
        mean_error_mw=float(error.mean()),
    )


def select_model(history: pd.Series) -> tuple[ModelConfig, pd.DataFrame]:
    if history.index.max() >= FORECAST_START:
        raise ValueError("Model selection must not see any 2020 data.")
    records: list[dict[str, object]] = []
    for start in VALIDATION_STARTS:
        origin = pd.Timestamp(start, tz="UTC")
        index = target_hours(origin)
        if index[-1] >= FORECAST_START:
            raise ValueError("Validation windows must precede the final forecast.")
        train = history.loc[history.index < origin]
        actual = history.reindex(index)
        if actual.isna().any():
            raise ValueError(f"Incomplete validation week: {start}")
        for config in CANDIDATES:
            result = forecast_week(train, origin, config)
            score = error_metrics(actual, result.predictions)
            records.append({"origin_utc": str(origin), "model": config.name, **asdict(score)})
            print(f"Validation {start}: {config.name}, MAE {score.mae_mw:,.1f} MW", flush=True)
        for name, lag in (("previous_week", 168), ("previous_52_weeks", 8736)):
            score = error_metrics(actual, seasonal_baseline(train, index, lag))
            records.append({"origin_utc": str(origin), "model": name, **asdict(score)})
    validation = pd.DataFrame.from_records(records)
    scores = validation.groupby("model")["mae_mw"].mean()
    best = min(CANDIDATES, key=lambda config: float(scores[config.name]))
    return best, validation


def plot_forecast(frame: pd.DataFrame, metrics: ErrorMetrics, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    surface, ink, secondary, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
    actual_color, forecast_color = "#2a78d6", "#eb6834"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "text.color": ink}):
        fig, (load_ax, error_ax) = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.set_facecolor(surface)
        fig.subplots_adjust(left=0.09, right=0.86, top=0.78, bottom=0.19, hspace=0.14)
        fig.suptitle(
            "German electricity load | 1–7 January 2020", x=0.09, y=0.96, ha="left", fontsize=19
        )
        fig.text(
            0.09,
            0.895,
            f"168-hour forecast issued at 2020-01-01 00:00 UTC, using only earlier loads\n"
            f"MAE {metrics.mae_mw:,.0f} MW   |   RMSE {metrics.rmse_mw:,.0f} MW   |   MAPE {metrics.mape_percent:.2f}%",
            ha="left",
            va="top",
            color=secondary,
            linespacing=1.7,
        )
        for ax in (load_ax, error_ax):
            ax.set_facecolor(surface)
            ax.grid(axis="y", color=grid, linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(grid)
            ax.tick_params(colors=secondary, length=0, pad=7)
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        load_ax.plot(frame.index, frame["actual_mw"], color=actual_color, lw=1.6, label="Actual")
        load_ax.plot(
            frame.index,
            frame["forecast_mw"],
            color=forecast_color,
            lw=1.6,
            ls="--",
            label="Forecast",
        )
        load_ax.set_ylabel("Load (MW)", color=secondary)
        load_ax.legend(loc="upper left", ncol=2, frameon=False)
        last = frame.index[-1]
        # Leader lines preserve identity even when the two endpoint values converge.
        for column, label, offset in (
            ("actual_mw", "Actual", 14),
            ("forecast_mw", "Forecast", -14),
        ):
            load_ax.annotate(
                label,
                xy=(last, frame[column].iloc[-1]),
                xytext=(14, offset),
                textcoords="offset points",
                color=ink,
                va="center",
                arrowprops={"arrowstyle": "-", "color": secondary, "lw": 0.7},
                annotation_clip=False,
            )
        error_ax.axhline(0, color=secondary, lw=0.8)
        error_ax.plot(frame.index, frame["error_mw"], color=forecast_color, lw=1.4)
        error_ax.set_ylabel("Error (MW)", color=secondary)
        bound = max(1.0, float(frame["error_mw"].abs().max()) * 1.2)
        error_ax.set_ylim(-bound, bound)
        error_ax.xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))  # type: ignore[no-untyped-call]
        error_ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%a\n%d Jan", tz="UTC")  # type: ignore[no-untyped-call]
        )
        error_ax.set_xlim(frame.index[0], last)
        error_ax.set_xlabel("Hour beginning (UTC)", color=secondary, labelpad=10)
        fig.text(
            0.09,
            0.035,
            "Error = forecast − actual. Source: supplied OPSD / ENTSO-E load series. Full hourly table: forecast.csv",
            color=secondary,
            fontsize=9,
        )
        fig.savefig(output_dir / "forecast_vs_actual.png", dpi=180, facecolor=surface)
        fig.savefig(output_dir / "forecast_vs_actual.svg", facecolor=surface)
        plt.close(fig)


def run(data_path: Path, output_dir: Path) -> None:
    data = read_load(data_path)
    history, actual = split_data(data)
    config, validation = select_model(history)
    print(f"Selected before target scoring: {config.name}", flush=True)
    result = forecast_week(history, FORECAST_START, config)
    index = pd.DatetimeIndex(result.predictions.index)
    frame = pd.DataFrame(
        {
            "actual_mw": actual,
            "forecast_mw": result.predictions,
            "previous_week_mw": seasonal_baseline(history, index, 168),
            "previous_52_weeks_mw": seasonal_baseline(history, index, 8736),
        },
        index=index,
    )
    frame["error_mw"] = frame["forecast_mw"] - frame["actual_mw"]
    frame["absolute_error_mw"] = frame["error_mw"].abs()
    frame["absolute_percentage_error"] = 100 * frame["absolute_error_mw"] / frame["actual_mw"]
    scores = {
        name: error_metrics(actual, frame[column])
        for name, column in (
            ("forecast", "forecast_mw"),
            ("previous_week", "previous_week_mw"),
            ("previous_52_weeks", "previous_52_weeks_mw"),
        )
    }
    mean_validation = validation.groupby("model")["mae_mw"].mean()
    report: dict[str, object] = {
        "input_file": data_path.name,
        "input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "source": "Supplied OPSD CSV, DE_load_actual_entsoe_transparency; load in MW",
        "timestamp_convention": "Hour beginning, UTC; calendar features in Europe/Berlin",
        "forecast_start_utc": str(index[0]),
        "forecast_end_inclusive_utc": str(index[-1]),
        "forecast_hours": len(index),
        "history_start_utc": str(history.index[0]),
        "history_end_utc": str(history.index[-1]),
        "history_rows": len(history),
        "training_label_start_utc": str(result.train_start),
        "training_label_end_utc": str(result.train_end),
        "training_rows": result.train_rows,
        "lag_warmup_rows": len(history) - result.train_rows,
        "ignored_rows_after_forecast_week": int((data.index > index[-1]).sum()),
        "imputed_loads": 0,
        "model": config.name,
        "parameters": {**asdict(config), "early_stopping": False},
        "fitted_iterations": result.iterations,
        "selection": "Minimum mean MAE over five pre-2020 fixed-origin weeks; two HGB candidates only",
        "validation_mean_mae_mw": {
            str(name): float(value) for name, value in mean_validation.items()
        },
        "metrics": {name: asdict(score) for name, score in scores.items()},
        "mae_improvement_percent": {
            name: 100 * (1 - scores["forecast"].mae_mw / scores[name].mae_mw)
            for name in ("previous_week", "previous_52_weeks")
        },
        "error_sign": "forecast minus actual; negative means underprediction",
        "metric_definitions": {
            "mae_mw": "mean(abs(forecast - actual))",
            "rmse_mw": "sqrt(mean((forecast - actual)**2))",
            "mape_percent": "100 * mean(abs(forecast - actual) / actual)",
            "wape_percent": "100 * sum(abs(forecast - actual)) / sum(actual)",
        },
        "limitations": [
            "One retrospective week, not an estimate of typical year-round performance.",
            "No weather, economic predictors, regional load weights, or uncertainty intervals.",
            "National holidays plus an Epiphany indicator; regional effects and bridge days are approximate.",
            "Previous-week baseline copies Christmas week; 52-week baseline aligns weekdays, not holidays.",
            "Fixed-hour annual lags need not align local hours across DST or calendar dates across leap years.",
            "The supplied historical series may include revisions; publication-time data availability is not known.",
        ],
        "versions": {
            package: version(package)
            for package in ("numpy", "pandas", "scikit-learn", "matplotlib", "holidays")
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "forecast.csv", index_label="utc_timestamp")
    validation.to_csv(output_dir / "validation.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    daily_records: list[dict[str, object]] = []
    for day, group in frame.groupby(index.date):
        score = error_metrics(group["actual_mw"], group["forecast_mw"])
        daily_records.append({"date_utc": str(day), "hours": len(group), **asdict(score)})
    pd.DataFrame.from_records(daily_records).to_csv(output_dir / "daily_metrics.csv", index=False)
    plot_forecast(frame, scores["forecast"], output_dir)
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved forecast, plot and metrics to {output_dir.resolve()}")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "opsd_de_load.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    args = parser.parse_args()
    try:
        run(args.data, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Forecast failed: {exc}\n")


if __name__ == "__main__":
    main()
