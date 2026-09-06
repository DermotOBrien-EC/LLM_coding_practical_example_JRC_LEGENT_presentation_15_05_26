"""Fixed-origin German hourly load forecast with a strictly held-out test week."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import timedelta, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

from holidays.countries.germany import Germany
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

FloatArray = NDArray[np.float64]
HORIZON_HOURS = 168
FORECAST_START = "2020-01-01"
LOCAL_TIMEZONE = "Europe/Berlin"
LAGS = (168, 336, 8736, 8760)
WARMUP_HOURS = max(LAGS)
VALIDATION_STARTS = ("2018-01-01", "2019-01-01", "2019-11-04", "2019-12-02", "2019-12-23")
TIMESTAMP_COLUMN = "utc_timestamp"
LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelSpec:
    name: str
    use_lags: bool
    max_iter: int = 250
    learning_rate: float = 0.06
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 30
    l2_regularization: float = 10.0


MODEL_SPECS = (
    ModelSpec("calendar", use_lags=False),
    ModelSpec("calendar_lags", use_lags=True),
)


@dataclass(frozen=True)
class Scores:
    mae_mw: float
    rmse_mw: float
    mape_pct: float
    wape_pct: float
    bias_mw: float


@dataclass(frozen=True)
class FitResult:
    forecast: pd.Series
    training_rows: int
    training_start: pd.Timestamp
    training_end: pd.Timestamp


@dataclass(frozen=True)
class ForecastRun:
    forecast: pd.Series
    selected_model: str
    spec: ModelSpec
    validation: pd.DataFrame
    training_rows: int
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    available_history_rows: int


def forecast_index(start: str = FORECAST_START) -> pd.DatetimeIndex:
    return pd.date_range(pd.to_datetime(start, utc=True), periods=HORIZON_HOURS, freq="h")


def check_hourly(series: pd.Series) -> None:
    index = series.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None or str(index.tz) != "UTC":
        raise ValueError("Load must have a timezone-aware UTC DatetimeIndex.")
    if len(index) == 0 or index.hasnans or index.has_duplicates:
        raise ValueError("Timestamps must be nonempty, valid and unique.")
    expected = pd.date_range(index.min(), index.max(), freq="h")
    if not index.equals(expected) or not index.equals(index.floor("h")):
        raise ValueError("Load must be sorted, on the hour, with no missing hours.")
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Load must contain only finite, strictly positive MW values.")


def load_data(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if set(frame.columns) != {TIMESTAMP_COLUMN, LOAD_COLUMN}:
        raise ValueError(f"Expected exactly {TIMESTAMP_COLUMN!r} and {LOAD_COLUMN!r} columns.")
    index = pd.DatetimeIndex(pd.to_datetime(frame[TIMESTAMP_COLUMN], utc=True, errors="raise"))
    values = pd.to_numeric(frame[LOAD_COLUMN], errors="raise").to_numpy(dtype=float)
    series = pd.Series(values, index=index, name="load_mw").sort_index()
    check_hourly(series)
    return series


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert(LOCAL_TIMEZONE)
    dates = local.date
    national = Germany(years=range(local.year.min() - 1, local.year.max() + 2))
    holiday_flags = {
        day: (
            float(day in national),
            float(day + timedelta(days=1) in national),
            float(day - timedelta(days=1) in national),
        )
        for day in set(dates)
    }
    flags = np.array([holiday_flags[day] for day in dates])
    annual_phase = 2 * np.pi * (local.dayofyear.to_numpy() - 1) / 365.25
    # A continuous position across year-end lets trees learn industrial shutdown patterns.
    shutdown_day = np.where(
        (local.month == 12) & (local.day >= 20),
        local.day - 20,
        np.where((local.month == 1) & (local.day <= 7), local.day + 11, -1),
    )
    return pd.DataFrame(
        {
            "hour": local.hour,
            "day_of_week": local.dayofweek,
            "hour_of_week": local.dayofweek * 24 + local.hour,
            "month": local.month,
            "day_of_month": local.day,
            "day_of_year": local.dayofyear,
            "annual_sin": np.sin(annual_phase),
            "annual_cos": np.cos(annual_phase),
            "trend_days": (index - pd.Timestamp("2015-01-01", tz="UTC")).total_seconds() / 86400,
            "national_holiday": flags[:, 0],
            "day_before_holiday": flags[:, 1],
            "day_after_holiday": flags[:, 2],
            "shutdown_day": shutdown_day,
            "epiphany": ((local.month == 1) & (local.day == 6)).astype(int),
        },
        index=index,
        dtype=float,
    )


def lag_features(history: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            f"lag_{lag}h": history.reindex(index - pd.Timedelta(hours=lag)).to_numpy(dtype=float)
            for lag in LAGS
        },
        index=index,
    )


def fit_forecast(history: pd.Series, future: pd.DatetimeIndex, spec: ModelSpec) -> FitResult:
    check_hourly(history)
    if future.tz is None or str(future.tz) != "UTC" or len(future) != HORIZON_HOURS:
        raise ValueError("Forecast timestamps must be exactly 168 UTC hours.")
    if not future.equals(pd.date_range(future[0], periods=HORIZON_HOURS, freq="h")):
        raise ValueError("Forecast timestamps must be consecutive hours.")
    if history.index[-1] >= future[0]:
        raise ValueError("All history must be strictly before the forecast origin.")
    if history.index[-1] + pd.Timedelta(hours=1) != future[0]:
        raise ValueError("History must end immediately before the forecast origin.")
    if len(history) < WARMUP_HOURS + HORIZON_HOURS:
        raise ValueError("Need at least 365 days plus one week of complete history.")

    # Match fitting rows across variants rather than giving calendar-only a longer history.
    training_index = history.index[WARMUP_HOURS:]
    train_x = calendar_features(training_index)
    future_x = calendar_features(future)
    if spec.use_lags:
        train_x = train_x.join(lag_features(history, training_index))
        future_x = future_x.join(lag_features(history, future))
    if train_x.isna().any().any() or future_x.isna().any().any():
        raise ValueError("All model inputs must be available before the forecast origin.")
    model = HistGradientBoostingRegressor(
        max_iter=spec.max_iter,
        learning_rate=spec.learning_rate,
        max_leaf_nodes=spec.max_leaf_nodes,
        min_samples_leaf=spec.min_samples_leaf,
        l2_regularization=spec.l2_regularization,
        loss="squared_error",
        early_stopping=False,
        random_state=2020,
        categorical_features=None,
    )
    with threadpool_limits(limits=2):
        model.fit(train_x, history.loc[training_index].to_numpy(dtype=float))
        values = np.asarray(model.predict(future_x), dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Model produced invalid load predictions.")
    return FitResult(
        pd.Series(values, index=future, name="forecast_mw"),
        len(training_index),
        training_index[0],
        training_index[-1],
    )


def score(actual: FloatArray, predicted: FloatArray) -> Scores:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Metrics require nonempty, equally shaped one-dimensional arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all() or (actual <= 0).any():
        raise ValueError("Metrics require finite values and positive actual load.")
    error = predicted - actual
    return Scores(
        mae_mw=float(np.abs(error).mean()),
        rmse_mw=float(np.sqrt(np.square(error).mean())),
        mape_pct=float(100 * (np.abs(error) / actual).mean()),
        wape_pct=float(100 * np.abs(error).sum() / np.abs(actual).sum()),
        bias_mw=float(error.mean()),
    )


def select_and_forecast(
    data: pd.Series,
    specs: tuple[ModelSpec, ...] = MODEL_SPECS,
    validation_starts: tuple[str, ...] = VALIDATION_STARTS,
) -> ForecastRun:
    future = forecast_index()
    history = data.loc[data.index < future[0]].copy()
    check_hourly(history)
    if not specs or not validation_starts or len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Need uniquely named models and at least one validation week.")
    rows: list[dict[str, str | int | float]] = []
    for start in validation_starts:
        valid_index = forecast_index(start)
        if valid_index[-1] >= future[0]:
            raise ValueError("Every validation week must end before the test week.")
        past = history.loc[history.index < valid_index[0]]
        actual = history.reindex(valid_index).to_numpy(dtype=float)
        baseline = past.reindex(valid_index - pd.Timedelta(hours=168)).to_numpy(dtype=float)
        baseline_scores = score(actual, baseline)
        for spec in specs:
            fitted = fit_forecast(past, valid_index, spec)
            metrics = score(actual, fitted.forecast.to_numpy(dtype=float))
            rows.append(
                {
                    "origin": str(valid_index[0]),
                    "model": spec.name,
                    "training_rows": fitted.training_rows,
                    **asdict(metrics),
                    "baseline_mae_mw": baseline_scores.mae_mw,
                    "baseline_mape_pct": baseline_scores.mape_pct,
                }
            )
    validation = pd.DataFrame(rows)
    mean_mae = validation.groupby("model", sort=False)["mae_mw"].mean()
    selected = str(mean_mae.sort_values(kind="stable").index[0])
    selected_spec = next(spec for spec in specs if spec.name == selected)
    final = fit_forecast(history, future, selected_spec)
    return ForecastRun(
        final.forecast,
        selected,
        selected_spec,
        validation,
        final.training_rows,
        final.training_start,
        final.training_end,
        len(history),
    )


def save_plot(table: pd.DataFrame, metrics: Scores, model_name: str, output: Path) -> None:
    surface, ink, secondary = "#fcfcfb", "#0b0b0b", "#52514e"
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "text.color": ink}):
        fig, ax = plt.subplots(figsize=(13, 5.8), facecolor=surface)
        ax.set_facecolor(surface)
        for column, label, color, style in (
            ("actual_mw", "Actual", "#2a78d6", "-"),
            ("forecast_mw", "Forecast", "#eb6834", "--"),
        ):
            ax.plot(table.index, table[column], label=label, color=color, linestyle=style, lw=1.5)
        ax.set_title("German electricity load | 1–7 January 2020", loc="left", pad=42, fontsize=17)
        ax.text(
            0,
            1.045,
            f"168-hour backtest, cutoff: 1 Jan 2020 UTC  |  {model_name}  |  "
            f"MAE {metrics.mae_mw:,.0f} MW  |  MAPE {metrics.mape_pct:.2f}%",
            transform=ax.transAxes,
            color=secondary,
            fontsize=10,
        )
        ax.set_ylabel("Electricity load (MW)", color=secondary)
        ax.set_xlabel("Hour (UTC)", color=secondary, labelpad=10)
        ax.xaxis.set_major_locator(mdates.DayLocator(tz=timezone.utc))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d Jan", tz=timezone.utc))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color("#c3c2b7")
        ax.tick_params(colors=secondary, length=0, pad=7)
        ax.legend(loc="upper left", frameon=False, ncol=2, labelcolor=ink)
        endpoint_actual = float(table["actual_mw"].iloc[-1])
        endpoint_forecast = float(table["forecast_mw"].iloc[-1])
        close = abs(endpoint_actual - endpoint_forecast) < 0.04 * float(
            table[["actual_mw", "forecast_mw"]].max().max()
            - table[["actual_mw", "forecast_mw"]].min().min()
        )
        for label, value, offset in (
            ("Actual", endpoint_actual, 12 if close else 0),
            ("Forecast", endpoint_forecast, -12 if close else 0),
        ):
            ax.annotate(
                label,
                xy=(table.index[-1], value),
                xytext=(12, offset),
                textcoords="offset points",
                va="center",
                color=ink,
                fontsize=10,
                arrowprops={"arrowstyle": "-", "color": "#898781", "lw": 0.6},
            )
        ax.set_xlim(table.index[0], table.index[-1] + pd.Timedelta(hours=13))
        fig.text(
            0.085,
            0.015,
            "Source: supplied OPSD/ENTSO-E data. Training uses pre-2020 values only. "
            "Full hourly values: forecast.csv. Y-axis is not zero-based.",
            fontsize=9,
            color=secondary,
        )
        fig.subplots_adjust(left=0.085, right=0.97, bottom=0.19, top=0.79)
        fig.savefig(output / "forecast.png", dpi=160, facecolor=surface)
        fig.savefig(output / "forecast.svg", facecolor=surface)
        plt.close(fig)


def run(input_path: Path, output: Path) -> dict[str, object]:
    data = load_data(input_path)
    result = select_and_forecast(data)
    future = result.forecast.index
    # The untouched test outcomes enter the computation only after model selection and refitting.
    actual = data.reindex(future).to_numpy(dtype=float)
    predicted = result.forecast.to_numpy(dtype=float)
    history = data.loc[data.index < future[0]]
    baseline = history.reindex(future - pd.Timedelta(hours=168)).to_numpy(dtype=float)
    model_scores, baseline_scores = score(actual, predicted), score(actual, baseline)
    table = pd.DataFrame(
        {
            "actual_mw": actual,
            "forecast_mw": predicted,
            "baseline_previous_week_mw": baseline,
            "error_mw": predicted - actual,
            "absolute_error_mw": np.abs(predicted - actual),
            "absolute_percentage_error_pct": 100 * np.abs(predicted - actual) / actual,
        },
        index=future,
    )
    table.index.name = TIMESTAMP_COLUMN
    daily = [
        {
            "date_utc": str(day.date()),
            "hours": len(group),
            **asdict(score(group.actual_mw.to_numpy(), group.forecast_mw.to_numpy())),
        }
        for day, group in table.groupby(table.index.normalize())
    ]
    report: dict[str, object] = {
        "forecast_start_utc": str(future[0]),
        "forecast_end_inclusive_utc": str(future[-1]),
        "hours": len(future),
        "selected_model": result.selected_model,
        "model_class": "sklearn.ensemble.HistGradientBoostingRegressor",
        "model_spec": asdict(result.spec),
        "other_model_parameters": {
            "early_stopping": False,
            "random_state": 2020,
            "loss": "squared_error",
            "categorical_features": None,
            "thread_limit": 2,
        },
        "calendar_timezone": LOCAL_TIMEZONE,
        "lag_hours": list(LAGS) if result.spec.use_lags else [],
        "available_history_rows": result.available_history_rows,
        "history_start_utc": str(history.index[0]),
        "history_end_utc": str(history.index[-1]),
        "fitted_rows": result.training_rows,
        "fitted_start_utc": str(result.training_start),
        "fitted_end_utc": str(result.training_end),
        "common_warmup_hours": WARMUP_HOURS,
        "model_metrics": asdict(model_scores),
        "baseline_metrics": asdict(baseline_scores),
        "baseline_definition": "Same UTC hour 168 hours earlier; forecast fixed before test week.",
        "mae_improvement_over_baseline_pct": 100
        * (1 - model_scores.mae_mw / baseline_scores.mae_mw)
        if baseline_scores.mae_mw > 0
        else None,
        "selection_metric": "Mean MAE across five equally weighted pre-2020 168-hour weeks",
        "validation_mean_mae_mw": result.validation.groupby("model")["mae_mw"].mean().to_dict(),
        "validation": result.validation.to_dict(orient="records"),
        "daily_metrics_utc": daily,
        "bias_definition": "Mean forecast minus actual; positive means overforecast.",
        "data_audit": {
            "input_file": str(input_path.resolve()),
            "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "rows": len(data),
            "start_utc": str(data.index[0]),
            "end_utc": str(data.index[-1]),
            "missing_hours": 0,
            "missing_or_invalid_load_values": 0,
            "duplicate_timestamps": 0,
            "imputation": "None; invalid or missing data raises an error.",
        },
        "versions": {
            "python": platform.python_version(),
            **{
                package: version(package)
                for package in ("numpy", "pandas", "scikit-learn", "matplotlib", "holidays")
            },
        },
        "limitations": [
            "Single held-out week; not an estimate of year-round or future performance.",
            "Retrospective backtest on supplied actuals, not a vintage-aware operational replay.",
            "No weather forecasts, industrial schedules or predictive intervals.",
            "National holidays plus Jan 6 flag; other regional holidays are not explicitly modeled.",
            "365-day lag matches calendar date for this test week, not universally across leap years.",
            "The previous-week baseline is weak here because its source week includes Christmas.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "forecast.csv", float_format="%.10f")
    result.validation.to_csv(output / "validation.csv", index=False, float_format="%.10f")
    pd.DataFrame(daily).to_csv(output / "daily_metrics.csv", index=False, float_format="%.10f")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save_plot(table, model_scores, result.selected_model, output)
    print(f"Model: {result.selected_model}; fitted on {result.training_rows:,} pre-2020 hours")
    print(f"Forecast: {future[0]} through {future[-1]} ({len(future)} hours)")
    print(f"MAE: {model_scores.mae_mw:,.2f} MW | RMSE: {model_scores.rmse_mw:,.2f} MW")
    print(f"MAPE: {model_scores.mape_pct:.2f}% | WAPE: {model_scores.wape_pct:.2f}%")
    print(f"Bias (forecast - actual): {model_scores.bias_mw:+,.2f} MW")
    print(f"Previous-week baseline MAE: {baseline_scores.mae_mw:,.2f} MW")
    print(f"Outputs: {output.resolve()}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE_DIR / "opsd_de_load.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs")
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
