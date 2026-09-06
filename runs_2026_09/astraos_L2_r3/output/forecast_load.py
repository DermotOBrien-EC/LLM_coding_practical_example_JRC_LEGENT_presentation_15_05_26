#!/usr/bin/env python3
"""Forecast Germany's first 168 UTC hours of 2020 without future-load leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

from holidays.countries.germany import Germany
import matplotlib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

FORECAST_START = pd.Timestamp("2020-01-01T00:00:00Z")
HORIZON = 168
WARMUP_HOURS = 4 * HORIZON
LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
LAGS = (168, 336, 504, 672, 8736, 8760)
VALIDATION_ORIGINS = (
    "2018-01-01",
    "2019-01-01",
    "2019-04-15",
    "2019-07-01",
    "2019-10-01",
    "2019-12-25",
)
BASELINES = {"previous_week": 168, "previous_52_weeks": 8736, "previous_365_days": 8760}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    use_lags: bool
    max_leaf_nodes: int
    max_iter: int = 300


CANDIDATES = (
    ModelSpec("calendar_only", use_lags=False, max_leaf_nodes=31),
    ModelSpec("calendar_lags_small", use_lags=True, max_leaf_nodes=15),
    ModelSpec("calendar_lags", use_lags=True, max_leaf_nodes=31),
)


class Scores(TypedDict):
    mae_mw: float
    rmse_mw: float
    mape_percent: float
    wape_percent: float
    bias_mw: float


def forecast_index(origin: pd.Timestamp) -> pd.DatetimeIndex:
    if origin.tzinfo is None:
        raise ValueError("Forecast origin must be timezone-aware.")
    return pd.date_range(origin.tz_convert("UTC"), periods=HORIZON, freq="h")


def load_series(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if set(frame.columns) != {"utc_timestamp", LOAD_COLUMN}:
        raise ValueError(f"Expected utc_timestamp and {LOAD_COLUMN} columns.")
    index = pd.DatetimeIndex(pd.to_datetime(frame["utc_timestamp"], utc=True, errors="raise"))
    values = pd.to_numeric(frame[LOAD_COLUMN], errors="raise").to_numpy(dtype=float)
    if len(index) == 0 or index.hasnans or index.has_duplicates:
        raise ValueError("Data must contain unique, valid hourly timestamps.")
    series = pd.Series(values, index=index, name="load_mw").sort_index()
    expected = pd.date_range(series.index[0], series.index[-1], freq="h")
    if not series.index.equals(expected):
        raise ValueError("Data must be a complete hourly series; missing hours are not imputed.")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("All load values must be finite and strictly positive.")
    return series


def build_features(index: pd.DatetimeIndex, history: pd.Series, *, use_lags: bool) -> pd.DataFrame:
    local = index.tz_convert("Europe/Berlin")
    years = list(range(int(local.year.min()) - 2, int(local.year.max()) + 2))
    calendar = Germany(years=years)
    dates = local.date
    national = np.array([int(day in calendar) for day in dates])
    features = pd.DataFrame(index=index)
    features["hour"] = local.hour
    features["day_of_week"] = local.dayofweek
    features["month"] = local.month
    features["day_of_month"] = local.day
    features["day_of_year"] = local.dayofyear
    features["year"] = local.year
    features["national_holiday"] = national
    features["workday"] = ((local.dayofweek < 5) & (national == 0)).astype(int)
    features["holiday_yesterday"] = [int(day - timedelta(days=1) in calendar) for day in dates]
    features["holiday_tomorrow"] = [int(day + timedelta(days=1) in calendar) for day in dates]
    # These dates affect industry even when they are not nationwide public holidays.
    features["christmas_eve"] = ((local.month == 12) & (local.day == 24)).astype(int)
    features["new_year_eve"] = ((local.month == 12) & (local.day == 31)).astype(int)
    features["epiphany"] = ((local.month == 1) & (local.day == 6)).astype(int)
    features["year_end_break"] = (
        ((local.month == 12) & (local.day >= 24)) | ((local.month == 1) & (local.day <= 6))
    ).astype(int)
    for name, phase in (
        ("hour", local.hour.to_numpy() / 24),
        ("week", (local.dayofweek.to_numpy() * 24 + local.hour.to_numpy()) / 168),
        ("annual", (local.dayofyear.to_numpy() - 1) / 365.2425),
    ):
        features[f"{name}_sin"] = np.sin(2 * np.pi * phase)
        features[f"{name}_cos"] = np.cos(2 * np.pi * phase)
    if use_lags:
        # Every lag is at least the full horizon: even hour 168 uses pre-origin load.
        for lag in LAGS:
            lag_index = index - pd.Timedelta(hours=lag)
            features[f"load_lag_{lag}h"] = history.reindex(lag_index).to_numpy(dtype=float)
            lag_local = lag_index.tz_convert("Europe/Berlin")
            features[f"lag_{lag}h_holiday"] = [int(day in calendar) for day in lag_local.date]
        weekly = [f"load_lag_{lag}h" for lag in LAGS[:4]]
        features["recent_same_hour_mean"] = features[weekly].mean(axis=1)
    return features.astype(float)


def fit_predict(data: pd.Series, origin: pd.Timestamp, spec: ModelSpec) -> NDArray[np.float64]:
    history = data.loc[data.index < origin]
    if len(history) < WARMUP_HOURS + HORIZON:
        raise ValueError("At least five weeks of pre-origin history are required.")
    expected = pd.date_range(history.index[0], origin - pd.Timedelta(hours=1), freq="h")
    if not history.index.equals(expected) or not np.isfinite(history.to_numpy()).all():
        raise ValueError("Training history must be complete up to the forecast origin.")
    target_index = forecast_index(origin)
    training_index = history.index[WARMUP_HOURS:]
    x_train = build_features(training_index, history, use_lags=spec.use_lags)
    x_future = build_features(target_index, history, use_lags=spec.use_lags)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=spec.max_iter,
        max_leaf_nodes=spec.max_leaf_nodes,
        min_samples_leaf=30,
        l2_regularization=10.0,
        early_stopping=False,
        random_state=42,
    )
    # Limit native parallelism for reproducible, efficient runs on large-core machines.
    with threadpool_limits(limits=2):
        model.fit(x_train, history.loc[training_index].to_numpy(dtype=float))
        prediction = np.asarray(model.predict(x_future), dtype=np.float64)
    if prediction.shape != (HORIZON,) or not np.isfinite(prediction).all():
        raise ValueError("Model did not produce 168 finite predictions.")
    return prediction


def score(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> Scores:
    if (
        actual.ndim != 1
        or actual.shape != predicted.shape
        or actual.size == 0
        or not np.isfinite(actual).all()
        or not np.isfinite(predicted).all()
        or (actual <= 0).any()
    ):
        raise ValueError("Metrics require aligned, finite vectors and strictly positive actuals.")
    error = predicted - actual
    return Scores(
        mae_mw=float(np.mean(np.abs(error))),
        rmse_mw=float(np.sqrt(np.mean(error**2))),
        mape_percent=float(100 * np.mean(np.abs(error) / actual)),
        wape_percent=float(100 * np.sum(np.abs(error)) / np.sum(actual)),
        bias_mw=float(np.mean(error)),
    )


def baseline_predict(history: pd.Series, index: pd.DatetimeIndex, lag: int) -> NDArray[np.float64]:
    return np.asarray(history.reindex(index - pd.Timedelta(hours=lag)), dtype=np.float64)


def select_model(
    history: pd.Series, *, origins: list[pd.Timestamp] | None = None
) -> tuple[ModelSpec, pd.DataFrame]:
    chosen_origins = (
        [pd.Timestamp(value, tz="UTC") for value in VALIDATION_ORIGINS]
        if origins is None
        else origins
    )
    if not chosen_origins or any(
        origin + pd.Timedelta(hours=HORIZON) > FORECAST_START for origin in chosen_origins
    ):
        raise ValueError("Every validation week must finish before the target week.")
    rows: list[dict[str, object]] = []
    for origin in chosen_origins:
        index = forecast_index(origin)
        actual = history.reindex(index).to_numpy(dtype=float)
        for spec in CANDIDATES:
            predicted = fit_predict(history, origin, spec)
            metrics = score(actual, predicted)
            rows.append({"origin_utc": origin.isoformat(), "model": spec.name, **metrics})
            print(
                f"Validation {origin.date()} {spec.name}: RMSE {metrics['rmse_mw']:,.1f} MW",
                flush=True,
            )
        available = history.loc[history.index < origin]
        for name, lag in BASELINES.items():
            metrics = score(actual, baseline_predict(available, index, lag))
            rows.append({"origin_utc": origin.isoformat(), "model": name, **metrics})
    results = pd.DataFrame(rows)
    means = results.groupby("model")["rmse_mw"].mean()
    winner = min(CANDIDATES, key=lambda spec: float(means.loc[spec.name]))
    return winner, results


def plot_forecast(frame: pd.DataFrame, metrics: Scores, output: Path) -> None:
    surface, ink, secondary = "#fcfcfb", "#0b0b0b", "#52514e"
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": ink,
            "axes.labelcolor": secondary,
            "xtick.color": secondary,
            "ytick.color": secondary,
        }
    ):
        fig, ax = plt.subplots(figsize=(13, 5.8), facecolor=surface)
        ax.set_facecolor(surface)
        series = (
            ("actual_mw", "Actual", "#2a78d6", "-"),
            ("forecast_mw", "Forecast", "#eb6834", "--"),
        )
        for column, label, color, linestyle in series:
            ax.plot(
                frame.index,
                frame[column],
                label=label,
                color=color,
                lw=1.5,
                linestyle=linestyle,
                solid_capstyle="round",
            )
        ax.set_title(
            "German hourly electricity load | 1–7 January 2020",
            loc="left",
            fontsize=17,
            fontweight="bold",
            pad=46,
        )
        ax.text(
            0,
            1.07,
            f"168-hour fixed-origin backtest | MAE {metrics['mae_mw']:,.0f} MW"
            f" | RMSE {metrics['rmse_mw']:,.0f} MW | MAPE {metrics['mape_percent']:.2f}%",
            transform=ax.transAxes,
            color=secondary,
            fontsize=11,
        )
        ax.set_ylabel("Electricity load (MW)")
        ax.set_xlabel("Date (UTC)", labelpad=12)
        ax.xaxis.set_major_locator(mdates.DayLocator(tz=timezone.utc))  # type: ignore[no-untyped-call]
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%d %b", tz=timezone.utc)  # type: ignore[no-untyped-call]
        )
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, pad=8)
        ax.legend(loc="upper left", ncols=2, frameon=False, labelcolor=ink)
        last = frame.index[-1]
        end_values = [float(frame[column].iloc[-1]) for column, _, _, _ in series]
        separated = abs(end_values[0] - end_values[1]) >= 1800
        for number, (column, label, color, _) in enumerate(series):
            value = float(frame[column].iloc[-1])
            offset = (
                0
                if separated
                else (
                    14
                    if value == max(end_values) and number == end_values.index(max(end_values))
                    else -14
                )
            )
            ax.annotate(
                label,
                xy=(last, value),
                xytext=(12, offset),
                textcoords="offset points",
                va="center",
                color=ink,
                arrowprops={"arrowstyle": "-", "color": "#898781", "lw": 0.6},
            )
            ax.plot(
                last,
                value,
                "o",
                color=color,
                markersize=6,
                markeredgecolor=surface,
                markeredgewidth=1.5,
            )
        ax.set_xlim(frame.index[0], last + pd.Timedelta(hours=16))
        fig.text(
            0.09,
            0.025,
            "Source: supplied Open Power System Data series. UTC horizon; German-local calendar features.\n"
            "Solid = observed load; dashed = forecast. Full hourly values are in forecast.csv.",
            color=secondary,
            fontsize=9,
        )
        fig.subplots_adjust(left=0.09, right=0.97, bottom=0.20, top=0.78)
        fig.savefig(output / "forecast.png", dpi=160, facecolor=surface)
        fig.savefig(output / "forecast.svg", facecolor=surface)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--data", type=Path, default=here / "opsd_de_load.csv")
    parser.add_argument("--output-dir", type=Path, default=here / "outputs")
    args = parser.parse_args()
    data = load_series(args.data)
    history = data.loc[data.index < FORECAST_START]
    index = forecast_index(FORECAST_START)
    winner, validation = select_model(history)
    print(f"Selected {winner.name} using mean validation RMSE. Training final model...", flush=True)
    predicted = fit_predict(history, FORECAST_START, winner)
    # Actual target loads are first consulted after selection and prediction are complete.
    actual = data.reindex(index).to_numpy(dtype=float)
    scores: dict[str, Scores] = {winner.name: score(actual, predicted)}
    frame = pd.DataFrame({"actual_mw": actual, "forecast_mw": predicted}, index=index)
    frame.index.name = "utc_timestamp"
    for name, lag in BASELINES.items():
        baseline = baseline_predict(history, index, lag)
        scores[name] = score(actual, baseline)
        frame[f"{name}_mw"] = baseline
    frame["error_mw"] = predicted - actual
    frame["absolute_error_mw"] = np.abs(predicted - actual)
    frame["absolute_percentage_error"] = 100 * np.abs(predicted - actual) / actual
    output: Path = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "forecast.csv", float_format="%.10f")
    validation.to_csv(output / "validation_scores.csv", index=False, float_format="%.10f")
    daily = pd.DataFrame(
        [
            {
                "date_utc": str(day.date()),
                **score(group["actual_mw"].to_numpy(), group["forecast_mw"].to_numpy()),
            }
            for day, group in frame.groupby(frame.index.normalize())
        ]
    )
    daily.to_csv(output / "daily_scores.csv", index=False)
    metadata = {
        "input_file": str(args.data.resolve()),
        "input_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "forecast_start_utc": index[0].isoformat(),
        "forecast_end_utc_inclusive": index[-1].isoformat(),
        "forecast_hours": len(index),
        "historical_start_utc": history.index[0].isoformat(),
        "historical_end_utc": history.index[-1].isoformat(),
        "historical_rows": len(history),
        "fitted_rows": len(history) - WARMUP_HOURS,
        "fitted_start_utc": history.index[WARMUP_HOURS].isoformat(),
        "warmup_hours": WARMUP_HOURS,
        "selected_model": asdict(winner),
        "selection_rule": "Lowest equally weighted mean RMSE across six pre-2020 168-hour folds; trained models only.",
        "validation_origins_utc": list(VALIDATION_ORIGINS),
        "validation_mean_rmse_mw": validation.groupby("model")["rmse_mw"].mean().to_dict(),
        "test_scores": scores,
        "dependencies": {
            name: version(name)
            for name in (
                "numpy",
                "pandas",
                "matplotlib",
                "scikit-learn",
                "holidays",
                "threadpoolctl",
            )
        },
    }
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    plot_forecast(frame, scores[winner.name], output)
    metrics_table = pd.DataFrame(scores).T
    metrics_table.index.name = "method"
    metrics_table.to_csv(output / "accuracy.csv")
    lines = [
        "# German hourly electricity load forecast",
        "",
        f"Selected model: **{winner.name}**, a histogram gradient-boosted tree regressor.",
        "",
        f"Forecast: {index[0].isoformat()} through {index[-1].isoformat()}, inclusive (168 hours).",
        f"Historical observations: {history.index[0].isoformat()} through {history.index[-1].isoformat()} ({len(history):,} rows).",
        f"The first {WARMUP_HOURS} hours provide lag warmup, leaving {len(history) - WARMUP_HOURS:,} fitted rows.",
        "",
        "## Holdout accuracy",
        "",
        "| Method | MAE (MW) | RMSE (MW) | MAPE (%) | WAPE (%) | Bias (MW) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in scores.items():
        lines.append(
            f"| {name} | {values['mae_mw']:,.1f} | {values['rmse_mw']:,.1f} | "
            f"{values['mape_percent']:.2f} | {values['wape_percent']:.2f} | {values['bias_mw']:+,.1f} |"
        )
    lines.extend(
        [
            "",
            "MAE is the average absolute hourly error. RMSE penalizes larger misses more strongly.",
            "MAPE is the average absolute percentage error, not an 'accuracy percentage'.",
            "WAPE divides total absolute error by total actual load. Positive bias means overprediction.",
            "",
            "## Method and safeguards",
            "",
            "- All 168 predictions are issued together; no observed January load feeds later predictions.",
            "- Time is UTC for the forecast and evaluation. Calendar features use Europe/Berlin local time.",
            "- Features include daily/weekly/annual cycles, date, nationwide holidays and adjacent days, and year-end calendar flags.",
            "- Epiphany is a separate date flag, not incorrectly labeled a nationwide public holiday.",
            "- Lagged candidates use load from 1, 2, 3, 4 and 52 weeks, and 365 days earlier. Every lag is at least 168 hours.",
            "- Early annual lags can be missing; the tree model handles them natively. No backward filling is used.",
            "- Three fixed candidate models are compared on six rolling-origin seven-day validation windows before 2020.",
            "- Validation includes the first weeks of 2018 and 2019 and a Christmas week, as well as ordinary seasons.",
            "- Selection minimizes equally weighted mean validation RMSE among the trained candidates; baselines are reported separately.",
            "- Each fold refits on strictly earlier data. Early stopping is disabled, so there is no random validation split.",
            "- Target actuals and later-2020 observations are not used to choose or fit the model.",
            "- Input timestamp gaps, duplicates and invalid loads cause an error rather than silent imputation.",
            "",
            "## Limitations",
            "",
            "This is a retrospective holdout evaluation of one holiday-heavy week, not a guarantee of future accuracy.",
            "The preceding-week baseline copies Christmas week, so also compare the annual baselines.",
            "The 52-week baseline aligns weekdays; the 365-day baseline aligns calendar dates for this non-leap interval.",
            "There are no weather, industrial-activity or state-level school-holiday inputs, and no calibrated prediction intervals.",
            "Only Epiphany is explicitly flagged as a regional holiday; other regional effects are left to date features.",
            "The supplied OPSD file may contain retrospective revisions; its historical publication vintage is not verified.",
            "",
            "## Outputs",
            "",
            "- `forecast.csv`: all 168 actuals, forecasts, baselines and hourly errors.",
            "- `forecast.png` and `forecast.svg`: actual versus forecast.",
            "- `accuracy.csv`, `daily_scores.csv`, `validation_scores.csv`: numerical results.",
            "- `metrics.json`: results, selection details, timestamps, input digest and dependency versions.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(
        "\nFinal holdout scores:\n"
        + metrics_table.to_string(float_format=lambda value: f"{value:,.2f}")
    )
    print(f"\nArtifacts saved in {output.resolve()}")


if __name__ == "__main__":
    main()
