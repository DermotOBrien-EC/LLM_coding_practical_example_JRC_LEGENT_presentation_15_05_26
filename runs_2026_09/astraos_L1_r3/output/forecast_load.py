from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date
from importlib.metadata import version
from pathlib import Path
from typing import Any

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

CUTOFF = pd.Timestamp("2020-01-01", tz="UTC")
HORIZON = 168
LOAD_COLUMN = "DE_load_actual_entsoe_transparency"
LAGS = (168, 336, 504, 672, 8736, 8760, 8784)
TREES = 650
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Fold:
    start: str
    kind: str


FOLDS = (
    Fold("2017-01-01", "new_year"),
    Fold("2018-01-01", "new_year"),
    Fold("2019-01-01", "new_year"),
    Fold("2019-10-09", "robustness"),
    Fold("2019-11-06", "robustness"),
    Fold("2019-12-04", "robustness"),
    Fold("2019-12-18", "robustness"),
    Fold("2019-12-25", "robustness"),
)


def load_history(path: Path, cutoff: pd.Timestamp = CUTOFF) -> pd.Series:
    raw = pd.read_csv(path, dtype=str)
    if set(raw.columns) != {"utc_timestamp", LOAD_COLUMN}:
        raise ValueError("Unexpected input columns")
    timestamps = pd.to_datetime(raw["utc_timestamp"], utc=True, errors="raise")
    # Future values are discarded before numeric parsing, filling, or feature creation.
    keep = timestamps < cutoff
    index = pd.DatetimeIndex(timestamps.loc[keep])
    values = pd.to_numeric(raw.loc[keep, LOAD_COLUMN], errors="raise").to_numpy(dtype=float)
    history = pd.Series(values, index=index, name=LOAD_COLUMN).sort_index()
    if history.empty or history.index.has_duplicates:
        raise ValueError("Empty or duplicated historical timestamps")
    expected = pd.date_range(history.index.min(), cutoff, freq="h", inclusive="left")
    if not history.index.equals(expected):
        raise ValueError("Historical data must be complete and hourly through the cutoff")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Historical loads must be finite and positive")
    return history


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert("Europe/Berlin")
    dates = local.date
    years = range(int(local.year.min()) - 1, int(local.year.max()) + 2)
    national = holidays.country_holidays("DE", years=years, language="en_US")
    unique_dates = set(dates)
    holiday_by_date = {day: int(day in national) for day in unique_dates}
    new_year_by_date = {
        day: (day - date(day.year + int(day.month >= 7), 1, 1)).days for day in unique_dates
    }
    christmas_by_date = {
        day: (day - date(day.year - int(day.month < 7), 12, 25)).days for day in unique_dates
    }
    ny_distance = np.array([new_year_by_date[day] for day in dates])
    xmas_distance = np.array([christmas_by_date[day] for day in dates])
    is_holiday = np.array([holiday_by_date[day] for day in dates])
    day_of_year = local.dayofyear.to_numpy()
    hour = local.hour.to_numpy()
    month = local.month.to_numpy()
    day = local.day.to_numpy()
    new_year = (month == 1) & (day == 1)
    christmas = (month == 12) & np.isin(day, [25, 26])
    holiday_kind = np.select([new_year, christmas, is_holiday > 0], [2, 3, 1], default=0)
    features = pd.DataFrame(
        {
            "hour": hour,
            "weekday": local.dayofweek,
            "hour_of_week": local.dayofweek * 24 + hour,
            "month": month,
            "day_of_month": day,
            "day_of_year": day_of_year,
            "year": local.year,
            "national_holiday": is_holiday,
            "holiday_kind": holiday_kind,
            "new_year_distance": np.clip(ny_distance, -21, 21),
            "christmas_distance": np.clip(xmas_distance, -21, 21),
            "epiphany": ((month == 1) & (day == 6)).astype(int),
            "christmas_eve": ((month == 12) & (day == 24)).astype(int),
            "new_year_eve": ((month == 12) & (day == 31)).astype(int),
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
        },
        index=index,
    )
    for harmonic in (1, 2, 3):
        phase = 2 * np.pi * harmonic * (day_of_year - 1) / 365.25
        features[f"annual_sin_{harmonic}"] = np.sin(phase)
        features[f"annual_cos_{harmonic}"] = np.cos(phase)
    return features


def build_features(index: pd.DatetimeIndex, history: pd.Series, *, with_lags: bool) -> pd.DataFrame:
    features = calendar_features(index)
    if not with_lags:
        return features
    weekly_columns: list[str] = []
    for lag in LAGS:
        source = index - pd.Timedelta(hours=lag)
        column = f"load_lag_{lag}"
        features[column] = history.reindex(source).to_numpy(dtype=float)
        lag_calendar = calendar_features(source)
        for name in ("hour", "weekday", "national_holiday", "new_year_distance"):
            features[f"lag_{lag}_{name}"] = lag_calendar[name].to_numpy()
        if lag <= 672:
            weekly_columns.append(column)
    features["weekly_median"] = features[weekly_columns].median(axis=1)
    features["weekly_mean"] = features[weekly_columns].mean(axis=1)
    return features


def new_model(n_estimators: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=24,
        reg_lambda=20.0,
        colsample_bytree=0.9,
        random_state=2020,
        n_jobs=2,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def forecast_candidates(
    history: pd.Series, start: pd.Timestamp, *, n_estimators: int = TREES
) -> dict[str, FloatArray]:
    available = history.loc[history.index < start].copy()
    target = pd.date_range(start, periods=HORIZON, freq="h")
    if available.empty or available.index.max() != start - pd.Timedelta(hours=1):
        raise ValueError("Forecast origin must immediately follow observed history")
    weekly = np.column_stack(
        [available.reindex(target - pd.Timedelta(hours=lag)).to_numpy() for lag in LAGS[:4]]
    )
    analog = available.reindex(target - pd.Timedelta(hours=8736)).to_numpy(dtype=float)
    recent_index = pd.date_range(start - pd.Timedelta(hours=672), periods=672, freq="h")
    recent_mean = float(available.reindex(recent_index).mean())
    annual_mean = float(available.reindex(recent_index - pd.Timedelta(hours=8736)).mean())
    predictions: dict[str, FloatArray] = {
        "weekly_naive": weekly[:, 0],
        "four_week_median": np.median(weekly, axis=1),
        "scaled_annual_naive": analog * recent_mean / annual_mean,
    }
    for label, with_lags in (("calendar_lgbm", False), ("lag_calendar_lgbm", True)):
        training_index = available.index
        if with_lags:
            # Keep early rows with absent annual lags: LightGBM handles their missing
            # features, preserving the few historical New Year examples available.
            training_index = training_index[672:]
        train_x = build_features(training_index, available, with_lags=with_lags)
        test_x = build_features(target, available, with_lags=with_lags)
        age_days = (start - training_index).total_seconds().to_numpy() / 86400.0
        weights = np.exp(-np.log(2) * age_days / 730.0)
        model = new_model(n_estimators)
        model.fit(train_x, available.loc[training_index].to_numpy(), sample_weight=weights)
        predictions[label] = np.asarray(model.predict(test_x), dtype=float)
    predictions["equal_blend"] = (
        predictions["calendar_lgbm"] + predictions["lag_calendar_lgbm"]
    ) / 2.0
    for label, values in predictions.items():
        if values.shape != (HORIZON,) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"Invalid forecast from {label}")
    return predictions


def backtest(history: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for fold in FOLDS:
        start = pd.Timestamp(fold.start, tz="UTC")
        target = pd.date_range(start, periods=HORIZON, freq="h")
        if target.max() >= CUTOFF:
            raise ValueError("A backtest cannot use 2020 outcomes")
        actual = history.reindex(target).to_numpy(dtype=float)
        if not np.isfinite(actual).all():
            raise ValueError("Missing validation observations")
        # Labels remain outside the forecasting call, including preprocessing.
        candidates = forecast_candidates(history.loc[history.index < start], start)
        for name, values in candidates.items():
            errors = values - actual
            metrics.append(
                {
                    "fold_start_utc": start.isoformat(),
                    "fold_type": fold.kind,
                    "model": name,
                    "mae_mw": float(np.mean(np.abs(errors))),
                    "rmse_mw": float(np.sqrt(np.mean(errors**2))),
                    "mape_pct": float(100 * np.mean(np.abs(errors) / actual)),
                    "bias_mw": float(np.mean(errors)),
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "utc_timestamp": target,
                        "fold_start_utc": start,
                        "fold_type": fold.kind,
                        "model": name,
                        "actual_load_mw": actual,
                        "forecast_load_mw": values,
                    }
                )
            )
        print(f"Completed {fold.kind} backtest {fold.start}", flush=True)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def select_model(metrics: pd.DataFrame) -> str:
    january = metrics.loc[metrics["fold_type"] == "new_year"]
    if january.empty:
        raise ValueError("No New Year validation folds")
    scores = january.groupby("model")["mae_mw"].mean().sort_values()
    return str(scores.index[0])


def make_forecast_frame(values: FloatArray) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    if values.shape != (HORIZON,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Forecast must contain 168 finite positive loads")
    return pd.DataFrame(
        {
            "utc_timestamp": pd.date_range(CUTOFF, periods=HORIZON, freq="h"),
            "forecast_load_mw": values,
        }
    )


def markdown_table(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |"]
    lines.append("| " + " | ".join(["---"] * len(frame.columns)) + " |")
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def write_report(
    directory: Path,
    history: pd.Series,
    forecast: pd.DataFrame,
    metrics: pd.DataFrame,
    winner: str,
    source_sha: str,
) -> None:
    january = metrics.loc[metrics["fold_type"] == "new_year"]
    comparison = january.groupby("model")[["mae_mw", "rmse_mw", "mape_pct"]].mean()
    comparison = comparison.sort_values("mae_mw").round(2).reset_index()
    folds = metrics.loc[metrics["model"] == winner].drop(columns="model").round(2)
    learned_by_fold = (
        january.loc[january["model"].isin(["calendar_lgbm", "lag_calendar_lgbm", "equal_blend"])]
        .pivot(index="fold_start_utc", columns="model", values="mae_mw")
        .round(2)
        .reset_index()
    )
    indexed = forecast.set_index("utc_timestamp")["forecast_load_mw"]
    daily = indexed.resample("D").agg(["mean", "min", "max"])
    daily.columns = ["mean_load_mw", "minimum_load_mw", "maximum_load_mw"]
    daily.index.name = "date_utc"
    daily.round(2).to_csv(directory / "daily_summary.csv")
    daily_display = daily.round(0).astype(int).reset_index()
    daily_display["date_utc"] = daily_display["date_utc"].dt.strftime("%Y-%m-%d")
    best_baseline = (
        january.loc[
            january["model"].isin(["weekly_naive", "four_week_median", "scaled_annual_naive"])
        ]
        .groupby("model")["mae_mw"]
        .mean()
        .min()
    )
    winner_mae = float(january.loc[january["model"] == winner, "mae_mw"].mean())
    skill = 100 * (1 - winner_mae / best_baseline)
    report = f"""# German hourly electricity load forecast

## Forecast

- Window: **1 January 2020 00:00 to 7 January 2020 23:00 UTC**, 168 hourly intervals.
- File: `forecast.csv`; columns: `utc_timestamp`, `forecast_load_mw`.
- Values are hourly mean electrical power, interpreted as MW under the OPSD/ENTSO-E column convention. They are not instantaneous peaks or hourly MWh values.
- UTC matches the supplied timestamps. In Germany this is 1 January 01:00 to 8 January 00:00 CET. Calendar features use Europe/Berlin local time.
- Selected model: **{winner}**.
- Forecast mean load: **{indexed.mean():,.0f} MW**. Sum of hourly mean loads times one hour: **{indexed.sum() / 1e6:.3f} TWh**.
- Lowest hourly forecast: **{indexed.min():,.0f} MW** at {indexed.idxmin().isoformat()}.
- Highest hourly forecast: **{indexed.max():,.0f} MW** at {indexed.idxmax().isoformat()}.

### Daily summary (UTC)

{markdown_table(daily_display)}

## Data and information cutoff

The supplied `opsd_de_load.csv` is the sole data source. The series is
`{LOAD_COLUMN}`. All observations on or after 2020-01-01 00:00 UTC
are removed before load-value parsing, feature generation, fitting, or validation.
The target week's actual loads were not used to select, fit, or score the forecast.
This is a retrospective forecast with an enforced historical information cutoff,
not a reconstruction using the now-known 2020 outcomes.

Training coverage: {history.index.min().isoformat()} through
{history.index.max().isoformat()}, **{len(history):,} observations**. The input
history has no missing hours, duplicate timestamps, nonfinite values, or
nonpositive loads. No load imputation was required. Historical load range:
{history.min():,.0f} to {history.max():,.0f} MW. The input was not independently
reconciled against the original provider or an as-of-2019 data vintage, so
historical revisions in this supplied series cannot be ruled out.

## Method fixed before validation

Six candidates were compared:

1. Previous-week load, lag 168 hours.
2. Median of the previous four same-weekday/hour loads, lags 168/336/504/672.
3. Previous year's weekday-aligned load, lag 8736 hours (52 weeks), scaled
   by the ratio of the latest 28-day mean to the 28-day mean 52 weeks earlier.
   This baseline is weekday-aligned, not holiday-aligned.
4. Calendar-only LightGBM regression.
5. LightGBM with the same calendar plus historical load lags.
6. A fixed 50:50 average of candidates 4 and 5.

Calendar features include local hour, weekday, month, day of month/year,
annual Fourier terms, year, German nationwide public holidays, New Year's Day,
Christmas, Christmas Eve, New Year's Eve, and a regional Epiphany indicator.
Signed, clipped distances to New Year and Christmas allow a holiday-period
ramp without classifying every weekday as an ordinary workday. Epiphany is
an indicator for 6 January, not an assumed nationwide closure or state-weighted
load estimate. No school-holiday calendar or weather forecast is used.
Historical weekday Epiphany examples are scarce and receive reduced weight
under the recency weighting, making the 6 January estimate particularly
uncertain.

The lag model uses {list(LAGS)}-hour lags, the corresponding local calendar
states, and weekly median/mean features. The annual lags span 364/365/366 days,
so the model can distinguish weekday and calendar-date analogs. It omits the
first 672 training hours but keeps subsequent rows with absent annual lags,
using LightGBM's missing-feature handling. It does not invent missing loads.
Every target-week load lag is observed before the forecast origin; no recursive
predictions or realized loads from inside the forecast week enter the features.

Both regressors use {TREES} trees, learning rate 0.035, 31 leaves, minimum
24 observations per leaf, L2 regularization 20, column sampling 0.9,
seed 2020, two threads, and deterministic column-wise training.
Training weights decay with a 730-day half-life. There is no early stopping,
no scored-week tuning of tree count, and no broad hyperparameter search.
The year feature in a tree model does not extrapolate a linear trend beyond
2019. The selected candidate is refitted on all eligible pre-2020 history.

## Backtests

Selection minimizes mean hourly absolute error (MAE) across the complete
first weeks of January in 2017, 2018, and 2019. Each is a fixed-origin,
168-hour forecast trained solely on earlier observations. All three weeks
have equal length, so mean fold MAE equals pooled hourly MAE. The following
RMSE column is the mean of the three fold RMSE values, not pooled RMSE.

{markdown_table(comparison)}

The selected model's MAE is {skill:.1f}% lower than the best of the three
seasonal baselines on these selection folds. These baselines are imperfect:
the previous week contains Christmas, while a 52-week analog can place
New Year's Day on a different holiday status.

### Learned candidates by January week (MAE, MW)

{markdown_table(learned_by_fold)}

The lowest-error component changes across years. The fixed blend is selected
for its three-year mean performance, not because it dominates every fold.

### Selected model by validation week

{markdown_table(folds)}

The five late-2019 weeks are separate robustness checks and do not enter
model selection. January's three folds represent only three New Year
transitions, starting on Sunday, Monday, and Tuesday, whereas 2020 starts
on Wednesday. These are selection-set errors, not an unbiased estimate of
the selected model's future error. The observations within each week are
strongly correlated, so 504 hourly residuals are not 504 independent holiday
examples. No calibrated prediction intervals are claimed or supplied.
Unusual weather, industrial shutdowns, and calendar-specific return-to-work
patterns can materially shift the realized load.

## Reproduction and artifacts

From this directory, using the supplied project environment:

```sh
../../.venv/bin/python forecast_load.py
../../.venv/bin/python -m pytest -q test_forecast_load.py
```

- `forecast.csv`: all 168 point forecasts, MW, two decimal places.
- `daily_summary.csv`: daily mean/minimum/maximum point forecasts, MW.
- `backtest_metrics.csv`: all models on all eight validation weeks.
- `backtest_predictions.csv`: historical validation labels and forecasts for independent metric recomputation; contains no 2020 target labels.
- `run_metadata.json`: cutoff, model settings, versions, and source fingerprint.
- `forecast_load.py` and `test_forecast_load.py`: executable method and tests.

Input SHA-256: `{source_sha}`.
"""
    if chr(0x2014) in report:
        raise ValueError("Report contains a prohibited em dash")
    (directory / "forecast_report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast German load for the first week of 2020")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("opsd_de_load.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    history = load_history(args.input)
    print(f"Using {len(history)} pre-2020 observations, last: {history.index.max()}", flush=True)
    metrics, validation_predictions = backtest(history)
    winner = select_model(metrics)
    print(f"Selected model: {winner}", flush=True)
    values = forecast_candidates(history, CUTOFF)[winner]
    forecast = make_forecast_frame(values)
    directory: Path = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(directory / "forecast.csv", index=False, float_format="%.2f")
    metrics.to_csv(directory / "backtest_metrics.csv", index=False, float_format="%.6f")
    validation_predictions.to_csv(
        directory / "backtest_predictions.csv", index=False, float_format="%.6f"
    )
    source_sha = hashlib.sha256(args.input.read_bytes()).hexdigest()
    metadata = {
        "cutoff_exclusive_utc": CUTOFF.isoformat(),
        "forecast_hours": HORIZON,
        "training_rows": len(history),
        "training_last_utc": history.index.max().isoformat(),
        "input_sha256": source_sha,
        "selected_model": winner,
        "selection_metric": "mean January 2017/2018/2019 MAE only",
        "folds": [asdict(fold) for fold in FOLDS],
        "lags_hours": list(LAGS),
        "training_half_life_days": 730,
        "lightgbm_parameters": new_model(TREES).get_params(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
            "holidays": version("holidays"),
        },
    }
    (directory / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(directory, history, forecast, metrics, winner, source_sha)
    print(f"Saved {directory / 'forecast.csv'}", flush=True)
    print(f"Mean load {values.mean():.2f} MW; energy {values.sum() / 1e6:.6f} TWh", flush=True)


if __name__ == "__main__":
    main()
