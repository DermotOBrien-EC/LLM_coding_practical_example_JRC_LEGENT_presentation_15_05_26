from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import holidays
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

VALUE_COLUMN = "DE_load_actual_entsoe_transparency"
ORIGIN = "2020-01-01"
BACKTEST_YEARS = (2017, 2018, 2019)
LAGS = (168, 336, 504, 672, 8736, 8760)


def target_hours(origin: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(origin, tz="UTC"), periods=168, freq="h")


def load_history(path: Path, origin: str) -> pd.Series:
    frame = pd.read_csv(path)
    timestamps = pd.to_datetime(frame["utc_timestamp"], utc=True)
    before = timestamps < pd.Timestamp(origin, tz="UTC")
    history = pd.Series(
        pd.to_numeric(frame.loc[before, VALUE_COLUMN], errors="raise").to_numpy(dtype=float),
        index=pd.DatetimeIndex(timestamps[before]),
        name="load_mw",
    ).sort_index()
    if history.empty or history.index.has_duplicates:
        raise ValueError("Training history must have unique, nonempty timestamps")
    expected = pd.date_range(history.index.min(), history.index.max(), freq="h")
    if not history.index.equals(expected):
        raise ValueError("Training history must be continuous hourly data")
    if not np.isfinite(history.to_numpy()).all() or (history <= 0).any():
        raise ValueError("Training load must be finite and positive")
    if history.index[-1] != pd.Timestamp(origin, tz="UTC") - pd.Timedelta(hours=1):
        raise ValueError("Training history must end immediately before forecast origin")
    return history


def build_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert("Europe/Berlin")
    calendar = holidays.country_holidays("DE", years=sorted(set(local.year)))
    national = np.array([date in calendar for date in local.date], dtype=int)
    month = local.month.to_numpy()
    day = local.day.to_numpy()
    hour = local.hour.to_numpy()
    weekday = local.dayofweek.to_numpy()
    new_year = ((month == 1) & (day == 1)).astype(int)
    christmas = ((month == 12) & (day == 25)).astype(int)
    boxing_day = ((month == 12) & (day == 26)).astype(int)
    epiphany = ((month == 1) & (day == 6)).astype(int)
    workday = ((weekday < 5) & (national == 0)).astype(int)
    day_kind = np.where(national, 6, weekday)
    day_kind = np.where(new_year, 7, day_kind)
    day_kind = np.where(christmas, 8, day_kind)
    day_kind = np.where(boxing_day, 9, day_kind)
    # Christmas shutdown and January recovery are not captured by public holidays alone.
    year_end_day = np.where(
        (month == 12) & (day >= 20), day - 32, np.where((month == 1) & (day <= 10), day, -100)
    )
    year_end_block = np.where(
        year_end_day == -100, -1, (year_end_day + 12) * 12 + workday * 6 + hour // 4
    )
    features = pd.DataFrame(
        {
            "hour": hour,
            "weekday": weekday,
            "month": month,
            "day": day,
            "day_of_year": local.dayofyear,
            "trend_years": (index - pd.Timestamp("2015-01-01", tz="UTC")).total_seconds()
            / (365.25 * 86400),
            "national_holiday": national,
            "new_year": new_year,
            "christmas": christmas,
            "boxing_day": boxing_day,
            "epiphany": epiphany,
            "workday": workday,
            "day_kind": day_kind,
            "year_end_day": year_end_day,
            "hour_kind": day_kind * 24 + hour,
            "month_hour": month * 24 + hour,
            "year_end_block": year_end_block,
            "epiphany_hour": np.where(epiphany, hour // 4, -1),
        },
        index=index,
    )
    for harmonic in (1, 2, 3):
        angle = 2 * np.pi * harmonic * local.dayofyear.to_numpy() / 365.25
        features[f"annual_sin_{harmonic}"] = np.sin(angle)
        features[f"annual_cos_{harmonic}"] = np.cos(angle)
    return features


def forecast_candidates(history: pd.Series, target: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    if history.index.max() >= target.min():
        raise ValueError("All training observations must be before the forecast origin")
    if history.index.max() != target.min() - pd.Timedelta(hours=1):
        raise ValueError("History must end one hour before the forecast origin")
    if len(target) != 168 or (np.diff(target.asi8) != 3_600_000_000_000).any():
        raise ValueError("Forecast target must contain 168 consecutive hourly timestamps")
    if len(history) < 8760:
        raise ValueError("At least one year of historical load is required")
    train_x = build_features(history.index)
    target_x = build_features(target)
    age_days = (target[0] - history.index).total_seconds().to_numpy() / 86400
    weights = np.exp(-np.log(2) * age_days / 730)
    y = history.to_numpy() / 1000
    predictions: dict[str, np.ndarray] = {}
    predictions["weekly_naive"] = history.reindex(target - pd.Timedelta(days=7)).to_numpy()

    categorical = ["hour_kind", "month_hour", "year_end_block", "epiphany_hour"]
    numeric = ["trend_years"] + [
        f"annual_{kind}_{harmonic}" for harmonic in (1, 2, 3) for kind in ("sin", "cos")
    ]
    transform = ColumnTransformer(
        [
            ("calendar", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("season", "passthrough", numeric),
        ]
    )
    ridge = make_pipeline(transform, Ridge(alpha=2.0, solver="lsqr", tol=1e-7))
    ridge.fit(train_x, y, ridge__sample_weight=weights)
    # Recent ordinary days anchor the level without treating the Christmas shutdown as a trend.
    recent = (age_days <= 56) & (train_x["year_end_day"].to_numpy() == -100)
    recent &= train_x["national_holiday"].to_numpy() == 0
    correction = float(np.mean(y[recent] - ridge.predict(train_x.loc[recent])))
    predictions["calendar_ridge"] = (ridge.predict(target_x) + correction) * 1000

    for model_name in ("calendar_boosting", "lagged_boosting"):
        x = train_x.copy()
        future_x = target_x.copy()
        if model_name == "lagged_boosting":
            # Every lag is at least the whole horizon, so no recursive or future actuals are used.
            for lag in LAGS:
                x[f"lag_{lag}"] = (
                    history.reindex(history.index - pd.Timedelta(hours=lag)).to_numpy() / 1000
                )
                future_x[f"lag_{lag}"] = (
                    history.reindex(target - pd.Timedelta(hours=lag)).to_numpy() / 1000
                )
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.04,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=40,
            reg_lambda=5.0,
            random_state=2020,
            n_jobs=2,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        model.fit(x, y, sample_weight=weights)
        predictions[model_name] = model.predict(future_x) * 1000
    predictions["holiday_ensemble"] = np.mean(
        [predictions[name] for name in ("calendar_ridge", "calendar_boosting", "lagged_boosting")],
        axis=0,
    )
    if any(
        len(values) != 168 or not np.isfinite(values).all() or (values <= 0).any()
        for values in predictions.values()
    ):
        raise ValueError("Every candidate must produce 168 finite, positive hourly forecasts")
    return predictions


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(np.mean(np.abs(error) / actual) * 100),
        "bias_mw": float(np.mean(error)),
    }


def run(input_path: Path, output_dir: Path) -> dict[str, Any]:
    history = load_history(input_path, ORIGIN)
    metrics: list[dict[str, Any]] = []
    validation_rows: list[pd.DataFrame] = []
    for year in BACKTEST_YEARS:
        target = target_hours(f"{year}-01-01")
        train = history.loc[history.index < target[0]]
        actual = history.reindex(target).to_numpy()
        predictions = forecast_candidates(train, target)
        for name, predicted in predictions.items():
            row = {"year": year, "model": name, **score(actual, predicted)}
            metrics.append(row)
            validation_rows.append(
                pd.DataFrame(
                    {
                        "utc_timestamp": target,
                        "year": year,
                        "model": name,
                        "actual_mw": actual,
                        "forecast_mw": predicted,
                        "residual_mw": actual - predicted,
                    }
                )
            )
            print(json.dumps(row), flush=True)
    metrics_frame = pd.DataFrame(metrics)
    validation = pd.concat(validation_rows, ignore_index=True)
    average = metrics_frame.groupby("model")[["mae_mw", "rmse_mw", "mape_pct", "bias_mw"]].mean()
    average = average.sort_values("mae_mw")
    selected = str(average.index[0])
    target = target_hours(ORIGIN)
    predicted = forecast_candidates(history, target)[selected]
    residuals = validation.loc[validation["model"] == selected, "residual_mw"].to_numpy()
    lower_error, upper_error = np.quantile(residuals, [0.05, 0.95])
    forecast = pd.DataFrame(
        {
            "utc_timestamp": target,
            "germany_timestamp": target.tz_convert("Europe/Berlin"),
            "forecast_mw": predicted,
            "lower_90_mw": np.maximum(0, predicted + lower_error),
            "upper_90_mw": predicted + upper_error,
        }
    )
    daily = (
        forecast.set_index("utc_timestamp")
        .resample("D")["forecast_mw"]
        .agg(["mean", "min", "max", "sum"])
    )
    daily.columns = ["mean_load_mw", "min_load_mw", "max_load_mw", "energy_mwh"]
    metadata: dict[str, Any] = {
        "forecast_origin_utc": str(target[0]),
        "forecast_end_utc": str(target[-1]),
        "forecast_hours": len(target),
        "units_assumption": "MW, following the supplied OPSD ENTSO-E load series convention",
        "training_start_utc": str(history.index[0]),
        "training_end_utc": str(history.index[-1]),
        "training_rows": len(history),
        "training_sha256": hashlib.sha256(history.to_csv().encode()).hexdigest(),
        "input_file_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "selected_model": selected,
        "selection_rule": "Lowest mean January-week MAE across 2017, 2018 and 2019",
        "backtest_years": list(BACKTEST_YEARS),
        "selected_model_backtest": average.loc[selected].to_dict(),
        "baseline_backtest": average.loc["weekly_naive"].to_dict(),
        "uncertainty": {
            "method": "5th and 95th percentiles of pooled selected-model validation residuals",
            "residual_count": len(residuals),
            "lower_offset_mw": float(lower_error),
            "upper_offset_mw": float(upper_error),
            "caveat": "Nominal pointwise 90% empirical bands, not independently calibrated coverage. "
            "Model selection and interval estimation reuse the same three January weeks. "
            "Hourly residuals are correlated; these are not simultaneous weekly bands.",
        },
        "forecast_mean_mw": float(predicted.mean()),
        "forecast_energy_mwh": float(predicted.sum()),
        "forecast_min_mw": float(predicted.min()),
        "forecast_min_utc": str(target[int(predicted.argmin())]),
        "forecast_peak_mw": float(predicted.max()),
        "forecast_peak_utc": str(target[int(predicted.argmax())]),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "holidays": version("holidays"),
        },
        "post_origin_actuals_used": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_dir / "forecast.csv", index=False, float_format="%.2f")
    daily.to_csv(output_dir / "daily_forecast.csv", float_format="%.2f")
    metrics_frame.to_csv(output_dir / "backtest_metrics.csv", index=False, float_format="%.3f")
    average.to_csv(output_dir / "model_comparison.csv", float_format="%.3f")
    validation.to_csv(output_dir / "backtest_predictions.csv", index=False, float_format="%.3f")
    (output_dir / "forecast_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leakage-free German hourly load forecast")
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("opsd_de_load.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    run(args.input, args.output_dir)
