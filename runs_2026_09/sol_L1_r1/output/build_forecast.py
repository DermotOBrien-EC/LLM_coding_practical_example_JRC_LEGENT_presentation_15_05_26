from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path(__file__).with_name("opsd_de_load.csv")
FORECAST_PATH = Path(__file__).with_name("forecast_jan_2020.csv")
DIAGNOSTICS_PATH = Path(__file__).with_name("forecast_diagnostics.json")
TARGET_COLUMN = "DE_load_actual_entsoe_transparency"
TARGET_START = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
TARGET_END = pd.Timestamp("2020-01-08 00:00:00", tz="UTC")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    lookback_years: int | None
    objective: str
    decay_half_life_days: float | None
    winter_weight: float
    num_leaves: int
    min_child_samples: int


def load_series() -> pd.Series:
    frame = pd.read_csv(DATA_PATH, parse_dates=["utc_timestamp"])
    series = frame.set_index("utc_timestamp")[TARGET_COLUMN].astype(float)
    series = series.asfreq("h")
    if series.isna().any():
        raise ValueError("The hourly input series contains missing values")
    return series


def exact_year_lag(series: pd.Series, years: int) -> pd.Series:
    source_index = series.index - pd.DateOffset(years=years)
    return pd.Series(series.reindex(source_index).to_numpy(), index=series.index)


def make_features(series: pd.Series) -> pd.DataFrame:
    index = series.index
    local = index.tz_convert("Europe/Berlin")
    local_dates = pd.Index(local.date)
    german_holidays = holidays.country_holidays(
        "DE", years=range(index.min().year - 1, index.max().year + 2)
    )

    features = pd.DataFrame(index=index)
    features["trend_years"] = (index - index.min()).total_seconds() / (365.25 * 86400)
    features["hour"] = local.hour
    features["day_of_week"] = local.dayofweek
    features["day_of_month"] = local.day
    features["month"] = local.month
    features["day_of_year"] = local.dayofyear
    features["week_of_year"] = local.isocalendar().week.to_numpy(dtype=float)
    features["is_weekend"] = (local.dayofweek >= 5).astype(int)
    features["is_holiday"] = np.fromiter(
        (date in german_holidays for date in local_dates), dtype=int, count=len(index)
    )
    features["day_type"] = np.where(
        features["is_holiday"].to_numpy() == 1,
        3,
        np.where(local.dayofweek == 5, 1, np.where(local.dayofweek == 6, 2, 0)),
    )
    features["hour_day_type"] = features["hour"] + 24 * features["day_type"]

    month_day = local.strftime("%m-%d")
    special_days = {
        "12-24": 1,
        "12-25": 2,
        "12-26": 3,
        "12-27": 4,
        "12-28": 5,
        "12-29": 6,
        "12-30": 7,
        "12-31": 8,
        "01-01": 9,
        "01-02": 10,
        "01-03": 11,
        "01-04": 12,
        "01-05": 13,
        "01-06": 14,
        "01-07": 15,
    }
    features["year_end_day"] = pd.Series(month_day, index=index).map(special_days).fillna(0)
    features["is_bridge_period"] = (
        ((local.month == 12) & (local.day >= 24)) | ((local.month == 1) & (local.day <= 6))
    ).astype(int)

    hour_angle = 2 * np.pi * local.hour / 24
    week_angle = 2 * np.pi * (local.dayofweek * 24 + local.hour) / 168
    features["hour_sin"] = np.sin(hour_angle)
    features["hour_cos"] = np.cos(hour_angle)
    features["week_sin"] = np.sin(week_angle)
    features["week_cos"] = np.cos(week_angle)
    for harmonic in range(1, 5):
        angle = 2 * np.pi * harmonic * (local.dayofyear - 1 + local.hour / 24) / 365.25
        features[f"year_sin_{harmonic}"] = np.sin(angle)
        features[f"year_cos_{harmonic}"] = np.cos(angle)

    weekly_lags = [168, 336, 504, 672, 840]
    for lag in weekly_lags:
        features[f"lag_{lag}"] = series.shift(lag)
    features["lag_week_mean"] = features[[f"lag_{lag}" for lag in weekly_lags]].mean(axis=1)
    features["lag_week_std"] = features[[f"lag_{lag}" for lag in weekly_lags]].std(axis=1)
    features["lag_year_exact"] = exact_year_lag(series, 1)
    features["lag_two_year_exact"] = exact_year_lag(series, 2)
    features["lag_364d"] = series.shift(24 * 364)
    features["lag_365d"] = series.shift(24 * 365)
    features["lag_366d"] = series.shift(24 * 366)
    features["recent_level"] = series.shift(168).rolling(24 * 28, min_periods=24 * 14).mean()
    features["prior_year_level"] = (
        features["lag_year_exact"].shift(168).rolling(24 * 28, min_periods=24 * 14).mean()
    )
    features["annual_level_ratio"] = (
        features["recent_level"] / features["prior_year_level"]
    ).clip(0.85, 1.15)
    return features


def make_sample_weights(
    index: pd.DatetimeIndex, cutoff: pd.Timestamp, spec: ModelSpec
) -> np.ndarray:
    weights = np.ones(len(index), dtype=float)
    if spec.decay_half_life_days is not None:
        age_days = (cutoff - index).total_seconds() / 86400
        weights *= np.exp2(-age_days / spec.decay_half_life_days)
    if spec.winter_weight != 1.0:
        local_month = index.tz_convert("Europe/Berlin").month
        weights *= np.where(np.isin(local_month, [11, 12, 1, 2, 3]), spec.winter_weight, 1.0)
    return weights


def fit_predict_model(
    features: pd.DataFrame,
    target: pd.Series,
    forecast_index: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
    spec: ModelSpec,
) -> np.ndarray:
    train_mask = features.index < cutoff
    if spec.lookback_years is not None:
        train_mask &= features.index >= cutoff - pd.DateOffset(years=spec.lookback_years)
    train_coverage = features.loc[train_mask].notna().mean(axis=0)
    valid_columns = features.columns[
        features.loc[forecast_index].notna().all(axis=0) & (train_coverage >= 0.5)
    ]
    x_train = features.loc[train_mask, valid_columns]
    y_train = target.loc[train_mask]
    complete = x_train.notna().all(axis=1) & y_train.notna()
    x_train = x_train.loc[complete]
    y_train = y_train.loc[complete]
    weights = make_sample_weights(x_train.index, cutoff, spec)

    model = lgb.LGBMRegressor(
        objective=spec.objective,
        n_estimators=650,
        learning_rate=0.025,
        num_leaves=spec.num_leaves,
        min_child_samples=spec.min_child_samples,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=25.0,
        reg_lambda=100.0,
        random_state=41,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train, sample_weight=weights)
    return np.asarray(model.predict(features.loc[forecast_index, valid_columns]), dtype=float)


def baseline_predictions(
    features: pd.DataFrame, forecast_index: pd.DatetimeIndex
) -> dict[str, np.ndarray]:
    target_features = features.loc[forecast_index]
    annual_scale = target_features["annual_level_ratio"].fillna(1.0).to_numpy()
    return {
        "lag_168": target_features["lag_168"].to_numpy(),
        "mean_5_weeks": target_features["lag_week_mean"].to_numpy(),
        "same_date_last_year": target_features["lag_year_exact"].to_numpy() * annual_scale,
        "same_weekday_last_year": target_features["lag_364d"].to_numpy() * annual_scale,
        "annual_weekday_blend": (
            0.55 * target_features["lag_year_exact"].to_numpy()
            + 0.45 * target_features["lag_364d"].to_numpy()
        )
        * annual_scale,
    }


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "mae_mw": float(mean_absolute_error(actual, prediction)),
        "rmse_mw": float(mean_squared_error(actual, prediction) ** 0.5),
        "mape_percent": float(np.mean(np.abs((actual - prediction) / actual)) * 100),
    }


def main() -> None:
    target = load_series()
    features = make_features(target)
    specs = [
        ModelSpec("lgb_all_l1", None, "regression_l1", 730.0, 1.5, 31, 48),
        ModelSpec("lgb_4y_l1", 4, "regression_l1", 730.0, 1.5, 31, 48),
        ModelSpec("lgb_3y_l1", 3, "regression_l1", 540.0, 1.75, 31, 48),
        ModelSpec("lgb_3y_l2", 3, "regression", 540.0, 1.75, 31, 48),
        ModelSpec("lgb_4y_small", 4, "regression_l1", 900.0, 2.0, 20, 72),
    ]
    backtest_years = [2017, 2018, 2019]
    predictions_by_candidate: dict[str, list[np.ndarray]] = {}
    actual_by_fold: list[np.ndarray] = []
    fold_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for year in backtest_years:
        start = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
        end = start + pd.Timedelta(days=7)
        forecast_index = pd.date_range(start, end, inclusive="left", freq="h")
        actual = target.loc[forecast_index].to_numpy()
        actual_by_fold.append(actual)
        candidates = baseline_predictions(features, forecast_index)
        for spec in specs:
            candidates[spec.name] = fit_predict_model(
                features, target, forecast_index, start, spec
            )
        for name, prediction in candidates.items():
            predictions_by_candidate.setdefault(name, []).append(prediction)
            fold_metrics.setdefault(name, {})[str(year)] = metrics(actual, prediction)

    all_actual = np.concatenate(actual_by_fold)
    summary_metrics: dict[str, dict[str, float]] = {}
    for name, fold_predictions in predictions_by_candidate.items():
        summary_metrics[name] = metrics(all_actual, np.concatenate(fold_predictions))

    ranked_models = sorted(summary_metrics, key=lambda name: summary_metrics[name]["mae_mw"])
    top_models = [name for name in ranked_models if name.startswith("lgb_")][:3]
    stacked = np.column_stack(
        [np.concatenate(predictions_by_candidate[name]) for name in top_models]
    )
    best_weight = np.full(len(top_models), 1 / len(top_models))
    best_mae = float("inf")
    if len(top_models) == 3:
        for first in np.linspace(0, 1, 21):
            for second in np.linspace(0, 1 - first, 21):
                third = 1 - first - second
                weight = np.array([first, second, third])
                score = mean_absolute_error(all_actual, stacked @ weight)
                if score < best_mae:
                    best_mae = float(score)
                    best_weight = weight
    ensemble_name = "ensemble_" + "_".join(top_models)
    ensemble_backtest = stacked @ best_weight
    summary_metrics[ensemble_name] = metrics(all_actual, ensemble_backtest)

    final_index = pd.date_range(TARGET_START, TARGET_END, inclusive="left", freq="h")
    final_model_predictions: dict[str, np.ndarray] = {}
    spec_by_name = {spec.name: spec for spec in specs}
    for name in top_models:
        final_model_predictions[name] = fit_predict_model(
            features, target.where(target.index < TARGET_START), final_index, TARGET_START, spec_by_name[name]
        )
    final_prediction = np.column_stack(
        [final_model_predictions[name] for name in top_models]
    ) @ best_weight

    output = pd.DataFrame(
        {
            "utc_timestamp": final_index,
            "DE_load_forecast_mw": np.round(final_prediction, 1),
        }
    )
    output.to_csv(FORECAST_PATH, index=False)

    diagnostics = {
        "training_cutoff_utc": TARGET_START.isoformat(),
        "forecast_end_exclusive_utc": TARGET_END.isoformat(),
        "hours": len(final_index),
        "ranked_backtest_metrics": {
            name: summary_metrics[name]
            for name in sorted(summary_metrics, key=lambda item: summary_metrics[item]["mae_mw"])
        },
        "fold_metrics": fold_metrics,
        "ensemble": {
            "members": top_models,
            "weights": best_weight.tolist(),
            "backtest_metrics": summary_metrics[ensemble_name],
        },
        "forecast_summary": {
            "minimum_mw": float(final_prediction.min()),
            "maximum_mw": float(final_prediction.max()),
            "mean_mw": float(final_prediction.mean()),
            "daily_mean_mw": {
                str(day.date()): float(value)
                for day, value in pd.Series(final_prediction, index=final_index)
                .groupby(final_index.floor("D"))
                .mean()
                .items()
            },
        },
    }
    DIAGNOSTICS_PATH.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(json.dumps(diagnostics, indent=2))
    print(f"Wrote {FORECAST_PATH}")


if __name__ == "__main__":
    main()
