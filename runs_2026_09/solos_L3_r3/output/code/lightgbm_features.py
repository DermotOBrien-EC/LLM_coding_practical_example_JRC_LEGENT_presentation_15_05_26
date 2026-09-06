from __future__ import annotations

import time
from typing import Any

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import ForecastResult, QUANTILES, SEED, mape_pct

FEATURE_NAMES = (
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_public_holiday_de",
    "lag_24h",
    "lag_168h",
    "lag_8736h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_mean_168h",
    "rolling_std_168h",
)
GERMAN_HOLIDAYS = holidays.Germany(years=range(2015, 2021))
CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 30,
        "max_depth": -1,
    },
    {
        "n_estimators": 700,
        "learning_rate": 0.035,
        "num_leaves": 63,
        "min_child_samples": 40,
        "max_depth": -1,
    },
    {
        "n_estimators": 450,
        "learning_rate": 0.06,
        "num_leaves": 31,
        "min_child_samples": 60,
        "max_depth": 12,
    },
)


def _calendar_columns(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour": index.hour,
            "day_of_week": index.dayofweek,
            "month": index.month,
            "is_weekend": (index.dayofweek >= 5).astype(int),
            "is_public_holiday_de": [
                int(timestamp.date() in GERMAN_HOLIDAYS) for timestamp in index
            ],
        },
        index=index,
    )


def build_training_frame(series: pd.Series) -> pd.DataFrame:
    target = series.astype(float).rename("target")
    frame = _calendar_columns(pd.DatetimeIndex(series.index))
    frame["lag_24h"] = target.shift(24)
    frame["lag_168h"] = target.shift(168)
    frame["lag_8736h"] = target.shift(8_736)
    past = target.shift(1)
    frame["rolling_mean_24h"] = past.rolling(24).mean()
    frame["rolling_std_24h"] = past.rolling(24).std(ddof=0)
    frame["rolling_mean_168h"] = past.rolling(168).mean()
    frame["rolling_std_168h"] = past.rolling(168).std(ddof=0)
    frame["target"] = target
    return frame.dropna()


def make_feature_row(history: pd.Series, timestamp: pd.Timestamp) -> dict[str, float]:
    if timestamp <= history.index[-1]:
        raise ValueError("Feature timestamp must be after the supplied history")
    if len(history) < 8_736:
        raise ValueError("At least 8,736 hours of history are required")
    row: dict[str, float] = {
        "hour": float(timestamp.hour),
        "day_of_week": float(timestamp.dayofweek),
        "month": float(timestamp.month),
        "is_weekend": float(timestamp.dayofweek >= 5),
        "is_public_holiday_de": float(timestamp.date() in GERMAN_HOLIDAYS),
        "lag_24h": float(history.loc[timestamp - pd.Timedelta(hours=24)]),
        "lag_168h": float(history.loc[timestamp - pd.Timedelta(hours=168)]),
        "lag_8736h": float(history.loc[timestamp - pd.Timedelta(hours=8_736)]),
        "rolling_mean_24h": float(history.iloc[-24:].mean()),
        "rolling_std_24h": float(history.iloc[-24:].std(ddof=0)),
        "rolling_mean_168h": float(history.iloc[-168:].mean()),
        "rolling_std_168h": float(history.iloc[-168:].std(ddof=0)),
    }
    return row


def _model_parameters(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        **candidate,
        "subsample": 0.9,
        "subsample_freq": 1,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def _fit_point_model(frame: pd.DataFrame, candidate: dict[str, Any]) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(objective="regression_l1", **_model_parameters(candidate))
    model.fit(frame.loc[:, FEATURE_NAMES], frame["target"])
    return model


def recursive_forecast(
    model: lgb.LGBMRegressor,
    observed_history: pd.Series,
    forecast_index: pd.DatetimeIndex,
) -> np.ndarray:
    history = observed_history.astype(float).copy()
    forecast = np.empty(len(forecast_index), dtype=float)
    for position, timestamp in enumerate(forecast_index):
        row = make_feature_row(history, timestamp)
        predictors = pd.DataFrame([row], columns=FEATURE_NAMES)
        value = float(model.predict(predictors)[0])
        forecast[position] = value
        history.loc[timestamp] = value
    return forecast


def residual_quantile_forecast(
    point_forecast: np.ndarray,
    validation_actual: pd.Series,
    validation_forecast: np.ndarray,
    forecast_index: pd.DatetimeIndex,
) -> dict[float, np.ndarray]:
    point = np.asarray(point_forecast, dtype=float)
    validation_point = np.asarray(validation_forecast, dtype=float)
    if point.shape != (len(forecast_index),):
        raise ValueError("Point forecast and forecast index lengths must match")
    if validation_point.shape != (len(validation_actual),):
        raise ValueError("Validation actual and forecast lengths must match")

    residuals = pd.Series(
        validation_actual.to_numpy(dtype=float) - validation_point,
        index=validation_actual.index,
    )
    residuals_by_hour = residuals.groupby(residuals.index.hour)
    quantiles: dict[float, np.ndarray] = {}
    for probability in QUANTILES:
        hourly_offsets = residuals_by_hour.quantile(probability).reindex(forecast_index.hour)
        if hourly_offsets.isna().any():
            raise ValueError("Validation residuals do not cover every forecast hour")
        quantiles[probability] = point + hourly_offsets.to_numpy(dtype=float)
    return quantiles


def run_lightgbm(
    train: pd.Series,
    validation: pd.Series,
    train_validation: pd.Series,
    test_index: pd.DatetimeIndex,
) -> ForecastResult:
    started = time.perf_counter()
    train_frame = build_training_frame(train)
    validation_records: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    best_validation_forecast: np.ndarray | None = None
    best_mape = float("inf")

    for candidate in CANDIDATES:
        model = _fit_point_model(train_frame, candidate)
        forecast = recursive_forecast(model, train, pd.DatetimeIndex(validation.index))
        candidate_mape = mape_pct(validation.to_numpy(), forecast)
        validation_records.append({**candidate, "validation_mape_pct": candidate_mape})
        if candidate_mape < best_mape:
            best_mape = candidate_mape
            best_candidate = candidate
            best_validation_forecast = forecast

    if best_candidate is None or best_validation_forecast is None:
        raise RuntimeError("LightGBM validation produced no candidate")

    final_frame = build_training_frame(train_validation)
    final_model = _fit_point_model(final_frame, best_candidate)
    point = recursive_forecast(final_model, train_validation, test_index)
    quantiles = residual_quantile_forecast(
        point,
        validation,
        best_validation_forecast,
        test_index,
    )
    gain = final_model.booster_.feature_importance(importance_type="gain")
    gain_total = float(gain.sum())
    if gain_total <= 0.0:
        raise RuntimeError("LightGBM produced no positive feature gain")
    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "gain_pct": gain / gain_total * 100.0,
        }
    )
    return ForecastResult(
        name="lightgbm",
        point=point,
        quantiles=quantiles,
        runtime_seconds=time.perf_counter() - started,
        hyperparameters={
            **_model_parameters(best_candidate),
            "objective_point": "regression_l1",
            "forecast_quantile_levels": list(QUANTILES),
            "interval_method": "validation_residual_quantiles_by_utc_hour",
            "interval_calibration_observations": len(validation),
            "interval_calibration_observations_per_hour": len(validation) // 24,
            "year_lag_hours": 8_736,
            "selection_metric": "validation_mape_pct",
            "validation_mape_pct": best_mape,
            "validation_forecast_mode": "recursive_without_observed_validation_updates",
            "test_forecast_mode": "recursive_without_observed_test_updates",
            "candidates": validation_records,
        },
        extras={"feature_importance": feature_importance},
    )
