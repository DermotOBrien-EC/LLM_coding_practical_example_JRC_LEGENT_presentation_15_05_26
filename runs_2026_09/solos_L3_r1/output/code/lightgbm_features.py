from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from common import (
    QUANTILES,
    SEED,
    DataSplits,
    ForecastResult,
    enforce_non_crossing_quantiles,
    german_holiday_indicator,
    mean_absolute_percentage_error,
)

FEATURE_NAMES = [
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
]

CANDIDATES: list[dict[str, Any]] = [
    {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "max_depth": -1,
    },
    {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 40,
        "max_depth": -1,
    },
    {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 40,
        "max_depth": -1,
    },
]


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour": index.hour.astype(np.int16),
            "day_of_week": index.dayofweek.astype(np.int8),
            "month": index.month.astype(np.int8),
            "is_weekend": (index.dayofweek >= 5).astype(np.int8),
            "is_public_holiday_de": german_holiday_indicator(index),
        },
        index=index,
    )


def make_training_frame(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    frame = _calendar_features(series.index)
    frame["lag_24h"] = series.shift(24)
    frame["lag_168h"] = series.shift(168)
    frame["lag_8736h"] = series.shift(24 * 364)
    shifted = series.shift(1)
    frame["rolling_mean_24h"] = shifted.rolling(24).mean()
    frame["rolling_std_24h"] = shifted.rolling(24).std(ddof=1)
    frame["rolling_mean_168h"] = shifted.rolling(168).mean()
    frame["rolling_std_168h"] = shifted.rolling(168).std(ddof=1)
    valid = frame.notna().all(axis=1)
    features = frame.loc[valid, FEATURE_NAMES].astype(float)
    target = series.loc[valid].astype(float)
    if len(features) != len(series) - 24 * 364:
        raise ValueError("The LightGBM training-frame history boundary is unexpected.")
    return features, target


def _make_model(
    configuration: dict[str, Any], objective: str = "regression_l1", alpha: float | None = None
) -> LGBMRegressor:
    parameters: dict[str, Any] = {
        **configuration,
        "objective": objective,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 0.1,
        "importance_type": "gain",
    }
    if alpha is not None:
        parameters["alpha"] = alpha
    return LGBMRegressor(**parameters)


def _recursive_forecast(
    model: LGBMRegressor,
    history: pd.Series,
    forecast_index: pd.DatetimeIndex,
) -> pd.Series:
    if history.index[-1] + pd.Timedelta(hours=1) != forecast_index[0]:
        raise ValueError("Recursive LightGBM history does not end at the forecast origin.")
    if len(history) < 24 * 364:
        raise ValueError("Recursive LightGBM history is shorter than its longest lag.")

    values = history.to_numpy(dtype=float).tolist()
    predictions = np.empty(len(forecast_index), dtype=float)
    calendar = _calendar_features(forecast_index)

    for position, timestamp in enumerate(forecast_index):
        last_24 = np.asarray(values[-24:], dtype=float)
        last_168 = np.asarray(values[-168:], dtype=float)
        row = pd.DataFrame(
            [
                {
                    "hour": float(calendar.at[timestamp, "hour"]),
                    "day_of_week": float(calendar.at[timestamp, "day_of_week"]),
                    "month": float(calendar.at[timestamp, "month"]),
                    "is_weekend": float(calendar.at[timestamp, "is_weekend"]),
                    "is_public_holiday_de": float(
                        calendar.at[timestamp, "is_public_holiday_de"]
                    ),
                    "lag_24h": values[-24],
                    "lag_168h": values[-168],
                    "lag_8736h": values[-24 * 364],
                    "rolling_mean_24h": float(last_24.mean()),
                    "rolling_std_24h": float(last_24.std(ddof=1)),
                    "rolling_mean_168h": float(last_168.mean()),
                    "rolling_std_168h": float(last_168.std(ddof=1)),
                }
            ],
            columns=FEATURE_NAMES,
        )
        prediction = float(model.predict(row)[0])
        predictions[position] = prediction
        values.append(prediction)

    return pd.Series(predictions, index=forecast_index, name="forecast")


def run(splits: DataSplits) -> ForecastResult:
    started = time.perf_counter()
    train_features, train_target = make_training_frame(splits.train)

    validation_scores: list[float] = []
    for configuration in CANDIDATES:
        candidate = _make_model(configuration)
        candidate.fit(train_features, train_target)
        validation_forecast = _recursive_forecast(
            candidate, splits.train, splits.validation.index
        )
        validation_scores.append(
            mean_absolute_percentage_error(splits.validation, validation_forecast)
        )

    best_position = int(np.argmin(validation_scores))
    selected = dict(CANDIDATES[best_position])

    final_features, final_target = make_training_frame(splits.train_validation)
    point_model = _make_model(selected)
    point_model.fit(final_features, final_target)
    point_forecast = _recursive_forecast(
        point_model, splits.train_validation, splits.test.index
    )

    raw_quantiles: dict[float, pd.Series] = {}
    for quantile in QUANTILES:
        quantile_model = _make_model(selected, objective="quantile", alpha=quantile)
        quantile_model.fit(final_features, final_target)
        raw_quantiles[quantile] = _recursive_forecast(
            quantile_model, splits.train_validation, splits.test.index
        )
    quantiles = enforce_non_crossing_quantiles(raw_quantiles, splits.test.index)

    gain = point_model.booster_.feature_importance(importance_type="gain")
    importance = pd.Series(gain, index=FEATURE_NAMES, dtype=float).sort_values(
        ascending=False
    )

    hyperparameters: dict[str, Any] = {
        **selected,
        "objective_point": "regression_l1",
        "quantile_objective_levels": list(QUANTILES),
        "year_lag_hours": 24 * 364,
        "forecast_protocol": "recursive from the forecast origin",
        "validation_mape_pct": validation_scores[best_position],
        "validation_candidates": [
            {**configuration, "validation_mape_pct": score}
            for configuration, score in zip(CANDIDATES, validation_scores, strict=True)
        ],
        "n_final_training_rows": len(final_features),
    }
    return ForecastResult(
        name="lightgbm",
        forecast=point_forecast,
        runtime_seconds=time.perf_counter() - started,
        hyperparameters=hyperparameters,
        quantiles=quantiles,
        feature_importance=importance,
    )
