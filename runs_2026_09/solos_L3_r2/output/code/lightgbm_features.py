from __future__ import annotations

from time import perf_counter
from typing import Any

import holidays
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from code.common import DataSplits, ForecastResult, mape_pct, make_utc_series

ANNUAL_LAG_HOURS = 364 * 24
FEATURE_COLUMNS = [
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
        "min_child_samples": 40,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 0.2,
    },
    {
        "n_estimators": 700,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 30,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 0.5,
    },
    {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 60,
        "subsample": 1.0,
        "colsample_bytree": 0.9,
        "reg_lambda": 0.5,
    },
]


def _holiday_calendar(index: pd.DatetimeIndex) -> holidays.HolidayBase:
    return holidays.Germany(years=range(int(index.year.min()), int(index.year.max()) + 1))


def _training_frame(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    index = series.index
    holiday_calendar = _holiday_calendar(index)
    frame = pd.DataFrame(index=index)
    frame["hour"] = index.hour
    frame["day_of_week"] = index.dayofweek
    frame["month"] = index.month
    frame["is_weekend"] = (index.dayofweek >= 5).astype(int)
    frame["is_public_holiday_de"] = [
        int(timestamp.date() in holiday_calendar) for timestamp in index
    ]
    frame["lag_24h"] = series.shift(24)
    frame["lag_168h"] = series.shift(168)
    frame["lag_8736h"] = series.shift(ANNUAL_LAG_HOURS)
    prior = series.shift(1)
    frame["rolling_mean_24h"] = prior.rolling(24, min_periods=24).mean()
    frame["rolling_std_24h"] = prior.rolling(24, min_periods=24).std(ddof=0)
    frame["rolling_mean_168h"] = prior.rolling(168, min_periods=168).mean()
    frame["rolling_std_168h"] = prior.rolling(168, min_periods=168).std(ddof=0)
    valid = frame.notna().all(axis=1)
    return frame.loc[valid, FEATURE_COLUMNS], series.loc[valid]


def _one_feature_row(
    timestamp: pd.Timestamp,
    history: list[float],
    holiday_calendar: holidays.HolidayBase,
) -> np.ndarray:
    if len(history) < ANNUAL_LAG_HOURS:
        raise ValueError("LightGBM history is too short for the annual lag")
    last_24 = np.asarray(history[-24:], dtype=float)
    last_168 = np.asarray(history[-168:], dtype=float)
    return np.asarray(
        [
            [
                timestamp.hour,
                timestamp.dayofweek,
                timestamp.month,
                int(timestamp.dayofweek >= 5),
                int(timestamp.date() in holiday_calendar),
                history[-24],
                history[-168],
                history[-ANNUAL_LAG_HOURS],
                float(last_24.mean()),
                float(last_24.std(ddof=0)),
                float(last_168.mean()),
                float(last_168.std(ddof=0)),
            ]
        ],
        dtype=float,
    )


def _recursive_forecast(
    model: LGBMRegressor,
    history_series: pd.Series,
    forecast_index: pd.DatetimeIndex,
) -> np.ndarray:
    history = history_series.astype(float).tolist()
    holiday_calendar = _holiday_calendar(history_series.index.append(forecast_index))
    predictions: list[float] = []
    booster = model.booster_
    for timestamp in forecast_index:
        row = _one_feature_row(timestamp, history, holiday_calendar)
        prediction = float(booster.predict(row, num_threads=1)[0])
        predictions.append(prediction)
        history.append(prediction)
    return np.asarray(predictions, dtype=float)


def _build_model(
    configuration: dict[str, Any], objective: str = "regression_l1", alpha: float | None = None
) -> LGBMRegressor:
    parameters = dict(configuration)
    parameters.update(
        {
            "objective": objective,
            "random_state": 2020,
            "n_jobs": 5,
            "verbosity": -1,
            "importance_type": "gain",
        }
    )
    if alpha is not None:
        parameters["alpha"] = alpha
    return LGBMRegressor(**parameters)


def _configuration_label(configuration: dict[str, Any]) -> str:
    return (
        f"leaves={configuration['num_leaves']},"
        f"trees={configuration['n_estimators']},"
        f"lr={configuration['learning_rate']}"
    )


def run(splits: DataSplits) -> ForecastResult:
    started = perf_counter()
    train_features, train_target = _training_frame(splits.train)
    validation_scores: dict[str, float] = {}
    successful: list[tuple[float, dict[str, Any]]] = []

    for configuration in CANDIDATES:
        model = _build_model(configuration)
        model.fit(train_features, train_target)
        validation_forecast = _recursive_forecast(
            model,
            splits.train,
            splits.validation.index,
        )
        score = mape_pct(splits.validation, validation_forecast)
        validation_scores[_configuration_label(configuration)] = score
        successful.append((score, configuration))

    validation_mape, chosen = min(successful, key=lambda item: item[0])
    combined_features, combined_target = _training_frame(splits.train_validation)
    point_model = _build_model(chosen)
    point_model.fit(combined_features, combined_target)
    point_values = _recursive_forecast(
        point_model,
        splits.train_validation,
        splits.test.index,
    )

    raw_quantiles: dict[float, np.ndarray] = {}
    for quantile in (0.025, 0.1, 0.5, 0.9, 0.975):
        quantile_model = _build_model(chosen, objective="quantile", alpha=quantile)
        quantile_model.fit(combined_features, combined_target)
        raw_quantiles[quantile] = _recursive_forecast(
            quantile_model,
            splits.train_validation,
            splits.test.index,
        )

    quantile_order = [0.025, 0.1, 0.5, 0.9, 0.975]
    raw_matrix = np.vstack([raw_quantiles[quantile] for quantile in quantile_order])
    crossing_count = int(np.sum(np.any(np.diff(raw_matrix, axis=0) < 0.0, axis=0)))
    ordered_matrix = np.sort(raw_matrix, axis=0)
    quantiles = {
        quantile: make_utc_series(
            ordered_matrix[position],
            splits.test.index,
            f"lightgbm_q{quantile}",
        )
        for position, quantile in enumerate(quantile_order)
    }

    gains = point_model.booster_.feature_importance(importance_type="gain")
    gain_total = float(gains.sum())
    normalized_gains = gains / gain_total * 100.0 if gain_total > 0.0 else gains
    feature_importance = pd.Series(
        normalized_gains,
        index=FEATURE_COLUMNS,
        name="gain_pct",
    ).sort_values(ascending=False)

    hyperparameters = dict(chosen)
    hyperparameters.update(
        {
            "objective": "regression_l1",
            "annual_lag_hours": ANNUAL_LAG_HOURS,
            "rolling_features_use_prior_values_only": True,
            "validation_and_test_forecasts_are_recursive": True,
            "candidate_validation_mape_pct": validation_scores,
            "quantile_objective_levels": quantile_order,
            "quantile_crossing_hours_before_sorting": crossing_count,
        }
    )
    return ForecastResult(
        name="lightgbm",
        forecast=make_utc_series(point_values, splits.test.index, "lightgbm"),
        runtime_seconds=perf_counter() - started,
        hyperparameters=hyperparameters,
        quantiles=quantiles,
        validation_mape_pct=validation_mape,
        feature_importance=feature_importance,
    )
