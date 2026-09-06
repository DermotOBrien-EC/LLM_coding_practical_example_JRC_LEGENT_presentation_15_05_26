"""LightGBM on engineered features: the "feature work" entry.

The idea is different from the statistical models. Instead of describing the
series with one equation, we turn every hour into a row of a table (what
hour of the day is it, what day of the week, is it a holiday, what was the
load 24 hours ago, a week ago, a year ago, what has the last day and the last
week averaged) and let a gradient-boosted tree ensemble learn the mapping
from that row to the load.

Two honesty rules govern the feature table.

1. No feature may look forward. Every rolling average and standard deviation
   is computed over the window that *ends one hour before* the row it
   describes. A rolling mean that included the row's own value would be
   telling the model the answer.

2. The 168-hour forecast is produced recursively. On the first test hour the
   24-hour lag is a genuine observation; by the second test day it is the
   model's own earlier prediction. We simulate exactly that: predictions are
   appended to a working copy of the history and the features for the next
   hour are recomputed from it. Errors therefore compound, which is the
   honest cost of using short lags over a week-long horizon.

Prediction intervals come from three extra fits with LightGBM's quantile
objective at the 10th, 50th and 90th percentiles, plus the 2.5th and 97.5th
for the 95 percent band.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from common import (
    HORIZON,
    QUANTILE_LEVELS,
    SEED,
    TEST_END,
    TEST_START,
    TRAIN_END,
    VAL_END,
    ForecastResult,
    calendar_features,
    mape,
    validation_origins,
)

# Lags in hours. 8760 is a plain calendar year; 8736 is 364 days, which is a
# whole number of weeks, so it lands on the same weekday. Both are offered to
# the model and the feature-importance figure shows which one it prefers.
LAGS: tuple[int, ...] = (24, 168, 8736, 8760)
ROLLING_WINDOWS: tuple[int, ...] = (24, 168)

FEATURE_NAMES: tuple[str, ...] = (
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_public_holiday_de",
    "lag_24h",
    "lag_168h",
    "lag_8736h",
    "lag_8760h",
    "roll_mean_24h",
    "roll_std_24h",
    "roll_mean_168h",
    "roll_std_168h",
)

CATEGORICAL: tuple[str, ...] = ("hour", "day_of_week", "month")

# A small, deliberately coarse grid. The point of the sweep is to avoid
# obviously wrong settings, not to squeeze the last decimal out of the model.
CANDIDATE_PARAMS: tuple[dict[str, Any], ...] = (
    {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31},
    {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 63},
    {"n_estimators": 1500, "learning_rate": 0.03, "num_leaves": 63},
    {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 127},
    {"n_estimators": 1500, "learning_rate": 0.03, "num_leaves": 127},
)


def build_features(series: pd.Series) -> pd.DataFrame:
    """Turn the load series into the model's table, one row per hour.

    Rows whose lags reach back before the start of the series come out with
    missing values and are dropped by the caller.
    """
    frame = calendar_features(series.index)
    for lag in LAGS:
        frame[f"lag_{lag}h"] = series.shift(lag)
    for window in ROLLING_WINDOWS:
        shifted = series.shift(1)  # window ends at t-1: no peeking at y_t
        frame[f"roll_mean_{window}h"] = shifted.rolling(window).mean()
        frame[f"roll_std_{window}h"] = shifted.rolling(window).std()
    return frame[list(FEATURE_NAMES)]


def _row_features(history: np.ndarray, position: int, calendar_row: pd.Series) -> np.ndarray:
    """Feature vector for a single hour, given a working copy of the history.

    `history` holds observed load followed by whatever the model has already
    predicted; `position` is the index in that array of the hour we are about
    to forecast.
    """
    values = [
        calendar_row["hour"],
        calendar_row["day_of_week"],
        calendar_row["month"],
        calendar_row["is_weekend"],
        calendar_row["is_public_holiday_de"],
    ]
    for lag in LAGS:
        values.append(history[position - lag])
    for window in ROLLING_WINDOWS:
        block = history[position - window : position]
        values.append(block.mean())
        values.append(block.std(ddof=1))
    return np.asarray(values, dtype="float64")


def recursive_forecast(
    model: LGBMRegressor,
    series: pd.Series,
    origin: pd.Timestamp,
    horizon: int = HORIZON,
    companions: dict[float, LGBMRegressor] | None = None,
) -> tuple[pd.Series, dict[float, pd.Series]]:
    """Roll the model forward `horizon` hours from `origin`.

    The point model drives the recursion; any `companions` (the quantile
    models) are evaluated on exactly the same feature rows, so their bands are
    centred on the same trajectory the point forecast took.
    """
    history_series = series.loc[: origin - pd.Timedelta(hours=1)]
    n_history = len(history_series)
    buffer = np.concatenate([history_series.to_numpy(dtype="float64"), np.zeros(horizon)])

    index = pd.date_range(origin, periods=horizon, freq="h", tz="UTC")
    calendar = calendar_features(index)

    point = np.empty(horizon, dtype="float64")
    rows = np.empty((horizon, len(FEATURE_NAMES)), dtype="float64")

    for step in range(horizon):
        position = n_history + step
        row = _row_features(buffer, position, calendar.iloc[step])
        rows[step] = row
        frame = pd.DataFrame(row.reshape(1, -1), columns=list(FEATURE_NAMES))
        for column in CATEGORICAL:
            frame[column] = frame[column].astype("int16").astype("category")
        prediction = float(model.predict(frame)[0])
        point[step] = prediction
        buffer[position] = prediction

    quantile_out: dict[float, pd.Series] = {}
    if companions:
        table = pd.DataFrame(rows, columns=list(FEATURE_NAMES), index=index)
        for column in CATEGORICAL:
            table[column] = table[column].astype("int16").astype("category")
        for level, companion in companions.items():
            quantile_out[level] = pd.Series(companion.predict(table), index=index)

    return pd.Series(point, index=index), quantile_out


def _fit_point(x: pd.DataFrame, y: pd.Series, params: dict[str, Any]) -> LGBMRegressor:
    model = LGBMRegressor(
        objective="regression",
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    model.fit(x, y, categorical_feature=list(CATEGORICAL))
    return model


def _fit_quantile(
    x: pd.DataFrame, y: pd.Series, params: dict[str, Any], alpha: float
) -> LGBMRegressor:
    model = LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    model.fit(x, y, categorical_feature=list(CATEGORICAL))
    return model


def _as_categorical(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in CATEGORICAL:
        out[column] = out[column].astype("int16").astype("category")
    return out


def _select(
    series: pd.Series, features: pd.DataFrame
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score each candidate by mean MAPE over the thirteen validation weeks."""
    train_mask = features.index <= TRAIN_END
    x_train = _as_categorical(features.loc[train_mask])
    y_train = series.loc[x_train.index]

    origins = validation_origins()
    log: list[dict[str, Any]] = []

    for params in CANDIDATE_PARAMS:
        t0 = time.perf_counter()
        model = _fit_point(x_train, y_train, params)
        scores = []
        for origin in origins:
            forecast, _ = recursive_forecast(model, series, origin)
            actual = series.loc[origin : origin + pd.Timedelta(hours=HORIZON - 1)]
            scores.append(mape(actual.to_numpy(), forecast.to_numpy()))
        log.append(
            {
                **params,
                "val_mape_pct": float(np.mean(scores)),
                "fit_and_score_seconds": round(time.perf_counter() - t0, 1),
            }
        )

    best = min(log, key=lambda row: row["val_mape_pct"])
    chosen = {key: best[key] for key in CANDIDATE_PARAMS[0]}
    return chosen, log


def run(series: pd.Series) -> ForecastResult:
    """Select on validation, refit on train+validation, forecast the test week."""
    start = time.perf_counter()

    features = build_features(series).dropna()
    params, log = _select(series, features)

    # Refit on everything before the test window.
    final_mask = features.index <= VAL_END
    x_final = _as_categorical(features.loc[final_mask])
    y_final = series.loc[x_final.index]

    point_model = _fit_point(x_final, y_final, params)
    companions = {
        float(level): _fit_quantile(x_final, y_final, params, float(level))
        for level in QUANTILE_LEVELS
    }

    point, quantiles = recursive_forecast(
        point_model, series, TEST_START, HORIZON, companions=companions
    )
    if not point.index.equals(series.loc[TEST_START:TEST_END].index):
        raise ValueError("LightGBM forecast index does not match the test window")

    # Quantile fits are independent, so their curves can cross. Sorting them
    # per hour keeps the reported bands well-formed without changing any
    # individual model.
    levels = sorted(quantiles)
    stacked = np.sort(np.column_stack([quantiles[q].to_numpy() for q in levels]), axis=1)
    quantiles = {q: pd.Series(stacked[:, i], index=point.index) for i, q in enumerate(levels)}

    importance = pd.Series(
        point_model.booster_.feature_importance(importance_type="gain"),
        index=list(FEATURE_NAMES),
    ).sort_values(ascending=False)

    return ForecastResult(
        name="lightgbm",
        point=point,
        quantiles=quantiles,
        hyperparameters={
            **params,
            "features": list(FEATURE_NAMES),
            "categorical_features": list(CATEGORICAL),
            "forecast_strategy": "recursive, 168 one-step-ahead calls",
            "selection_metric": "mean MAPE over 13 held-out validation weeks",
            "interval_method": "separate LGBM quantile fits at 0.025/0.1/0.5/0.9/0.975",
        },
        runtime_seconds=time.perf_counter() - start,
        selection_log=log,
        extra={"feature_importance_gain": importance},
    )
