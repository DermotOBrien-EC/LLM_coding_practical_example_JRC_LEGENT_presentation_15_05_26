from __future__ import annotations

from time import perf_counter
from typing import Any

import holidays
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .common import QUANTILES, SEED, TEST_INDEX, TRAIN_END, ForecastResult, mape, validation_blocks

FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_public_holiday_de",
    "lag_24h",
    "lag_168h",
    "lag_8760h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_mean_168h",
    "rolling_std_168h",
]
HOLIDAYS = holidays.Germany(years=range(2015, 2021))
CANDIDATES = [{"num_leaves": 15, "n_estimators": 250}, {"num_leaves": 31, "n_estimators": 400}]


def calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    return np.column_stack(
        [
            index.hour,
            index.dayofweek,
            index.month,
            (index.dayofweek >= 5).astype(int),
            [int(timestamp.date() in HOLIDAYS) for timestamp in index],
        ]
    ).astype(float)


def feature_frame(series: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(calendar_features(series.index), index=series.index, columns=FEATURES[:5])
    for lag in [24, 168, 8760]:
        frame[f"lag_{lag}h"] = series.shift(lag)
    past = series.shift(1)
    for window in [24, 168]:
        frame[f"rolling_mean_{window}h"] = past.rolling(window).mean()
        frame[f"rolling_std_{window}h"] = past.rolling(window).std(ddof=0)
    return frame[FEATURES]


def recursive_forecast(
    models: dict[float, Any], history: pd.Series, future: pd.DatetimeIndex
) -> np.ndarray:
    if future[0] != history.index[-1] + pd.Timedelta(hours=1):
        raise ValueError("Forecast must start immediately after observed history")
    if len(history) < 8760:
        raise ValueError("Insufficient annual lag history")
    if not future.equals(pd.date_range(future[0], periods=len(future), freq="h")):
        raise ValueError("Forecast timestamps must be hourly and contiguous")
    levels = sorted(models)
    median_column = levels.index(0.5)
    values = np.empty(len(history) + len(future), dtype=float)
    values[: len(history)] = history.to_numpy()
    calendar = calendar_features(future)
    output = np.empty((len(future), len(models)))
    for step in range(len(future)):
        position = len(history) + step
        last24, last168 = values[position - 24 : position], values[position - 168 : position]
        row = np.concatenate(
            [
                calendar[step],
                [
                    values[position - 24],
                    values[position - 168],
                    values[position - 8760],
                    last24.mean(),
                    last24.std(ddof=0),
                    last168.mean(),
                    last168.std(ddof=0),
                ],
            ]
        ).reshape(1, -1)
        predictions = [float(models[level].predict(row)[0]) for level in levels]
        output[step] = np.sort(predictions)
        # Only the predicted median enters any later hour's features.
        values[position] = output[step, median_column]
    return output


def fit_models(series: pd.Series, config: dict[str, int], levels: list[float]) -> dict[float, Any]:
    features = feature_frame(series)
    complete = features.notna().all(axis=1)
    x = features.loc[complete].to_numpy()
    y = series.loc[complete].to_numpy()
    models = {}
    for level in levels:
        model = LGBMRegressor(
            objective="quantile",
            alpha=level,
            learning_rate=0.05,
            min_child_samples=40,
            reg_lambda=1.0,
            verbosity=-1,
            random_state=SEED,
            n_jobs=4,
            deterministic=True,
            force_col_wise=True,
            **config,
        )
        model.fit(x, y)
        models[level] = model.booster_
    return models


def run(pretest: pd.Series) -> ForecastResult:
    start = perf_counter()
    train = pretest.loc[:TRAIN_END]
    records = []
    for config in CANDIDATES:
        models = fit_models(train, config, QUANTILES)
        actuals, predictions = [], []
        for history, target in validation_blocks(pretest):
            predictions.extend(recursive_forecast(models, history, target.index)[:, 2])
            actuals.extend(target.values)
        value = mape(np.array(actuals), np.array(predictions))
        records.append({"configuration": config, "mape_validation_pct": value})
        print(f"lightgbm validation {config}: {value:.4f}%", flush=True)
    selected = min(records, key=lambda record: record["mape_validation_pct"])["configuration"]
    final_models = fit_models(pretest, selected, QUANTILES)
    quantiles = recursive_forecast(final_models, pretest, TEST_INDEX)
    importance = final_models[0.5].feature_importance(importance_type="gain")
    return ForecastResult(
        name="lightgbm",
        point=quantiles[:, 2],
        quantiles=quantiles,
        runtime_seconds=perf_counter() - start,
        hyperparameters={
            **selected,
            "learning_rate": 0.05,
            "min_child_samples": 40,
            "reg_lambda": 1.0,
            "objective": "quantile",
            "quantiles": QUANTILES,
            "lag_hours": [24, 168, 8760],
            "rolling_ddof": 0,
            "seed": SEED,
            "forecast_strategy": "recursive median path, sorted marginal quantiles",
        },
        validation=records,
        diagnostics={
            "gain_importance": dict(zip(FEATURES, importance.tolist())),
            "training_rows_lost_to_lag_warmup": 8760,
            "interval_caveat": "One-step quantiles conditioned on recursive median features; lag uncertainty is not propagated.",
        },
    )
