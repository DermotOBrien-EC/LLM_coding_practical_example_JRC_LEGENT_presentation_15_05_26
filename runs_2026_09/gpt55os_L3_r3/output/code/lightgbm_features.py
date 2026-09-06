from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from common import (
    FINAL_TRAIN_END,
    MODEL_LABELS,
    ModelOutput,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    mape_pct,
    recursive_forecast,
    supervised_feature_frame,
)

FEATURE_COLUMNS = [
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

CANDIDATES = [
    {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 30},
    {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 30},
    {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 50},
]


def make_model(params: dict[str, object], objective: str = "regression", alpha: float | None = None) -> LGBMRegressor:
    kwargs: dict[str, object] = {
        **params,
        "objective": objective,
        "random_state": 42,
        "verbose": -1,
        "force_col_wise": True,
        "n_jobs": 4,
    }
    if alpha is not None:
        kwargs["alpha"] = alpha
    return LGBMRegressor(**kwargs)


def select_params(series: pd.Series) -> dict[str, object]:
    frame = supervised_feature_frame(series.loc[:VAL_END])
    train_frame = frame.loc[TRAIN_START:TRAIN_END]
    history = series.loc[:TRAIN_END]
    val_index = pd.date_range(VAL_START, VAL_END, freq="h")
    best_params = CANDIDATES[0]
    best_score = float("inf")
    for params in CANDIDATES:
        model = make_model(params)
        model.fit(train_frame[FEATURE_COLUMNS], train_frame["load"])
        val_forecast = recursive_forecast(model, history, val_index)
        score = mape_pct(series.loc[VAL_START:VAL_END], val_forecast)
        if score < best_score:
            best_score = score
            best_params = params
    return {**best_params, "validation_mape_pct": best_score}


def run(series: pd.Series) -> ModelOutput:
    selected = select_params(series)
    params = {k: v for k, v in selected.items() if k != "validation_mape_pct"}
    final_frame = supervised_feature_frame(series.loc[:FINAL_TRAIN_END])
    final_train = final_frame.loc[TRAIN_START:FINAL_TRAIN_END]
    history = series.loc[:FINAL_TRAIN_END]
    test_index = pd.date_range(TEST_START, TEST_END, freq="h")

    point_model = make_model(params)
    point_model.fit(final_train[FEATURE_COLUMNS], final_train["load"])
    forecast = recursive_forecast(point_model, history, test_index)

    quantile_forecasts: dict[float, pd.Series] = {}
    for q in [0.1, 0.5, 0.9]:
        q_model = make_model(params, objective="quantile", alpha=q)
        q_model.fit(final_train[FEATURE_COLUMNS], final_train["load"])
        quantile_forecasts[q] = recursive_forecast(q_model, history, test_index)

    q10 = quantile_forecasts[0.1]
    q50 = quantile_forecasts[0.5]
    q90 = quantile_forecasts[0.9]
    lower_95 = forecast - 1.25 * (forecast - q10).abs()
    upper_95 = forecast + 1.25 * (q90 - forecast).abs()

    importance = pd.Series(
        point_model.booster_.feature_importance(importance_type="gain"),
        index=FEATURE_COLUMNS,
        name=f"{MODEL_LABELS['lightgbm']} gain",
    ).sort_values(ascending=False)

    return ModelOutput(
        name="lightgbm",
        forecast=forecast,
        runtime_seconds=0.0,
        hyperparameters=selected,
        lower_80=q10,
        upper_80=q90,
        lower_95=lower_95,
        upper_95=upper_95,
        q10=q10,
        q50=q50,
        q90=q90,
        feature_importance=importance,
        fitted_model=point_model,
    )
