from __future__ import annotations

import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import (
    LOAD_COL,
    MODEL_LABELS,
    TEST_END,
    TEST_START,
    TIME_COL,
    TRAIN_END,
    VAL_END,
    VAL_START,
    ForecastResult,
    de_holiday_flags,
    mape_pct,
)

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
    {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 40},
    {"n_estimators": 700, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 60},
    {"n_estimators": 400, "learning_rate": 0.07, "num_leaves": 31, "min_child_samples": 80},
]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[[TIME_COL, LOAD_COL]].copy()
    idx = pd.DatetimeIndex(out[TIME_COL])
    out["hour"] = idx.hour
    out["day_of_week"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_public_holiday_de"] = de_holiday_flags(idx).to_numpy(dtype=int)
    out["lag_24h"] = out[LOAD_COL].shift(24)
    out["lag_168h"] = out[LOAD_COL].shift(168)
    out["lag_8736h"] = out[LOAD_COL].shift(8_736)
    shifted = out[LOAD_COL].shift(1)
    out["rolling_mean_24h"] = shifted.rolling(24).mean()
    out["rolling_std_24h"] = shifted.rolling(24).std()
    out["rolling_mean_168h"] = shifted.rolling(168).mean()
    out["rolling_std_168h"] = shifted.rolling(168).std()
    return out


def _fit(params: dict[str, Any], x: pd.DataFrame, y: pd.Series, objective: str = "regression", alpha: float | None = None) -> lgb.LGBMRegressor:
    kwargs: dict[str, Any] = {
        **params,
        "objective": objective,
        "random_state": 42,
        "verbosity": -1,
        "force_col_wise": True,
        "n_jobs": 4,
    }
    if alpha is not None:
        kwargs["alpha"] = alpha
    model = lgb.LGBMRegressor(**kwargs)
    model.fit(x, y)
    return model


def _select_hyperparameters(feats: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, float | str]]]:
    train_mask = (feats[TIME_COL] <= TRAIN_END) & feats[FEATURE_COLUMNS].notna().all(axis=1)
    val_mask = (feats[TIME_COL] >= VAL_START) & (feats[TIME_COL] <= VAL_END)
    x_train = feats.loc[train_mask, FEATURE_COLUMNS]
    y_train = feats.loc[train_mask, LOAD_COL]
    x_val = feats.loc[val_mask, FEATURE_COLUMNS]
    y_val = feats.loc[val_mask, LOAD_COL]
    scores: list[dict[str, float | str]] = []
    best_score = np.inf
    best_params = CANDIDATES[0]
    for i, params in enumerate(CANDIDATES):
        model = _fit(params, x_train, y_train)
        pred = model.predict(x_val)
        score = mape_pct(y_val, pred)
        scores.append({"candidate": str(i), "validation_mape_pct": float(score)})
        if score < best_score:
            best_score = score
            best_params = params
    return best_params, scores


def run(df: pd.DataFrame) -> ForecastResult:
    start = time.perf_counter()
    feats = make_features(df)
    best_params, scores = _select_hyperparameters(feats)
    train_val_mask = (feats[TIME_COL] <= VAL_END) & feats[FEATURE_COLUMNS].notna().all(axis=1)
    test_mask = (feats[TIME_COL] >= TEST_START) & (feats[TIME_COL] <= TEST_END)
    x_train_val = feats.loc[train_val_mask, FEATURE_COLUMNS]
    y_train_val = feats.loc[train_val_mask, LOAD_COL]
    x_test = feats.loc[test_mask, FEATURE_COLUMNS]
    test_index = pd.DatetimeIndex(feats.loc[test_mask, TIME_COL])

    mean_model = _fit(best_params, x_train_val, y_train_val)
    q10_model = _fit(best_params, x_train_val, y_train_val, objective="quantile", alpha=0.1)
    q50_model = _fit(best_params, x_train_val, y_train_val, objective="quantile", alpha=0.5)
    q90_model = _fit(best_params, x_train_val, y_train_val, objective="quantile", alpha=0.9)

    forecast = pd.Series(mean_model.predict(x_test), index=test_index, name="forecast")
    q10 = pd.Series(q10_model.predict(x_test), index=test_index, name="q10")
    q50 = pd.Series(q50_model.predict(x_test), index=test_index, name="q50")
    q90 = pd.Series(q90_model.predict(x_test), index=test_index, name="q90")
    q10, q90 = pd.concat([q10, q90], axis=1).min(axis=1), pd.concat([q10, q90], axis=1).max(axis=1)

    importances = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance_gain": mean_model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("importance_gain", ascending=False)

    return ForecastResult(
        name="lightgbm",
        display_name=MODEL_LABELS["lightgbm"],
        forecast=forecast,
        runtime_seconds=time.perf_counter() - start,
        hyperparameters={**best_params, "validation_scores": scores, "year_lag_hours": 8_736},
        q10=q10,
        q50=q50,
        q90=q90,
        lower80=q10,
        upper80=q90,
        lower95=q10 - (q90 - q10) * 0.46875,
        upper95=q90 + (q90 - q10) * 0.46875,
        fitted_model=mean_model,
        diagnostics={"feature_importance": importances},
    )
