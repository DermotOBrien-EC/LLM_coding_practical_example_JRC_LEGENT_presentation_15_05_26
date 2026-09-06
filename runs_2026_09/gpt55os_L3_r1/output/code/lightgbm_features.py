from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from common import ForecastResult, LOAD_COLUMN, SEED, elapsed_seconds, holiday_mask, mape_pct, timer

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_public_holiday_de",
    "lag_24h",
    "lag_168h",
    "lag_8736h",
    "roll_24h_mean",
    "roll_24h_std",
    "roll_168h_mean",
    "roll_168h_std",
]


def make_features(full: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=full.index)
    features["load"] = full[LOAD_COLUMN].astype(float)
    features["hour"] = features.index.hour
    features["day_of_week"] = features.index.dayofweek
    features["month"] = features.index.month
    features["is_weekend"] = (features.index.dayofweek >= 5).astype(int)
    features["is_public_holiday_de"] = holiday_mask(features.index).astype(int)
    shifted = features["load"].shift(1)
    features["lag_24h"] = features["load"].shift(24)
    features["lag_168h"] = features["load"].shift(168)
    features["lag_8736h"] = features["load"].shift(24 * 364)
    features["roll_24h_mean"] = shifted.rolling(24).mean()
    features["roll_24h_std"] = shifted.rolling(24).std()
    features["roll_168h_mean"] = shifted.rolling(168).mean()
    features["roll_168h_std"] = shifted.rolling(168).std()
    return features.dropna()


def _fit_model(params: dict[str, float | int], objective: str = "regression", alpha: float | None = None) -> LGBMRegressor:
    model_params: dict[str, float | int | str] = {
        "objective": objective,
        "random_state": SEED,
        "verbose": -1,
        "n_jobs": 4,
        "metric": "None",
        **params,
    }
    if alpha is not None:
        model_params["alpha"] = alpha
    return LGBMRegressor(**model_params)


def forecast(full: pd.DataFrame, train: pd.Series, validation: pd.Series, train_validation: pd.Series, test: pd.Series) -> ForecastResult:
    start = timer()
    frame = make_features(full.loc[: test.index.max(), [LOAD_COLUMN]])
    train_frame = frame.loc[train.index.min() : train.index.max()]
    val_frame = frame.loc[validation.index.min() : validation.index.max()]
    final_train_frame = frame.loc[train_validation.index.min() : train_validation.index.max()]
    test_frame = frame.loc[test.index.min() : test.index.max()]

    grid: list[dict[str, float | int]] = [
        {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 40, "subsample": 0.9, "colsample_bytree": 0.9},
        {"n_estimators": 700, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 60, "subsample": 0.9, "colsample_bytree": 0.85},
        {"n_estimators": 300, "learning_rate": 0.08, "num_leaves": 31, "min_child_samples": 80, "subsample": 0.8, "colsample_bytree": 0.8},
    ]
    best_params: dict[str, float | int] | None = None
    best_mape = float("inf")
    validation_scores: list[dict[str, float | int]] = []
    for params in grid:
        model = _fit_model(params)
        model.fit(train_frame[FEATURE_COLUMNS], train_frame["load"])
        pred = model.predict(val_frame[FEATURE_COLUMNS])
        score = mape_pct(val_frame["load"].to_numpy(), np.asarray(pred, dtype=float))
        validation_scores.append({**params, "validation_mape_pct": score})
        if score < best_mape:
            best_mape = score
            best_params = params
    if best_params is None:
        raise RuntimeError("LightGBM validation search produced no model")

    model = _fit_model(best_params)
    model.fit(final_train_frame[FEATURE_COLUMNS], final_train_frame["load"])
    point = np.asarray(model.predict(test_frame[FEATURE_COLUMNS]), dtype=float)

    quantiles: dict[float, np.ndarray] = {}
    for quantile in [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]:
        q_model = _fit_model(best_params, objective="quantile", alpha=quantile)
        q_model.fit(final_train_frame[FEATURE_COLUMNS], final_train_frame["load"])
        quantiles[quantile] = np.asarray(q_model.predict(test_frame[FEATURE_COLUMNS]), dtype=float)

    lower_ordered = np.minimum.reduce([quantiles[0.025], quantiles[0.1], quantiles[0.25], quantiles[0.5], quantiles[0.75], quantiles[0.9], quantiles[0.975]])
    upper_ordered = np.maximum.reduce([quantiles[0.025], quantiles[0.1], quantiles[0.25], quantiles[0.5], quantiles[0.75], quantiles[0.9], quantiles[0.975]])
    q025 = np.minimum(quantiles[0.025], quantiles[0.975])
    q975 = np.maximum(quantiles[0.025], quantiles[0.975])
    q10 = np.minimum(quantiles[0.1], quantiles[0.9])
    q90 = np.maximum(quantiles[0.1], quantiles[0.9])

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "gain": model.booster_.feature_importance(importance_type="gain"),
            "split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("gain", ascending=False)

    return ForecastResult(
        name="lightgbm",
        point=point,
        runtime_seconds=elapsed_seconds(start),
        hyperparameters={"selected": best_params, "validation_mape_pct": best_mape, "validation_grid": validation_scores},
        validation_mape_pct=best_mape,
        q025=q025,
        q10=q10,
        q25=np.maximum(q025, np.minimum(quantiles[0.25], q975)),
        q50=np.maximum(q025, np.minimum(quantiles[0.5], q975)),
        q75=np.maximum(q025, np.minimum(quantiles[0.75], q975)),
        q90=q90,
        q975=q975,
        feature_importance=importance,
        extra={"quantile_range_min": float(lower_ordered.min()), "quantile_range_max": float(upper_ordered.max())},
    )
