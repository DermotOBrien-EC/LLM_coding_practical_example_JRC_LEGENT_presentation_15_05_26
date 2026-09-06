"""LightGBM on engineered calendar-plus-lag features.

Features are built from the load series and the timestamp only. Rolling
windows use a `shift(1)` so no future value can leak into a row's own
features. Lags of 24 h, 168 h and 8760 h are always present inside the
training / validation / test windows because the earliest test hour is
2020-01-01 00:00 UTC and the training set reaches back to 2015-01-01.

We fit a mean-objective model for the point forecast and three quantile
models at 0.1, 0.5, 0.9 for prediction-interval bands and pinball loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import holidays as pyholidays
from lightgbm import LGBMRegressor

from common import Splits, TEST_START, VAL_START, align, mape


LAG_HOURS = [24, 168, 8760]
ROLL_WINDOWS = [24, 168]


def build_features(series: pd.Series) -> pd.DataFrame:
    """Build the full feature frame for every hour in `series`.

    The load column stays in the frame so we can align y with X after
    dropping rows whose lags fell outside the series.
    """
    df = pd.DataFrame({"y": series.astype("float64")})
    idx = df.index
    df["hour"] = idx.hour.astype("int16")
    df["day_of_week"] = idx.dayofweek.astype("int16")
    df["month"] = idx.month.astype("int16")
    df["is_weekend"] = (idx.dayofweek >= 5).astype("int8")

    years = sorted({idx.min().year, idx.max().year, *range(idx.min().year, idx.max().year + 1)})
    de_holidays = pyholidays.Germany(years=years)
    df["is_public_holiday_de"] = pd.Series(
        [1 if d.date() in de_holidays else 0 for d in idx],
        index=idx,
        dtype="int8",
    )

    for lag in LAG_HOURS:
        df[f"lag_{lag}h"] = df["y"].shift(lag)

    # Rolling means / stds computed on data strictly before each row.
    shifted = df["y"].shift(1)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}h"] = shifted.rolling(window=w, min_periods=w).mean()
        df[f"roll_std_{w}h"] = shifted.rolling(window=w, min_periods=w).std()

    return df


FEATURE_COLS: list[str] = (
    ["hour", "day_of_week", "month", "is_weekend", "is_public_holiday_de"]
    + [f"lag_{lag}h" for lag in LAG_HOURS]
    + [f"roll_mean_{w}h" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}h" for w in ROLL_WINDOWS]
)


def _fit_predict(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_pred: pd.DataFrame,
    params: dict,
    objective: str = "regression",
    alpha: float | None = None,
) -> np.ndarray:
    kwargs = dict(params)
    kwargs["objective"] = objective
    if alpha is not None:
        kwargs["alpha"] = alpha
    model = LGBMRegressor(**kwargs)
    model.fit(X_fit, y_fit)
    return model.predict(X_pred)


def run(splits: Splits) -> dict[str, object]:
    """Sweep num_leaves on the validation window, refit, predict test."""
    full_series = pd.concat([splits.trainval, splits.test])
    feats = build_features(full_series).dropna()

    train_mask = feats.index < VAL_START
    val_mask = (feats.index >= VAL_START) & (feats.index < TEST_START)
    test_mask = feats.index >= TEST_START

    X_train = feats.loc[train_mask, FEATURE_COLS]
    y_train = feats.loc[train_mask, "y"]
    X_val = feats.loc[val_mask, FEATURE_COLS]
    y_val = feats.loc[val_mask, "y"]
    X_test = feats.loc[test_mask, FEATURE_COLS]
    y_test = feats.loc[test_mask, "y"]

    base = dict(
        n_estimators=800,
        learning_rate=0.05,
        min_child_samples=30,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    val_scores: dict[int, float] = {}
    for num_leaves in (31, 63, 127):
        params = dict(base, num_leaves=num_leaves)
        pred_val = _fit_predict(X_train, y_train, X_val, params)
        val_scores[num_leaves] = mape(y_val.values, pred_val)

    best_leaves = min(val_scores, key=val_scores.get)

    X_refit = pd.concat([X_train, X_val])
    y_refit = pd.concat([y_train, y_val])
    params = dict(base, num_leaves=best_leaves)

    point_test = _fit_predict(X_refit, y_refit, X_test, params)

    q10 = _fit_predict(X_refit, y_refit, X_test, params, "quantile", 0.10)
    q50 = _fit_predict(X_refit, y_refit, X_test, params, "quantile", 0.50)
    q90 = _fit_predict(X_refit, y_refit, X_test, params, "quantile", 0.90)
    q025 = _fit_predict(X_refit, y_refit, X_test, params, "quantile", 0.025)
    q975 = _fit_predict(X_refit, y_refit, X_test, params, "quantile", 0.975)

    # Fit one final mean model to expose feature importances.
    imp_model = LGBMRegressor(**params)
    imp_model.fit(X_refit, y_refit)
    importances = pd.Series(
        imp_model.booster_.feature_importance(importance_type="gain"),
        index=X_refit.columns,
    ).sort_values(ascending=False)

    forecast = align(point_test, splits.test.index)
    return {
        "name": "lightgbm",
        "forecast": forecast,
        "lower_80": align(q10, splits.test.index),
        "upper_80": align(q90, splits.test.index),
        "lower_95": align(q025, splits.test.index),
        "upper_95": align(q975, splits.test.index),
        "quantiles": {
            "0.1": align(q10, splits.test.index),
            "0.5": align(q50, splits.test.index),
            "0.9": align(q90, splits.test.index),
        },
        "feature_importances": importances,
        "hyperparameters": {
            "num_leaves": best_leaves,
            "learning_rate": 0.05,
            "n_estimators": 800,
            "min_child_samples": 30,
            "features": FEATURE_COLS,
            "validation_mape_pct_by_num_leaves": {
                str(k): v for k, v in val_scores.items()
            },
        },
    }
