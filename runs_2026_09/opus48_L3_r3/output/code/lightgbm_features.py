"""LightGBM on hand-built calendar and history features.

This is the machine-learning workhorse of the study. Instead of modelling time
directly, we turn each hour into a row of plain features and let a gradient-
boosted tree ensemble learn the mapping from features to load. The features are:

- Calendar: hour of day, day of week, month, weekend flag, German public
  holiday flag. These are all knowable from the clock alone.
- Lags of the load itself: one day ago, one week ago, and about one year ago
  (8760 hours). These carry the daily, weekly and yearly rhythm.
- Rolling summaries of the recent past: the mean and spread of the load over the
  previous 24 hours and the previous 168 hours. All computed strictly from hours
  before the one being predicted, so nothing leaks from the future.

Forecasting the test week is done step by step (recursively): we predict the
first hour, treat that prediction as if it were observed, use it to build the
next hour's day-ago lag and rolling features, and continue. This is the honest
way to produce a genuine week-ahead forecast, because the true values inside the
test week are never looked at.

Prediction intervals come from separate models trained with the quantile
objective at the levels we need.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import common as c

# The recent-history lengths (in hours) used for lag and rolling features.
LAG_24 = 24
LAG_168 = 168
LAG_8760 = 8760
ROLL_SHORT = 24
ROLL_LONG = 168

# The order of feature columns. Kept fixed so training and prediction agree.
FEATURE_NAMES: list[str] = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_public_holiday_de",
    "lag_24h",
    "lag_168h",
    "lag_8760h",
    "roll_mean_24",
    "roll_std_24",
    "roll_mean_168",
    "roll_std_168",
]

# A small grid of settings; the best on the validation week is kept.
PARAM_GRID: list[dict] = [
    {"n_estimators": 400, "num_leaves": 31, "learning_rate": 0.05},
    {"n_estimators": 700, "num_leaves": 63, "learning_rate": 0.05},
    {"n_estimators": 900, "num_leaves": 31, "learning_rate": 0.03},
]

# Base settings shared by every fit.
BASE_PARAMS: dict = {
    "random_state": c.SEED,
    "n_jobs": -1,
    "verbosity": -1,
}


def build_features(load: pd.Series) -> pd.DataFrame:
    """Turn a load series into the full feature table (one row per hour).

    Rolling and lag features use only hours strictly before each row, so a row's
    features never contain that row's own load value.
    """
    idx = load.index
    frame = pd.DataFrame(index=idx)
    frame["hour"] = idx.hour
    frame["day_of_week"] = idx.dayofweek
    frame["month"] = idx.month
    frame["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    frame["is_public_holiday_de"] = c.is_public_holiday_de(idx).astype(int)
    frame["lag_24h"] = load.shift(LAG_24)
    frame["lag_168h"] = load.shift(LAG_168)
    frame["lag_8760h"] = load.shift(LAG_8760)
    past = load.shift(1)  # everything up to the previous hour
    frame["roll_mean_24"] = past.rolling(ROLL_SHORT).mean()
    frame["roll_std_24"] = past.rolling(ROLL_SHORT).std()
    frame["roll_mean_168"] = past.rolling(ROLL_LONG).mean()
    frame["roll_std_168"] = past.rolling(ROLL_LONG).std()
    return frame[FEATURE_NAMES]


def _feature_row(hist: np.ndarray, stamp: pd.Timestamp) -> np.ndarray:
    """Build one feature row for `stamp` from the history array `hist`.

    `hist` holds the load values for every hour strictly before `stamp`, in time
    order. This mirrors `build_features` exactly so recursive prediction sees the
    same feature definitions the model was trained on.
    """
    return np.array(
        [
            stamp.hour,
            stamp.dayofweek,
            stamp.month,
            1.0 if stamp.dayofweek >= 5 else 0.0,
            1.0 if c.is_public_holiday_de(pd.DatetimeIndex([stamp]))[0] else 0.0,
            hist[-LAG_24],
            hist[-LAG_168],
            hist[-LAG_8760],
            hist[-ROLL_SHORT:].mean(),
            hist[-ROLL_SHORT:].std(ddof=1),
            hist[-ROLL_LONG:].mean(),
            hist[-ROLL_LONG:].std(ddof=1),
        ],
        dtype=float,
    )


def _recursive_forecast(
    point_model: LGBMRegressor,
    history: pd.Series,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast `steps` hours ahead recursively.

    Returns the point forecast and the matrix of feature rows used at each step
    (so quantile models can be applied to the exact same rows afterwards).
    """
    hist = history.to_numpy(dtype=float).copy()
    last = history.index[-1]
    preds = np.empty(steps, dtype=float)
    rows = np.empty((steps, len(FEATURE_NAMES)), dtype=float)
    for i in range(steps):
        stamp = last + pd.Timedelta(hours=i + 1)
        row = _feature_row(hist, stamp)
        rows[i] = row
        yhat = float(point_model.predict(row.reshape(1, -1))[0])
        preds[i] = yhat
        hist = np.append(hist, yhat)  # feed the prediction back in as the lag
    return preds, rows


def _fit_point(train_load: pd.Series, params: dict) -> LGBMRegressor:
    """Fit the main (least-squares) model that gives the point forecast."""
    feats = build_features(train_load).dropna()
    target = train_load.loc[feats.index]
    model = LGBMRegressor(**BASE_PARAMS, **params)
    model.fit(feats.to_numpy(), target.to_numpy())
    return model


def run(series: pd.Series) -> c.ModelResult:
    """Tune on validation, refit on train+val, recursively forecast the test week."""
    start = time.perf_counter()

    # --- Step 1: tune hyperparameters on the validation week. ---------------
    train_load = c.train_series(series)
    history_for_val = train_load  # ends at 2019-09-30 23:00
    val_target = c.val_series(series).iloc[: c.HORIZON].to_numpy()

    val_scores: dict[str, float] = {}
    best_params = PARAM_GRID[0]
    best_score = float("inf")
    for params in PARAM_GRID:
        model = _fit_point(train_load, params)
        fc, _ = _recursive_forecast(model, history_for_val, c.HORIZON)
        score = c.mape(val_target, fc)
        key = f"n{params['n_estimators']}_l{params['num_leaves']}_lr{params['learning_rate']}"
        val_scores[key] = round(score, 3)
        if score < best_score:
            best_score, best_params = score, params

    # --- Step 2: refit the point model on train+val. ------------------------
    tv_load = c.train_plus_val_series(series)
    point_model = _fit_point(tv_load, best_params)

    # --- Step 3: recursive point forecast of the test week. -----------------
    point, feat_rows = _recursive_forecast(point_model, tv_load, c.HORIZON)

    # --- Step 4: quantile models for prediction intervals. ------------------
    # Trained on the same features; applied to the feature rows built along the
    # point-forecast path so the intervals line up with the point forecast.
    feats = build_features(tv_load).dropna()
    target = tv_load.loc[feats.index]
    quantiles: dict[float, np.ndarray] = {}
    for q in c.WINNER_QUANTILES:
        qmodel = LGBMRegressor(
            **BASE_PARAMS, **best_params, objective="quantile", alpha=q
        )
        qmodel.fit(feats.to_numpy(), target.to_numpy())
        quantiles[q] = qmodel.predict(feat_rows).astype(float)

    # Keep the quantile bands from crossing each other after independent fits.
    stacked = np.sort(np.vstack([quantiles[q] for q in c.WINNER_QUANTILES]), axis=0)
    for i, q in enumerate(c.WINNER_QUANTILES):
        quantiles[q] = stacked[i]

    gain = point_model.booster_.feature_importance(importance_type="gain")
    importances = dict(zip(FEATURE_NAMES, gain.astype(float).tolist()))

    runtime = time.perf_counter() - start
    return c.ModelResult(
        name="lightgbm",
        point=point,
        runtime_seconds=runtime,
        hyperparameters={
            **best_params,
            "features": FEATURE_NAMES,
            "forecast_style": "recursive_multi_step",
            "selection_metric": "validation_mape_pct",
            "validation_mape_by_params": val_scores,
        },
        quantiles=quantiles,
        extra={"feature_importance_gain": importances},
    )
