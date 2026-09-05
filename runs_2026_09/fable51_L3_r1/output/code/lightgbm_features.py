"""LightGBM on engineered calendar, lag and rolling features.

The model is a gradient-boosted tree ensemble that predicts the load at
hour t from twelve numbers: five calendar facts about t (hour, weekday,
month, weekend flag, German holiday flag), three earlier loads (24 h,
168 h and 364 days before t) and four summaries of the recent past (mean
and standard deviation of the last 24 h and of the last 168 h, ending at
t - 1 h).

Forecasting a whole week ahead with a 24-hour lag needs care: for the
third test day the "load 24 hours earlier" is itself a test-window value
that a real forecaster would not have. The model is therefore run
recursively: hour by hour, each prediction is appended to the history and
becomes the lag and rolling input for the hours after it. Nothing from the
test window is ever read.

Intervals come from five extra models trained with the quantile objective
(0.025, 0.1, 0.5, 0.9, 0.975), evaluated on the same recursive feature
rows as the point model.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import common

LAG_YEAR: int = 364 * 24  # 364 days keeps the weekday aligned (8,736 h; the task's ~8,760 h)
FEATURES: list[str] = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_public_holiday_de",
    "lag_24h",
    "lag_168h",
    "lag_364d",
    "roll_mean_24h",
    "roll_std_24h",
    "roll_mean_168h",
    "roll_std_168h",
]

BASE_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": common.SEED,
    "verbose": -1,
}
CANDIDATES: list[dict[str, Any]] = [
    {"num_leaves": 31, "n_estimators": 400},
    {"num_leaves": 31, "n_estimators": 1200},
    {"num_leaves": 127, "n_estimators": 400},
    {"num_leaves": 127, "n_estimators": 1200},
]


def build_features(series: pd.Series) -> pd.DataFrame:
    """Feature table for every hour of `series` (rows lacking a lag are NaN)."""
    out = common.calendar_features(series.index)
    out["lag_24h"] = series.shift(24)
    out["lag_168h"] = series.shift(168)
    out["lag_364d"] = series.shift(LAG_YEAR)
    prev = series.shift(1)  # strictly before t: the load at t is the target
    out["roll_mean_24h"] = prev.rolling(24).mean()
    out["roll_std_24h"] = prev.rolling(24).std()
    out["roll_mean_168h"] = prev.rolling(168).mean()
    out["roll_std_168h"] = prev.rolling(168).std()
    return out[FEATURES]


def training_frame(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    x = build_features(series)
    mask = x.notna().all(axis=1)
    return x.loc[mask], series.loc[mask]


def recursive_forecast(model: LGBMRegressor, history: pd.Series, origin: pd.Timestamp, horizon: int = common.HORIZON) -> tuple[pd.Series, pd.DataFrame]:
    """Forecast `horizon` hours from `origin`, feeding predictions back in.

    Returns the point forecast and the feature rows that produced it (the
    quantile models are evaluated on those same rows).
    """
    idx = common.horizon_index(origin, horizon)
    cal = common.calendar_features(idx).to_numpy(dtype=float)
    values = history.to_numpy(dtype=float)
    n_hist = len(values)
    buf = np.concatenate([values, np.full(horizon, np.nan)])
    rows = np.empty((horizon, len(FEATURES)))
    for h in range(horizon):
        i = n_hist + h
        last24 = buf[i - 24 : i]
        last168 = buf[i - 168 : i]
        rows[h, :5] = cal[h]
        rows[h, 5] = buf[i - 24]
        rows[h, 6] = buf[i - 168]
        rows[h, 7] = buf[i - LAG_YEAR]
        rows[h, 8] = last24.mean()
        rows[h, 9] = last24.std(ddof=1)
        rows[h, 10] = last168.mean()
        rows[h, 11] = last168.std(ddof=1)
        buf[i] = model.predict(rows[h : h + 1])[0]
    features = pd.DataFrame(rows, index=idx, columns=FEATURES)
    return pd.Series(buf[n_hist:], index=idx), features


def fit_point_model(x: pd.DataFrame, y: pd.Series, cand: dict[str, Any]) -> LGBMRegressor:
    model = LGBMRegressor(objective="regression", **BASE_PARAMS, **cand)
    model.fit(x, y)
    return model


def run(s: pd.Series) -> common.ModelResult:
    t0 = time.perf_counter()
    x_train, y_train = training_frame(common.train_slice(s))
    table: list[dict[str, Any]] = []
    for cand in CANDIDATES:
        tc = time.perf_counter()
        model = fit_point_model(x_train, y_train, cand)
        val_mape, rows = common.rolling_origin_mape(s, lambda hist, origin: recursive_forecast(model, hist, origin)[0])
        row = {**cand, "val_mape_pct": val_mape, "val_mape_by_origin_pct": [round(float(r["mape_pct"]), 3) for r in rows], "fit_seconds": round(time.perf_counter() - tc, 1)}
        table.append(row)
        print(f"lightgbm: {cand} val MAPE {val_mape:.2f} % ({row['fit_seconds']} s)")
    table = sorted(table, key=lambda r: r["val_mape_pct"])
    chosen = {k: table[0][k] for k in CANDIDATES[0]}

    # Refit on Train + Validation, then the recursive test-week forecast.
    x_all, y_all = training_frame(common.train_val_slice(s))
    point_model = fit_point_model(x_all, y_all, chosen)
    history = common.history_before(s, common.TEST_START)
    point, feature_rows = recursive_forecast(point_model, history, common.TEST_START)

    quantiles: dict[float, pd.Series] = {}
    for q in common.QUANTILES:
        qm = LGBMRegressor(objective="quantile", alpha=q, **BASE_PARAMS, **chosen)
        qm.fit(x_all, y_all)
        quantiles[q] = pd.Series(qm.predict(feature_rows), index=point.index)
    quantiles = common.sort_quantiles(quantiles)

    importance = pd.Series(point_model.booster_.feature_importance(importance_type="gain"), index=FEATURES)
    return common.ModelResult(
        name="lightgbm",
        point=point,
        quantiles=quantiles,
        hyperparameters={
            **{k: v for k, v in BASE_PARAMS.items() if k != "verbose"},
            **chosen,
            "features": FEATURES,
            "forecast_strategy": "recursive (predictions fed back as lags)",
            "n_training_rows": int(len(x_all)),
            "n_candidates": len(CANDIDATES),
        },
        runtime_seconds=time.perf_counter() - t0,
        validation_mape_pct=table[0]["val_mape_pct"],
        validation_table=table,
        notes="Point forecast from the squared-error model; intervals from five quantile-objective models on the same recursive feature rows.",
        extra={"feature_importance_gain": importance.sort_values(ascending=False).round(1).to_dict()},
    )


if __name__ == "__main__":
    common.silence_warnings()
    s = common.load_series()
    result = run(s)
    result.save()
    print(f"lightgbm: test MAPE {common.mape(common.test_slice(s), result.point):.2f} %, runtime {result.runtime_seconds:.0f} s")
