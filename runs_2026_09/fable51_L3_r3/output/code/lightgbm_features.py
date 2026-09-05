"""LightGBM on engineered calendar, lag and rolling-window features.

The model is a gradient-boosted tree regressor. It sees the clock (hour,
weekday, month, weekend and German public-holiday flags), the load 24 h,
168 h and one year (364 days, so the weekday matches) earlier, and moving
averages and standard deviations over the previous 24 h and 168 h.

Forecasting a whole week ahead needs care: the 24 h lag and the rolling
windows for day two onwards would refer to hours that have not happened
yet. We therefore forecast recursively, feeding the model's own predictions
back in as if they were observations. That keeps the forecast an honest
168-hour-ahead forecast, made from the same origin as every other model.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import (
    HORIZON,
    QUANTILES,
    SEED,
    ForecastResult,
    Timer,
    calendar_features,
    rolling_validation_mape,
    validation_origins,
)

LAG_YEAR = 8736  # 364 days = 52 weeks, so the weekday lines up
CALENDAR = ["hour", "day_of_week", "month", "is_weekend", "is_public_holiday_de"]
LAGS = {"lag_24h": 24, "lag_168h": 168, "lag_8736h": LAG_YEAR}
ROLL_WINDOWS = {"24h": 24, "168h": 168}
FEATURES = CALENDAR + list(LAGS) + [
    f"roll_{stat}_{w}" for w in ROLL_WINDOWS for stat in ("mean", "std")
]

# Candidate settings scored on the validation window. Small on purpose: the
# tree count and leaf count are what matter most for this kind of data.
CANDIDATES: list[dict[str, Any]] = [
    {"num_leaves": leaves, "n_estimators": trees, "learning_rate": 0.05}
    for leaves in (31, 63, 127)
    for trees in (300, 800)
]
FIXED_PARAMS: dict[str, Any] = {
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "random_state": SEED,
    "verbose": -1,
    "n_jobs": 8,
}


def build_features(series: pd.Series) -> pd.DataFrame:
    """Feature table for every hour of `series` that has a full year of history.

    Rolling windows end one hour before the row's own timestamp, so the
    target value never leaks into its own features.
    """
    frame = calendar_features(series.index)
    for name, lag in LAGS.items():
        frame[name] = series.shift(lag)
    shifted = series.shift(1)
    for suffix, window in ROLL_WINDOWS.items():
        roll = shifted.rolling(window, min_periods=window)
        frame[f"roll_mean_{suffix}"] = roll.mean()
        frame[f"roll_std_{suffix}"] = roll.std()
    frame["target"] = series
    return frame.dropna()


def _features_for_hour(values: np.ndarray, pos: int, calendar_row: np.ndarray) -> np.ndarray:
    """Feature vector for position `pos` of an array that holds actuals and,
    beyond the forecast origin, the model's own earlier predictions."""
    lag_vals = [values[pos - lag] for lag in LAGS.values()]
    roll_vals: list[float] = []
    for window in ROLL_WINDOWS.values():
        chunk = values[pos - window : pos]
        roll_vals += [float(chunk.mean()), float(chunk.std(ddof=1))]
    return np.concatenate([calendar_row, lag_vals, roll_vals])


def recursive_forecast(
    model: lgb.LGBMRegressor, history: pd.Series, horizon: int
) -> tuple[pd.Series, pd.DataFrame]:
    """Forecast `horizon` hours after `history`, one hour at a time.

    Returns the point forecast and the feature matrix that produced it, so
    the quantile models can be applied to exactly the same inputs.
    """
    index = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    calendar = calendar_features(index)[CALENDAR].to_numpy(dtype=float)
    values = np.concatenate([history.to_numpy(dtype=float), np.full(horizon, np.nan)])
    n_hist = len(history)
    rows = np.empty((horizon, len(FEATURES)))
    for step in range(horizon):
        rows[step] = _features_for_hour(values, n_hist + step, calendar[step])
        values[n_hist + step] = model.booster_.predict(rows[step : step + 1], num_threads=1)[0]
    features = pd.DataFrame(rows, index=index, columns=FEATURES)
    point = pd.Series(values[n_hist:], index=index, name="lightgbm")
    return point, features


def fit_point_model(table: pd.DataFrame, params: dict[str, Any]) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(objective="regression", **params, **FIXED_PARAMS)
    model.fit(table[FEATURES], table["target"])
    return model


def fit_quantile_models(
    table: pd.DataFrame, params: dict[str, Any]
) -> dict[float, lgb.LGBMRegressor]:
    models: dict[float, lgb.LGBMRegressor] = {}
    for tau in QUANTILES:
        model = lgb.LGBMRegressor(objective="quantile", alpha=tau, **params, **FIXED_PARAMS)
        model.fit(table[FEATURES], table["target"])
        models[tau] = model
    return models


def run(train: pd.Series, val: pd.Series, horizon: int = HORIZON) -> ForecastResult:
    with Timer() as timer:
        # 1. Fit on Train only and score each candidate with rolling
        #    one-week-ahead forecasts across the validation window.
        origins = validation_origins(val, horizon)
        history = pd.concat([train, val])
        train_table = build_features(train)
        scores: list[tuple[float, dict[str, Any]]] = []
        for params in CANDIDATES:
            model = fit_point_model(train_table, params)
            forecasts = [
                recursive_forecast(model, history[: o - pd.Timedelta(hours=1)], horizon)[0]
                for o in origins
            ]
            score = rolling_validation_mape(val, forecasts, origins)
            scores.append((score, params))
            print(f"  lightgbm {params}: validation MAPE {score:.2f} %", flush=True)
        best_score, best_params = min(scores, key=lambda s: s[0])

        # 2. Refit on Train + Validation with the chosen setting.
        full_table = build_features(history)
        point_model = fit_point_model(full_table, best_params)
        quantile_models = fit_quantile_models(full_table, best_params)

        # 3. Forecast the test week recursively; quantiles use the same inputs.
        point, feature_rows = recursive_forecast(point_model, history, horizon)
        raw_q = np.column_stack([quantile_models[t].predict(feature_rows) for t in QUANTILES])
        raw_q.sort(axis=1)  # keep the quantiles in order hour by hour
        quantiles = {tau: pd.Series(raw_q[:, i], index=point.index) for i, tau in enumerate(QUANTILES)}

    importance = pd.Series(
        point_model.booster_.feature_importance(importance_type="gain"), index=FEATURES
    ).sort_values(ascending=False)
    return ForecastResult(
        name="lightgbm",
        point=point,
        quantiles=quantiles,
        runtime_seconds=timer.seconds,
        hyperparameters={
            **best_params,
            **{k: v for k, v in FIXED_PARAMS.items() if k not in ("verbose", "n_jobs")},
            "features": FEATURES,
            "forecast_strategy": "recursive (own predictions fed back for lags < 168 h)",
            "quantile_objectives": list(QUANTILES),
            "candidates_scored": len(CANDIDATES),
        },
        validation_mape_pct=best_score,
        extras={
            "feature_importance_gain": importance,
            "point_model": point_model,
            "validation_scores": scores,
        },
    )


if __name__ == "__main__":
    from common import load_series, split

    train, val, _ = split(load_series())
    result = run(train, val)
    print(f"validation MAPE {result.validation_mape_pct:.2f} %  runtime {result.runtime_seconds:.1f} s")
    print(result.extras["feature_importance_gain"])
