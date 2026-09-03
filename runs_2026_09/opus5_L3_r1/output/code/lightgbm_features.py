"""Gradient-boosted trees on hand-built calendar and lag features.

This is the "classic applied forecasting" entry: instead of modelling the
time series as a process, we turn every hour into a row of a table and let
LightGBM learn a regression from that row to the load.

The features are of three kinds:

1. Calendar: what hour of the day it is, which weekday, which month,
   whether it is a weekend, whether it is a German public holiday. All of
   these are knowable years in advance from a calendar alone.
2. Lags: the load 24 hours ago, 168 hours ago (same hour last week) and
   8736 hours ago (364 days, which is 52 whole weeks, so the same hour on
   the same weekday roughly one year earlier).
3. Rolling summaries: the mean and standard deviation of the load over the
   previous 24 and 168 hours. These are computed on a series shifted by one
   hour, so a row never sees its own value.

One thing matters more than any hyperparameter here. The test week is
forecast from a single origin: midnight on 1 January 2020, with no load
observations after that point. The 24-hour lag and the rolling windows for,
say, 5 January are therefore not available as observations. Filling them in
from the held-out actuals would be leakage and would flatter this model
enormously. Instead the model runs recursively: it predicts hour 1, appends
its own prediction to the history, builds the features for hour 2 from that
extended history, and so on for all 168 hours. Errors compound, which is
honest and is what a production pipeline running once a week would face.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import common as c

LAGS: tuple[int, ...] = (24, 168, 8736)  # hours: one day, one week, 52 weeks
ROLL_WINDOWS: tuple[int, ...] = (24, 168)
MIN_HISTORY: int = max(max(LAGS), max(ROLL_WINDOWS) + 1)

FEATURE_NAMES: list[str] = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_public_holiday_de",
    "lag_24h",
    "lag_168h",
    "lag_8736h",
    "roll_mean_24h",
    "roll_std_24h",
    "roll_mean_168h",
    "roll_std_168h",
]

# The grid searched on the validation window. Deliberately small: with
# 40,000 training rows and twelve features, boosted trees are not fussy, and
# a larger grid would mostly measure noise.
PARAM_GRID: list[dict[str, Any]] = [
    {"n_estimators": n, "learning_rate": lr, "num_leaves": leaves}
    for n in (400, 900)
    for lr in (0.03, 0.08)
    for leaves in (31, 63)
]

FIXED_PARAMS: dict[str, Any] = {
    "min_child_samples": 40,
    "subsample": 0.9,
    "subsample_freq": 1,
    "colsample_bytree": 0.9,
    "random_state": c.SEED,
    "n_jobs": -1,
    "verbose": -1,
}


def build_feature_table(series: pd.Series) -> pd.DataFrame:
    """Turn a load series into one row of features per hour.

    Rows whose lags reach back before the start of the series are dropped,
    so the table starts 8736 hours after the series does.
    """
    calendar = c.calendar_features(series.index)
    frame = calendar.copy()

    for lag in LAGS:
        frame[f"lag_{lag}h"] = series.shift(lag)

    shifted = series.shift(1)  # so a window never contains the current hour
    for window in ROLL_WINDOWS:
        frame[f"roll_mean_{window}h"] = shifted.rolling(window).mean()
        frame[f"roll_std_{window}h"] = shifted.rolling(window).std()

    frame["target"] = series
    return frame.dropna()


def _feature_row(history: np.ndarray, calendar_row: np.ndarray) -> np.ndarray:
    """Build one feature vector from a history array ending at timestamp - 1h."""
    values = [
        *calendar_row,
        history[-24],
        history[-168],
        history[-8736],
        history[-24:].mean(),
        history[-24:].std(ddof=1),
        history[-168:].mean(),
        history[-168:].std(ddof=1),
    ]
    return np.asarray(values, dtype=float)


def recursive_forecast(
    model: LGBMRegressor,
    history: pd.Series,
    index: pd.DatetimeIndex,
    side_models: dict[str, LGBMRegressor] | None = None,
) -> tuple[pd.Series, pd.DataFrame | None]:
    """Forecast forward one hour at a time, feeding predictions back in.

    `history` must end exactly one hour before `index` starts. `side_models`
    are extra models (the quantile fits) scored on the same feature rows the
    point model generates, so all of them see an identical view of the past.
    """
    if history.index[-1] + pd.Timedelta(hours=1) != index[0]:
        raise ValueError("History must end exactly one hour before the forecast starts.")
    if len(history) < MIN_HISTORY:
        raise ValueError("Not enough history to build the lag and rolling features.")

    calendar = c.calendar_features(index).to_numpy(dtype=float)
    extended = history.to_numpy(dtype=float).copy()

    rows: list[np.ndarray] = []
    predictions: list[float] = []
    for step in range(len(index)):
        row = _feature_row(extended, calendar[step])
        rows.append(row)
        frame = pd.DataFrame(row.reshape(1, -1), columns=FEATURE_NAMES)
        point = float(model.predict(frame)[0])
        predictions.append(point)
        extended = np.append(extended, point)

    point_forecast = pd.Series(predictions, index=index, name="lightgbm")

    quantiles: pd.DataFrame | None = None
    if side_models:
        matrix = pd.DataFrame(np.vstack(rows), columns=FEATURE_NAMES)
        quantiles = pd.DataFrame(
            {name: model_q.predict(matrix) for name, model_q in side_models.items()},
            index=index,
        )
        quantiles = pd.DataFrame(
            np.sort(quantiles.to_numpy(), axis=1), index=index, columns=list(side_models.keys())
        )

    return point_forecast, quantiles


def _fit(table: pd.DataFrame, params: dict[str, Any], objective: str | None = None, alpha: float | None = None) -> LGBMRegressor:
    kwargs = {**FIXED_PARAMS, **params}
    if objective is not None:
        kwargs["objective"] = objective
    if alpha is not None:
        kwargs["alpha"] = alpha
    model = LGBMRegressor(**kwargs)
    model.fit(table[FEATURE_NAMES], table["target"])
    return model


def select_hyperparameters(series: pd.Series) -> tuple[dict[str, Any], float, list[dict[str, Any]]]:
    """Fit on train only, then score each candidate on the validation window.

    Scoring mimics the real task: the validation quarter is forecast in
    back-to-back 168-hour blocks, each one started from observed data and
    then run recursively to the end of the block.
    """
    train = series.loc[: c.TRAIN_END]
    train_table = build_feature_table(train)

    trace: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any]] | None = None

    for params in PARAM_GRID:
        model = _fit(train_table, params)
        errors: list[np.ndarray] = []
        for origin in c.validation_origins():
            block = pd.date_range(origin, periods=c.HORIZON, freq="h", tz="UTC")
            history = series.loc[: origin - pd.Timedelta(hours=1)]
            forecast, _ = recursive_forecast(model, history, block)
            actual = series.loc[block]
            errors.append(np.abs((actual.to_numpy() - forecast.to_numpy()) / actual.to_numpy()))
        score = float(np.mean(np.concatenate(errors)) * 100.0)
        trace.append({**params, "validation_mape_pct": score})
        if best is None or score < best[0]:
            best = (score, params)

    assert best is not None
    return best[1], best[0], trace


def run(series: pd.Series) -> c.ForecastResult:
    """Select on validation, refit on train+validation, forecast the test week."""
    start = time.time()

    best_params, best_score, trace = select_hyperparameters(series)

    fitting_data = c.train_plus_val(series)
    table = build_feature_table(fitting_data)

    point_model = _fit(table, best_params)
    quantile_models = {
        str(q): _fit(table, best_params, objective="quantile", alpha=q) for q in c.QUANTILES
    }

    test = series.loc[c.TEST_START : c.TEST_END]
    forecast, quantiles = recursive_forecast(point_model, fitting_data, test.index, quantile_models)

    importance = pd.Series(
        point_model.booster_.feature_importance(importance_type="gain"), index=FEATURE_NAMES
    ).sort_values(ascending=False)

    return c.ForecastResult(
        name="lightgbm",
        point_forecast=forecast,
        runtime_seconds=time.time() - start,
        hyperparameters={
            **best_params,
            **{k: v for k, v in FIXED_PARAMS.items() if k not in ("n_jobs", "verbose")},
            "features": FEATURE_NAMES,
            "prediction_mode": "recursive (own predictions feed the 24h lag and rolling windows)",
            "quantile_objective": "separate LGBMRegressor per quantile",
        },
        quantile_forecast=quantiles,
        validation_mape_pct=best_score,
        extras={"feature_importance_gain": importance.to_dict(), "grid_search": trace},
    )


if __name__ == "__main__":
    result = run(c.load_load_series())
    result.save()
    print(result.name, result.validation_mape_pct, result.runtime_seconds)
