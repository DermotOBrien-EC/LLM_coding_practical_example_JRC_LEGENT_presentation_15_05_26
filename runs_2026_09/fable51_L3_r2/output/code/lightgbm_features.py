"""LightGBM on engineered calendar, lag and rolling-window features.

The model is a gradient-boosted tree ensemble that learns "load at hour t
is a function of the clock, the calendar, the load one day / one week /
one year ago, and how the last day and week have been going". Because the
test horizon is a whole week but the most important lag is only 24 hours,
the forecast is built recursively: each predicted hour is appended to the
history and becomes the day-ago lag for the hour 24 steps later. Nothing
observed inside the forecast window is ever used.

Prediction intervals come from four extra models trained with LightGBM's
quantile objective; they read the same features along the point model's
recursive path.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from common import (
    HORIZON,
    QUANTILES,
    SEED,
    ForecastResult,
    Stopwatch,
    block_mape,
    calendar_features,
    log,
    sort_quantiles,
    validation_blocks,
)

CALENDAR: tuple[str, ...] = ("hour", "day_of_week", "month", "is_weekend", "is_public_holiday_de")
# 8,736 h = 52 weeks (same weekday one year ago); 8,760 h = 365 days (same
# calendar date one year ago, which is what matters for fixed-date holidays).
LAGS: dict[str, int] = {"lag_24h": 24, "lag_168h": 168, "lag_8736h": 8736, "lag_8760h": 8760}
ROLL_WINDOWS: tuple[int, ...] = (24, 168)
ROLL_FEATURES: tuple[str, ...] = tuple(
    f"roll_{stat}_{w}h" for w in ROLL_WINDOWS for stat in ("mean", "std")
)
FEATURES: tuple[str, ...] = CALENDAR + tuple(LAGS) + ROLL_FEATURES
MAX_LAG: int = max(LAGS.values())

FIXED_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "random_state": SEED,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}
SWEEP: dict[str, list[Any]] = {
    "objective": ["l2", "l1"],
    "num_leaves": [31, 63],
    "learning_rate": [0.03, 0.1],
}


def build_training_frame(series: pd.Series) -> pd.DataFrame:
    """Feature matrix plus target for every hour whose lags all exist.

    Rolling statistics are computed over the window that ends one hour
    before the row (hours t-w .. t-1), so no row ever sees its own value.
    """
    frame = calendar_features(series.index)
    for name, lag in LAGS.items():
        frame[name] = series.shift(lag)
    previous = series.shift(1)
    for w in ROLL_WINDOWS:
        frame[f"roll_mean_{w}h"] = previous.rolling(w).mean()
        frame[f"roll_std_{w}h"] = previous.rolling(w).std()
    frame["y"] = series
    return frame.dropna()


def fit_model(frame: pd.DataFrame, params: dict[str, Any]) -> LGBMRegressor:
    model = LGBMRegressor(**FIXED_PARAMS, **params)
    model.fit(frame[list(FEATURES)].to_numpy(dtype=float), frame["y"].to_numpy(dtype=float))
    return model


def recursive_forecast(
    models: dict[str, LGBMRegressor], history: pd.Series, index: pd.DatetimeIndex
) -> dict[str, pd.Series]:
    """Forecast `index` hour by hour, feeding the point forecast back as history.

    `models["point"]` drives the recursion; every other entry is predicted
    on the same feature rows (used for the quantile models).
    """
    n = len(index)
    values = np.concatenate([history.to_numpy(dtype=float), np.full(n, np.nan)])
    cal = calendar_features(index).to_numpy(dtype=float)
    preds = {key: np.empty(n) for key in models}
    offset = len(history)
    for h in range(n):
        pos = offset + h
        lag_vals = [values[pos - lag] for lag in LAGS.values()]
        roll_vals: list[float] = []
        for w in ROLL_WINDOWS:
            window = values[pos - w : pos]
            roll_vals.extend([float(window.mean()), float(window.std(ddof=1))])
        row = np.concatenate([cal[h], lag_vals, roll_vals])[None, :]
        for key, model in models.items():
            preds[key][h] = float(model.predict(row)[0])
        values[pos] = preds["point"][h]
    return {key: pd.Series(arr, index=index) for key, arr in preds.items()}


def run(train: pd.Series, val: pd.Series, test_index: pd.DatetimeIndex) -> ForecastResult:
    with Stopwatch() as sw:
        train_frame = build_training_frame(train)
        blocks = validation_blocks(val.index)
        history = pd.concat([train, val])

        # Stage 1: choose the tree settings on the 13 validation weeks.
        sweep_results: list[dict[str, Any]] = []
        for combo in itertools.product(*SWEEP.values()):
            params = dict(zip(SWEEP.keys(), combo))
            model = fit_model(train_frame, params)
            forecasts = [
                recursive_forecast(
                    {"point": model}, history.loc[: block[0] - pd.Timedelta(hours=1)], block
                )["point"]
                for block in blocks
            ]
            val_mape = block_mape(val, forecasts)
            sweep_results.append({**params, "val_mape_pct": round(val_mape, 4)})
            log(f"lightgbm sweep {params}: val MAPE {val_mape:.3f} %")
        best = min(sweep_results, key=lambda r: r["val_mape_pct"])
        chosen = {k: best[k] for k in SWEEP}

        # Stage 2: refit the chosen point model and the quantile models on
        # train + validation, then forecast the test week.
        full_frame = build_training_frame(history)
        models: dict[str, LGBMRegressor] = {"point": fit_model(full_frame, chosen)}
        for q in QUANTILES:
            if q == 0.5:
                continue
            models[f"q{q}"] = fit_model(full_frame, {**chosen, "objective": "quantile", "alpha": q})
        preds = recursive_forecast(models, history, test_index)
        quantiles = {q: preds[f"q{q}"] for q in QUANTILES if q != 0.5}
        quantiles[0.5] = preds["point"]
        quantiles = sort_quantiles(quantiles)

        importance = pd.Series(
            models["point"].booster_.feature_importance(importance_type="gain"),
            index=list(FEATURES),
        ).sort_values(ascending=False)

    return ForecastResult(
        name="lightgbm",
        point=preds["point"],
        quantiles=quantiles,
        runtime_seconds=sw.seconds,
        hyperparameters={
            **FIXED_PARAMS,
            **chosen,
            "features": list(FEATURES),
            "forecast_strategy": "recursive, hour by hour",
            "quantile_models": [q for q in QUANTILES if q != 0.5],
            "training_rows_train": int(len(train_frame)),
            "training_rows_train_val": int(len(full_frame)),
            "horizon_hours": HORIZON,
        },
        validation={"sweep": sweep_results, "chosen": chosen, "val_mape_pct": best["val_mape_pct"]},
        extras={"feature_importance_gain": importance},
    )
