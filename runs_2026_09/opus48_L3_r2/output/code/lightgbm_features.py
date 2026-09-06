"""Gradient-boosted trees on hand-built calendar and lag features.

This is the "feature engineering" entry in the bake-off. Instead of modelling
the time series as a curve, we turn every hour into a row of numbers a
plain regressor can chew on:

- Calendar: hour of day, day of week, month, weekend flag, German public
  holiday flag. These say *when* we are.
- Lags of the load itself: one day ago, one week ago, one year ago. These
  carry the actual recent level and shape of demand.
- Rolling summaries: the mean and spread of the load over the previous 24 and
  168 hours. These say how high and how choppy demand has been lately.

Two honesty points that matter:

1. Rolling windows use only hours *before* the row's own hour, never the hour
   itself, so a feature can never peek at its own target.
2. When we forecast the test week we do it hour by hour, feeding the model's
   own prediction back in as the "one day ago" value for the next hour. If we
   instead used the true test-week loads as lags we would be leaking the
   answer. The one-week and one-year lags fall safely before the forecast
   start, so those stay exact.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import common as C

FEATURE_NAMES: list[str] = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_public_holiday_de",
    "lag_24h",
    "lag_168h",
    "lag_8760h",
    "roll_24h_mean",
    "roll_24h_std",
    "roll_168h_mean",
    "roll_168h_std",
]

MAX_LAG = 8760  # the longest lag we need before a row becomes usable


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Time-of-clock features. All knowable a priori, no external data."""
    hol = C.german_holiday_flags(index)
    dow = np.asarray(index.dayofweek)
    return pd.DataFrame(
        {
            "hour": np.asarray(index.hour, dtype=float),
            "day_of_week": dow.astype(float),
            "month": np.asarray(index.month, dtype=float),
            "is_weekend": (dow >= 5).astype(float),
            "is_public_holiday_de": hol,
        },
        index=index,
    )


def _build_training_matrix(hist: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Turn a contiguous load history into a (features, target) table.

    Rolling features use `.shift(1)` so the window ends at the previous hour;
    the current hour (the target) is never inside its own feature.
    """
    prev = hist.shift(1)
    feats = _calendar_features(hist.index)
    feats["lag_24h"] = hist.shift(24)
    feats["lag_168h"] = hist.shift(168)
    feats["lag_8760h"] = hist.shift(8760)
    feats["roll_24h_mean"] = prev.rolling(24).mean()
    feats["roll_24h_std"] = prev.rolling(24).std()
    feats["roll_168h_mean"] = prev.rolling(168).mean()
    feats["roll_168h_std"] = prev.rolling(168).std()

    feats = feats[FEATURE_NAMES]
    target = hist
    usable = feats.dropna().index
    return feats.loc[usable], target.loc[usable]


def _recursive_forecast(
    models: dict[str, LGBMRegressor],
    point_key: str,
    hist: pd.Series,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Forecast `horizon` hours ahead, one hour at a time.

    `models` maps a key to a fitted regressor. `point_key` names the model
    whose predictions are fed back to build the next hour's lag features; every
    other model (the quantile fits) is evaluated on the same feature row so the
    intervals stay consistent with the point path.
    """
    vals = hist.to_numpy(dtype=float).tolist()
    start = hist.index[0]
    step = pd.Timedelta(hours=1)
    out: dict[str, list[float]] = {k: [] for k in models}

    for s in range(horizon):
        pos = len(vals)
        t = start + pos * step
        cal = _calendar_features(pd.DatetimeIndex([t])).iloc[0]
        w24 = np.asarray(vals[pos - 24 : pos], dtype=float)
        w168 = np.asarray(vals[pos - 168 : pos], dtype=float)
        row = [
            cal["hour"],
            cal["day_of_week"],
            cal["month"],
            cal["is_weekend"],
            cal["is_public_holiday_de"],
            vals[pos - 24],
            vals[pos - 168],
            vals[pos - 8760],
            w24.mean(),
            w24.std(ddof=1),
            w168.mean(),
            w168.std(ddof=1),
        ]
        x = pd.DataFrame([row], columns=FEATURE_NAMES)
        preds = {k: float(m.predict(x)[0]) for k, m in models.items()}
        for k, v in preds.items():
            out[k].append(v)
        vals.append(preds[point_key])  # feed the point forecast forward

    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _fit_point_model(x: pd.DataFrame, y: pd.Series, params: dict) -> LGBMRegressor:
    model = LGBMRegressor(
        objective="regression",
        random_state=C.SEED,
        n_jobs=-1,
        verbose=-1,
        **params,
    )
    model.fit(x, y)
    return model


def run(bundle: C.DataBundle) -> C.ForecastResult:
    start = time.time()

    # --- Hyperparameter selection on the first validation week ---------------
    # Fit only on data up to 2019-09-30, forecast 2019-10-01..07, score MAPE.
    hist_train = C.slice_inclusive(bundle.full, C.TRAIN_START, C.TRAIN_END)
    x_tr, y_tr = _build_training_matrix(hist_train)

    grid = [
        {"n_estimators": 400, "num_leaves": 31, "learning_rate": 0.05},
        {"n_estimators": 800, "num_leaves": 31, "learning_rate": 0.05},
        {"n_estimators": 800, "num_leaves": 63, "learning_rate": 0.03},
        {"n_estimators": 1200, "num_leaves": 63, "learning_rate": 0.03},
    ]
    val_actual = bundle.val_select.to_numpy()
    best_params, best_val = None, np.inf
    for params in grid:
        m = _fit_point_model(x_tr, y_tr, params)
        fc = _recursive_forecast({"p": m}, "p", hist_train, C.HORIZON)["p"]
        score = C.mape(val_actual, fc)
        if score < best_val:
            best_val, best_params = score, params

    # --- Refit on Train+Validation, then forecast the test week --------------
    hist_full = C.slice_inclusive(bundle.full, C.TRAIN_START, C.VAL_END)
    x_all, y_all = _build_training_matrix(hist_full)

    point_model = _fit_point_model(x_all, y_all, best_params)
    models: dict[str, LGBMRegressor] = {"point": point_model}
    for q in C.QUANTILE_LEVELS:
        qm = LGBMRegressor(
            objective="quantile",
            alpha=q,
            random_state=C.SEED,
            n_jobs=-1,
            verbose=-1,
            **best_params,
        )
        qm.fit(x_all, y_all)
        models[f"q{q}"] = qm

    fc = _recursive_forecast(models, "point", hist_full, C.HORIZON)
    point = fc["point"]
    quantiles = {q: fc[f"q{q}"] for q in C.QUANTILE_LEVELS}

    # The quantile fits are independent, so a lower quantile can occasionally
    # cross above a higher one. Sorting each hour's quantiles removes that
    # nonsense without changing the point forecast.
    stacked = np.sort(np.vstack([quantiles[q] for q in C.QUANTILE_LEVELS]), axis=0)
    quantiles = {q: stacked[i] for i, q in enumerate(C.QUANTILE_LEVELS)}

    runtime = time.time() - start
    return C.ForecastResult(
        name="lightgbm",
        point=point,
        runtime_seconds=runtime,
        hyperparameters=best_params,
        val_mape=float(best_val),
        quantiles=quantiles,
        extra={
            "feature_importance_gain": {
                name: float(imp)
                for name, imp in zip(
                    FEATURE_NAMES,
                    point_model.booster_.feature_importance(importance_type="gain"),
                )
            }
        },
    )


if __name__ == "__main__":
    b = C.build_bundle()
    res = run(b)
    print("lightgbm best params:", res.hyperparameters, "val MAPE:", res.val_mape)
    print("lightgbm test MAPE:", C.mape(b.test.to_numpy(), res.point))
