"""LightGBM regressor on hand-built features.

Features (all knowable a priori or derived from the load itself; no external
data):

  - hour of day, day of week, month, is_weekend, is_public_holiday_de
  - lag features from the load: t - 24 h, t - 168 h (one week), t - 8760 h
    (~one calendar year)
  - trailing 24 h and 168 h rolling mean and standard deviation of the load,
    computed only on data strictly before the row's timestamp

Hyperparameters (num_leaves, learning_rate, min_data_in_leaf, n_estimators)
are selected by validation MAPE on 2019-10-01..2019-12-31. The final model is
then refit on Train + Validation combined with the best set.

Multi-step inference is walk-forward: for a test hour t that is more than 24 h
past the forecast issue we substitute the model's own point forecast for the
missing observation, so no test-window truth ever leaks into the predictors.

For prediction intervals we fit three separate LGBM models with the pinball
(quantile) objective at q = 0.1, 0.5, 0.9. The 80 % interval is [q10, q90];
the 95 % interval is a Gaussian-approximation widening of that spread around
q50 (a proper conformal or joint-quantile interval is out of scope here).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import RANDOM_SEED, Split, mape

DE_HOLIDAYS = holidays.country_holidays("DE")


FEATURE_COLS: list[str] = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_public_holiday_de",
    "lag_24h",
    "lag_168h",
    "lag_8760h",
    "roll_mean_24h",
    "roll_std_24h",
    "roll_mean_168h",
    "roll_std_168h",
]


def _build_features(series: pd.Series) -> pd.DataFrame:
    """Build the feature matrix for every row in `series`.

    `series` is a UTC-indexed hourly load Series. Rolling and lag features use
    `.shift(...)` so a row cannot see its own load. Rows without enough
    history (first 8760 h) carry NaN in those columns.
    """
    df = pd.DataFrame({"load_mw": series.astype(float)})
    idx = df.index
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    dates = pd.Series(idx.date, index=idx)
    df["is_public_holiday_de"] = dates.map(lambda d: 1 if d in DE_HOLIDAYS else 0).astype(int)
    load = df["load_mw"]
    df["lag_24h"] = load.shift(24)
    df["lag_168h"] = load.shift(168)
    df["lag_8760h"] = load.shift(8760)
    shifted = load.shift(1)
    df["roll_mean_24h"] = shifted.rolling(window=24, min_periods=24).mean()
    df["roll_std_24h"] = shifted.rolling(window=24, min_periods=24).std()
    df["roll_mean_168h"] = shifted.rolling(window=168, min_periods=168).mean()
    df["roll_std_168h"] = shifted.rolling(window=168, min_periods=168).std()
    return df


@dataclass
class LGBResult:
    forecast: pd.Series
    lower_80: pd.Series
    upper_80: pd.Series
    lower_95: pd.Series
    upper_95: pd.Series
    runtime_seconds: float
    hyperparameters: dict[str, object]
    feature_importance: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def _fit_lgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, object],
    objective: str = "regression",
    alpha: float | None = None,
) -> lgb.LGBMRegressor:
    kw = dict(params)
    if objective == "quantile":
        kw["objective"] = "quantile"
        kw["alpha"] = alpha
    reg = lgb.LGBMRegressor(random_state=RANDOM_SEED, verbosity=-1, **kw)
    reg.fit(X_train, y_train)
    return reg


def _search_hyperparameters(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, object]:
    """Small ordered grid; deterministic, fast on this data (~40 k rows)."""
    grid: list[dict[str, object]] = []
    for n_leaves in (31, 63, 127):
        for lr in (0.05, 0.1):
            for n_est in (300, 600):
                grid.append({
                    "n_estimators": n_est,
                    "learning_rate": lr,
                    "num_leaves": n_leaves,
                    "min_child_samples": 20,
                    "subsample": 0.9,
                    "subsample_freq": 1,
                    "colsample_bytree": 0.9,
                })
    best_params: dict[str, object] | None = None
    best_mape = float("inf")
    for p in grid:
        model = _fit_lgb(X_tr, y_tr, p)
        pred = model.predict(X_val)
        m = mape(y_val.values, pred)
        if m < best_mape:
            best_mape = m
            best_params = dict(p)
    assert best_params is not None
    best_params["validation_mape_pct"] = best_mape
    return best_params


def _walk_forward(
    point_model: lgb.LGBMRegressor,
    q_models: dict[float, lgb.LGBMRegressor],
    known_series: pd.Series,
    test_index: pd.DatetimeIndex,
) -> tuple[pd.Series, dict[float, pd.Series]]:
    """Walk one hour at a time.

    `known_series` has the true load up to and INCLUDING the last hour before
    test_index[0]; test-window hours are NaN. After predicting hour t we
    overwrite that NaN with our own point forecast so lag_24h at t + 24 h
    (and later) reads the forecast instead of leaking truth.
    """
    working = known_series.copy()
    preds: list[float] = []
    q_preds: dict[float, list[float]] = {q: [] for q in q_models}
    for ts in test_index:
        feats = _build_features(working)
        row = feats.loc[[ts], FEATURE_COLS]
        yhat = float(point_model.predict(row)[0])
        preds.append(yhat)
        for q, model in q_models.items():
            q_preds[q].append(float(model.predict(row)[0]))
        working.loc[ts] = yhat
    point = pd.Series(preds, index=test_index, name="lightgbm")
    q_out = {q: pd.Series(v, index=test_index, name=f"q{q:.2f}") for q, v in q_preds.items()}
    return point, q_out


def run_lightgbm(split: Split) -> LGBResult:
    t0 = time.perf_counter()

    trainval = split.train_plus_val
    feats_all = _build_features(trainval).dropna(subset=FEATURE_COLS + ["load_mw"])
    tr_mask = feats_all.index <= split.train.index.max()
    val_mask = (feats_all.index >= split.val.index.min()) & (feats_all.index <= split.val.index.max())
    X_tr, y_tr = feats_all.loc[tr_mask, FEATURE_COLS], feats_all.loc[tr_mask, "load_mw"]
    X_val, y_val = feats_all.loc[val_mask, FEATURE_COLS], feats_all.loc[val_mask, "load_mw"]

    best_params = _search_hyperparameters(X_tr, y_tr, X_val, y_val)
    validation_mape = float(best_params.pop("validation_mape_pct"))

    X_full = feats_all[FEATURE_COLS]
    y_full = feats_all["load_mw"]
    point_model = _fit_lgb(X_full, y_full, best_params)
    q_models: dict[float, lgb.LGBMRegressor] = {}
    for q in (0.1, 0.5, 0.9):
        q_models[q] = _fit_lgb(X_full, y_full, best_params, objective="quantile", alpha=q)

    # Walk-forward inference. Give the walker the true history up to
    # test_index[0] - 1 h, then let it fill in its own predictions.
    known = split.full.copy()
    known.loc[split.test.index] = np.nan
    forecast, q_series = _walk_forward(point_model, q_models, known, split.test.index)

    q10, q50, q90 = q_series[0.1], q_series[0.5], q_series[0.9]
    lower_80, upper_80 = q10.rename("q10"), q90.rename("q90")
    half_80 = (q90 - q10) / 2.0
    scale = 1.9599639845400545 / 1.2815515655446004
    lower_95 = (q50 - scale * half_80).rename("q025")
    upper_95 = (q50 + scale * half_80).rename("q975")

    importance = pd.Series(point_model.feature_importances_, index=FEATURE_COLS, name="gain")

    runtime = time.perf_counter() - t0
    hp = dict(best_params)
    hp["validation_mape_pct"] = validation_mape
    return LGBResult(
        forecast=forecast,
        lower_80=lower_80,
        upper_80=upper_80,
        lower_95=lower_95,
        upper_95=upper_95,
        runtime_seconds=runtime,
        hyperparameters=hp,
        feature_importance=importance,
    )
