"""Gradient-boosted trees on hand-built calendar and lag features.

This is the "feature engineering" entry in the bake-off. Instead of letting a
model discover temporal structure on its own, we hand LightGBM the structure
directly as columns: what hour it is, what day of the week, whether it is a
weekend or a German public holiday, and what the load was 24 hours, one week,
and one year ago, plus short rolling summaries of recent load.

One subtlety decides whether this is an honest test. When we forecast the test
week, the "load 24 hours ago" feature for, say, Jan 3rd noon would be Jan 2nd
noon, which is itself inside the test window and therefore NOT something we
would know in a real day-ahead forecast. To avoid quietly feeding the model
the answer, we forecast the test week recursively: we predict hour by hour and
feed our own predictions back in as the lag/rolling inputs whenever the lookup
falls inside the forecast horizon. The one-week and one-year lags for the whole
test week fall before it, so those always use real observations.

Prediction intervals come from separate LightGBM fits with the "quantile"
objective, evaluated along the same recursive feature path as the point model.
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

import common as c

# Feature columns, in a fixed order the model and importances both rely on.
FEATURE_NAMES: list[str] = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "lag_24h", "lag_168h", "lag_8760h",
    "roll24_mean", "roll24_std", "roll168_mean", "roll168_std",
]

LAGS: dict[str, int] = {"lag_24h": 24, "lag_168h": 168, "lag_8760h": 8760}
ROLL_WINDOWS: list[int] = [24, 168]
# The earliest position for which every lag/rolling feature exists.
MIN_POS: int = max(max(LAGS.values()), max(ROLL_WINDOWS))  # 8760

QUANTILE_LEVELS: list[float] = [0.025, 0.1, 0.5, 0.9, 0.975]

# A small, sensible grid. The point is to genuinely pick by validation error,
# not to chase the last decimal of accuracy.
PARAM_GRID: list[dict] = [
    {"num_leaves": 31, "n_estimators": 300, "learning_rate": 0.05},
    {"num_leaves": 63, "n_estimators": 500, "learning_rate": 0.05},
    {"num_leaves": 31, "n_estimators": 600, "learning_rate": 0.03},
    {"num_leaves": 127, "n_estimators": 400, "learning_rate": 0.05},
]

_FIXED_PARAMS: dict = {
    "objective": "regression",
    "min_child_samples": 20,
    "subsample": 1.0,
    "n_jobs": -1,
    "random_state": c.SEED,
    "verbosity": -1,
}


def _calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    """The five a-priori calendar columns for a set of timestamps."""
    hour = index.hour.to_numpy()
    dow = index.dayofweek.to_numpy()
    month = index.month.to_numpy()
    is_weekend = (dow >= 5).astype(float)
    is_holiday = c.is_public_holiday_de(index).astype(float)
    return np.column_stack([hour, dow, month, is_weekend, is_holiday]).astype(float)


def _build_training_matrix(
    actual: np.ndarray, time_index: pd.DatetimeIndex, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised feature matrix and target for a set of training positions.

    Uses only real observations (this is training data), so it can be built in
    one shot with array indexing rather than the hour-by-hour recursion used
    for forecasting.
    """
    cal = _calendar_features(time_index[positions])
    lag_cols = [actual[positions - LAGS[name]] for name in LAGS]
    roll_cols: list[np.ndarray] = []
    for w in ROLL_WINDOWS:
        means = np.empty(len(positions))
        stds = np.empty(len(positions))
        for i, p in enumerate(positions):
            window = actual[p - w:p]  # the w hours strictly before p
            means[i] = window.mean()
            stds[i] = window.std(ddof=1)
        roll_cols.append(means)
        roll_cols.append(stds)
    features = np.column_stack([cal] + lag_cols + roll_cols)
    target = actual[positions]
    return features, target


def _features_at(work: np.ndarray, time_index: pd.DatetimeIndex, pos: int) -> np.ndarray:
    """One feature row at position `pos`, reading only earlier values.

    `work` holds real observations up to the forecast origin and our own
    predictions after it. Every lookup here points strictly backwards, so the
    true (unseen) test values are never touched.
    """
    ts = time_index[pos:pos + 1]
    cal = _calendar_features(ts)[0]
    lags = [work[pos - LAGS[name]] for name in LAGS]
    rolls: list[float] = []
    for w in ROLL_WINDOWS:
        window = work[pos - w:pos]
        rolls.append(window.mean())
        rolls.append(window.std(ddof=1))
    row = np.concatenate([cal, lags, rolls])
    return row


def _recursive_forecast(
    model: lgb.LGBMRegressor,
    actual: np.ndarray,
    time_index: pd.DatetimeIndex,
    origin_pos: int,
    horizon_index: pd.DatetimeIndex,
) -> tuple[pd.Series, np.ndarray]:
    """Predict `horizon_index` one hour at a time, feeding predictions back.

    Returns the point forecast and the matrix of feature rows actually used,
    so the quantile models can be scored along the identical path.
    """
    work = actual.copy()
    horizon_positions = [time_index.get_loc(t) for t in horizon_index]
    # Blank out the horizon so an accidental forward read fails loudly.
    for p in horizon_positions:
        work[p] = np.nan
    rows = np.empty((len(horizon_positions), len(FEATURE_NAMES)))
    preds = np.empty(len(horizon_positions))
    for i, p in enumerate(horizon_positions):
        row = _features_at(work, time_index, p)
        assert not np.isnan(row).any(), f"feature NaN at {time_index[p]}"
        yhat = float(model.predict(row.reshape(1, -1))[0])
        work[p] = yhat
        rows[i] = row
        preds[i] = yhat
    point = pd.Series(preds, index=horizon_index, name="lightgbm")
    return point, rows


def _fit_point(params: dict, features: np.ndarray, target: np.ndarray) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**{**_FIXED_PARAMS, **params})
    model.fit(features, target)
    return model


def run(splits: c.Splits) -> c.ModelResult:
    start = time.time()

    full = splits.full
    origin_start = full.index[0]
    # Continuous hourly axis from the first observation to the end of the test
    # week. The data has no gaps, so positions line up one-to-one with hours.
    time_index = pd.date_range(origin_start, c.TEST_END, freq="h")
    actual = np.full(len(time_index), np.nan)
    # Fill in every real observation we have (through the end of the test week;
    # the test values sit in the array but are only ever used as scoring truth,
    # never read as a feature thanks to the recursion blanking above).
    common_idx = full.index.intersection(time_index)
    actual[[time_index.get_loc(t) for t in common_idx]] = full.loc[common_idx].to_numpy()

    def pos(ts: pd.Timestamp) -> int:
        return time_index.get_loc(ts)

    train_positions = np.arange(MIN_POS, pos(c.TRAIN_END) + 1)
    trainval_positions = np.arange(MIN_POS, pos(c.VAL_END) + 1)

    # ---- hyperparameter selection on the validation window --------------
    train_feats, train_target = _build_training_matrix(actual, time_index, train_positions)
    val_index = splits.val.index
    best_params: dict | None = None
    best_val_mape = np.inf
    sweep: list[dict] = []
    for params in PARAM_GRID:
        model = _fit_point(params, train_feats, train_target)
        val_point, _ = _recursive_forecast(model, actual, time_index, pos(c.TRAIN_END), val_index)
        val_mape = c.mape(splits.val.to_numpy(), val_point.to_numpy())
        sweep.append({**params, "val_mape": round(val_mape, 4)})
        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_params = params

    assert best_params is not None

    # ---- refit on Train+Val, then forecast the test week ----------------
    tv_feats, tv_target = _build_training_matrix(actual, time_index, trainval_positions)
    point_model = _fit_point(best_params, tv_feats, tv_target)
    point, feat_rows = _recursive_forecast(
        point_model, actual, time_index, pos(c.VAL_END), splits.test.index
    )

    # ---- quantile fits for prediction intervals -------------------------
    quantiles: dict[float, pd.Series] = {}
    for q in QUANTILE_LEVELS:
        qmodel = lgb.LGBMRegressor(**{**_FIXED_PARAMS, **best_params, "objective": "quantile", "alpha": q})
        qmodel.fit(tv_feats, tv_target)
        qpred = qmodel.predict(feat_rows)
        quantiles[q] = pd.Series(qpred, index=splits.test.index, name=f"lightgbm_q{q}")
    # Guard against quantile crossing by sorting per hour.
    stacked = np.sort(np.column_stack([quantiles[q].to_numpy() for q in QUANTILE_LEVELS]), axis=1)
    for j, q in enumerate(QUANTILE_LEVELS):
        quantiles[q] = pd.Series(stacked[:, j], index=splits.test.index, name=f"lightgbm_q{q}")

    importances = dict(zip(FEATURE_NAMES, point_model.booster_.feature_importance(importance_type="gain")))

    return c.ModelResult(
        name="lightgbm",
        point=point,
        runtime_s=time.time() - start,
        hyperparameters={**best_params, "selected_by_val_mape": round(best_val_mape, 4), "sweep": sweep},
        quantiles=quantiles,
        extra={"feature_importance": {k: float(v) for k, v in importances.items()}},
    )
