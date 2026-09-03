"""Forecast models for the first-week-of-January task. All are 168h-ahead, no leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd

import loadfc

# Reference window used to estimate the current-year level bias, ending before the
# Christmas period so holiday weeks never contaminate the anchor.
REF_DAYS = 45
REF_END_OFFSET_DAYS = 14  # stop ~Dec 18 when forecasting from Dec 31


def _win(start: pd.Timestamp, hours: int = 168) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=hours, freq="h", tz=loadfc.TZ)


def seasonal_naive(train: pd.Series, target: pd.DatetimeIndex) -> pd.Series:
    """Load 168h earlier (the Christmas week, when forecasting the first week of January)."""
    vals = [train.get(ts - pd.Timedelta(days=7), np.nan) for ts in target]
    return pd.Series(vals, index=target, dtype=float)


def prior_years_calendar(train: pd.Series, target: pd.DatetimeIndex) -> pd.Series:
    """Mean of the same calendar date-hour in all prior years, rescaled to the current level."""
    naive = target.tz_localize(None)
    out = np.full(len(target), np.nan)
    for i, ts in enumerate(naive):
        vals = []
        for back in range(1, 6):
            try:
                prev = ts.replace(year=ts.year - back)
            except ValueError:  # Feb 29
                continue
            key = pd.Timestamp(prev).tz_localize(loadfc.TZ, nonexistent="shift_forward")
            v = train.get(key, np.nan)
            if np.isfinite(v):
                vals.append(v)
        if vals:
            out[i] = float(np.mean(vals))
    pred = pd.Series(out, index=target, dtype=float)
    return pred * _level_ratio_prior_years(train, target[0])


def _level_ratio_prior_years(train: pd.Series, origin: pd.Timestamp) -> float:
    """Current-year vs prior-years load level over a recent non-holiday reference window."""
    end = origin - pd.Timedelta(days=REF_END_OFFSET_DAYS)
    start = end - pd.Timedelta(days=REF_DAYS)
    cur = train.loc[start:end]
    if cur.empty:
        return 1.0
    prev_means = []
    for back in range(1, 6):
        p = train.loc[start - pd.DateOffset(years=back) : end - pd.DateOffset(years=back)]
        if len(p) > 24 * 20:
            prev_means.append(p.mean())
    if not prev_means:
        return 1.0
    return float(cur.mean() / np.mean(prev_means))


def _residual_offset(train: pd.Series, fitted: pd.Series, origin: pd.Timestamp) -> float:
    """Mean log residual over the recent non-holiday reference window (level correction)."""
    end = origin - pd.Timedelta(days=REF_END_OFFSET_DAYS)
    start = end - pd.Timedelta(days=REF_DAYS)
    a = train.loc[start:end]
    f = fitted.loc[start:end]
    if a.empty:
        return 0.0
    return float(np.mean(np.log(a.to_numpy()) - np.log(f.to_numpy())))


def ridge(
    train: pd.Series,
    target: pd.DatetimeIndex,
    alpha: float = 3.0,
    level_adjust: bool = True,
) -> pd.Series:
    """Ridge on log load over calendar/holiday/turn-of-year features."""
    Xtr = loadfc.design_matrix(loadfc.build_features(train.index))
    m = loadfc.RidgeLog(alpha=alpha).fit(Xtr, train)
    pred = pd.Series(
        m.predict(loadfc.design_matrix(loadfc.build_features(target))), index=target, dtype=float
    )
    if level_adjust:
        fitted = pd.Series(m.predict(Xtr), index=train.index, dtype=float)
        pred = pred * np.exp(_residual_offset(train, fitted, target[0]))
    return pred


def lgbm(
    train: pd.Series,
    target: pd.DatetimeIndex,
    level_adjust: bool = True,
    seed: int = 0,
) -> pd.Series:
    """LightGBM on log load over the same calendar features (no autoregressive lags)."""
    import lightgbm as lgb

    cat = ["hour", "dow", "daytype"]
    cols = ["hour", "dow", "doy", "year", "is_holiday", "reg_holiday", "daytype", "toy",
            "pre_holiday", "post_holiday"]

    ftr = loadfc.build_features(train.index)[cols].copy()
    fte = loadfc.build_features(target)[cols].copy()
    for c in cat:
        ftr[c] = ftr[c].astype("category")
        fte[c] = pd.Categorical(fte[c], categories=ftr[c].cat.categories)

    ds = lgb.Dataset(ftr, label=np.log(train.to_numpy(dtype=float)), categorical_feature=cat)
    params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": seed,
        "num_threads": 4,
    }
    booster = lgb.train(params, ds, num_boost_round=800)
    pred = pd.Series(np.exp(booster.predict(fte)), index=target, dtype=float)
    if level_adjust:
        fitted = pd.Series(np.exp(booster.predict(ftr)), index=train.index, dtype=float)
        pred = pred * np.exp(_residual_offset(train, fitted, target[0]))
    return pred
