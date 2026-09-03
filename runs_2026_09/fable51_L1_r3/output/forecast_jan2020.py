"""Forecast German hourly load for 1-7 January 2020 (Europe/Berlin) from OPSD data.

Training data ends 2019-12-31 23:00 local; the forecast horizon is 168 hours.
Every feature is computable at forecast time (calendar + lags >= 168 h).
Model selection uses the same week one year earlier as a holdout.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

CSV = "opsd_de_load.csv"
TZ = "Europe/Berlin"
HORIZON = 168


@dataclass(frozen=True)
class Scores:
    mae: float
    rmse: float
    mape: float

    def row(self, name: str) -> str:
        return f"{name:<28} MAE {self.mae:8.0f} MW  RMSE {self.rmse:8.0f} MW  MAPE {self.mape:5.2f} %"


def score(y: pd.Series, yhat: pd.Series) -> Scores:
    err = y - yhat
    return Scores(
        mae=float(err.abs().mean()),
        rmse=float(np.sqrt((err**2).mean())),
        mape=float((err.abs() / y).mean() * 100),
    )


def load_series() -> pd.Series:
    df = pd.read_csv(CSV, parse_dates=["utc_timestamp"])
    s = df.set_index("utc_timestamp")["DE_load_actual_entsoe_transparency"]
    s.index = s.index.tz_convert(TZ)
    s = s.sort_index()
    full = pd.date_range(s.index[0], s.index[-1], freq="h", tz=TZ)
    missing = full.difference(s.index)
    dups = int(s.index.duplicated().sum())
    print(f"rows={len(s)} span={s.index[0]} .. {s.index[-1]} missing_hours={len(missing)} dups={dups}")
    print(f"nulls={int(s.isna().sum())}  min={s.min():.0f} max={s.max():.0f} MW")
    s = s[~s.index.duplicated()].reindex(full).interpolate(limit=3)
    return s.rename("load")


def build_features(s: pd.Series) -> pd.DataFrame:
    idx = s.index
    de_hol = holidays.country_holidays("DE", years=range(2014, 2022))
    # Regional holidays observed only in some states (Epiphany, Corpus Christi, Reformation Day
    # before 2018 was regional, All Saints, Assumption). Load dips partially on those days.
    regional = holidays.country_holidays("DE", subdiv="BY", years=range(2014, 2022))
    dates = idx.normalize().date
    f = pd.DataFrame(index=idx)
    f["hour"] = idx.hour
    f["dow"] = idx.dayofweek
    f["doy"] = idx.dayofyear
    f["month"] = idx.month
    f["year"] = idx.year
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    nat = np.array([d in de_hol for d in dates])
    reg = np.array([(d in regional) and (d not in de_hol) for d in dates])
    f["is_holiday"] = nat.astype(int)
    f["is_regional_holiday"] = reg.astype(int)
    day_flag = pd.Series(nat.astype(int), index=idx)
    f["holiday_prev_day"] = day_flag.shift(24).fillna(0).astype(int).to_numpy()
    f["holiday_next_day"] = day_flag.shift(-24).fillna(0).astype(int).to_numpy()
    # Christmas/New Year trough: 24 Dec .. 6 Jan, industrial shutdown period.
    doy = idx.dayofyear
    f["xmas_period"] = ((doy >= 358) | (doy <= 6)).astype(int)
    f["days_into_xmas"] = np.where(doy >= 358, doy - 357, np.where(doy <= 6, doy + 8, 0))
    # Lags usable across the whole 168 h horizon.
    for lag in (168, 336, 504):
        f[f"lag_{lag}"] = s.shift(lag).to_numpy()
    # Same weekday one year earlier (364 days) and its neighbourhood mean.
    f["lag_364d"] = s.shift(364 * 24).to_numpy()
    f["lag_371d"] = s.shift(371 * 24).to_numpy()
    f["lag_357d"] = s.shift(357 * 24).to_numpy()
    # Recent level: mean of the most recent 7 days available before the horizon starts (t-168..t-336).
    f["mean_prev_week"] = s.shift(168).rolling(168).mean().to_numpy()
    # Year-over-year level drift: last 28 available days vs the same 28 days a year earlier.
    recent28 = s.shift(168).rolling(28 * 24).mean()
    yoy = recent28 / recent28.shift(364 * 24)
    f["yoy_level_ratio"] = yoy.to_numpy()
    f["lag_364d_scaled"] = (f["lag_364d"] * yoy).to_numpy()
    # Bridge day: a working day squeezed between a holiday and a weekend (or another holiday).
    wd = idx.dayofweek < 5
    hol_any = nat
    prev_off = pd.Series(hol_any | (idx.dayofweek >= 5), index=idx).shift(24).fillna(False).to_numpy()
    next_off = pd.Series(hol_any | (idx.dayofweek >= 5), index=idx).shift(-24).fillna(False).to_numpy()
    f["is_bridge_day"] = (wd & ~hol_any & prev_off & next_off).astype(int)
    return f


def fit_predict_lgb(f: pd.DataFrame, y: pd.Series, train_end: pd.Timestamp,
                    horizon_idx: pd.DatetimeIndex, seed: int = 0) -> pd.Series:
    train = f.loc[:train_end].dropna()
    ytr = y.loc[train.index]
    params = {
        "objective": "regression_l1",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": seed,
    }
    ds = lgb.Dataset(train, ytr, categorical_feature=["hour", "dow", "month"])
    model = lgb.train(params, ds, num_boost_round=1500)
    pred = model.predict(f.loc[horizon_idx])
    imp = pd.Series(model.feature_importance("gain"), index=train.columns).sort_values(ascending=False)
    print("top features by gain:", ", ".join(f"{k}={v:.0f}" for k, v in (imp / imp.sum() * 100).head(8).items()))
    return pd.Series(pred, index=horizon_idx, name="lgb")


def year_ago_scaled(s: pd.Series, train_end: pd.Timestamp, horizon_idx: pd.DatetimeIndex) -> pd.Series:
    """Same weekday-aligned hours 52 weeks earlier, rescaled by the ratio of the last 4 weeks of
    training data to the corresponding weeks a year earlier (captures level drift)."""
    ago = s.shift(364 * 24).loc[horizon_idx]
    recent = s.loc[train_end - pd.Timedelta(days=28) + pd.Timedelta(hours=1):train_end]
    recent_ago = s.shift(364 * 24).loc[recent.index]
    ratio = recent.mean() / recent_ago.mean()
    return (ago * ratio).rename("year_ago_scaled")


def run_window(s: pd.Series, f: pd.DataFrame, year: int, label: str) -> dict[str, pd.Series]:
    train_end = pd.Timestamp(f"{year-1}-12-31 23:00", tz=TZ)
    horizon_idx = pd.date_range(f"{year}-01-01 00:00", periods=HORIZON, freq="h", tz=TZ)
    y = s.loc[horizon_idx]
    preds: dict[str, pd.Series] = {}
    preds["seasonal_naive_168h"] = s.shift(168).loc[horizon_idx].rename("seasonal_naive_168h")
    preds["year_ago_scaled"] = year_ago_scaled(s, train_end, horizon_idx)
    lgb_runs = [fit_predict_lgb(f, s, train_end, horizon_idx, seed=k) for k in range(3)]
    preds["lightgbm"] = pd.concat(lgb_runs, axis=1).mean(axis=1).rename("lightgbm")
    preds["blend_lgb_yearago"] = (0.7 * preds["lightgbm"] + 0.3 * preds["year_ago_scaled"]).rename("blend")
    print(f"\n== {label}: 1-7 Jan {year}, train through {train_end} ==")
    for name, p in preds.items():
        print(score(y, p).row(name))
    preds["actual"] = y.rename("actual")
    return preds


def main() -> int:
    s = load_series()
    f = build_features(s)
    run_window(s, f, 2018, "HOLDOUT A (model selection)")
    run_window(s, f, 2019, "HOLDOUT B (model selection)")
    out = run_window(s, f, 2020, "TARGET")
    df = pd.DataFrame(out)
    df.index.name = "time_europe_berlin"
    df.round(0).to_csv("forecast_jan2020_week1.csv")
    daily = df.resample("D").agg(["mean", "min", "max"])[["actual", "lightgbm"]].round(0)
    print("\nDaily summary (MW), actual vs LightGBM:")
    print(daily.to_string())
    print("\nwrote forecast_jan2020_week1.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
