"""Forecast German hourly electricity load for the first week of January 2020.

Trains on OPSD load data strictly before 2020-01-01 UTC and forecasts the
168 hours 2020-01-01 00:00 .. 2020-01-07 23:00 UTC. The CSV also holds the
actuals for that week, so the forecast is scored against held-out truth.
"""

from __future__ import annotations

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

DATA = "opsd_de_load.csv"
H_START = pd.Timestamp("2020-01-01 00:00", tz="UTC")
H_END = pd.Timestamp("2020-01-07 23:00", tz="UTC")
HORIZON = 168


def load_data() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["utc_timestamp"])
    df = df.rename(columns={"DE_load_actual_entsoe_transparency": "load"})
    s = df.set_index("utc_timestamp")["load"].asfreq("h")
    assert s.notna().all(), "unexpected gaps in load series"
    return s


def calendar_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    # German national holidays (federal). Jan 1 is federal; Jan 6 is regional
    # only, so it is captured by the separate epiphany flag below.
    de_hol = holidays.Germany(years=range(2015, 2021))
    local = idx.tz_convert("Europe/Berlin")
    dates = local.normalize().tz_localize(None)
    is_hol = np.array([d.date() in de_hol for d in dates])
    prev = np.array([(d - pd.Timedelta(days=1)).date() in de_hol for d in dates])
    nxt = np.array([(d + pd.Timedelta(days=1)).date() in de_hol for d in dates])
    doy = local.dayofyear
    f = pd.DataFrame(index=idx)
    f["hour"] = local.hour
    f["dow"] = local.dayofweek
    f["month"] = local.month
    f["is_weekend"] = (local.dayofweek >= 5).astype(int)
    f["is_holiday"] = is_hol.astype(int)
    f["holiday_bridge"] = (prev | nxt).astype(int)
    f["epiphany"] = ((local.month == 1) & (local.day == 6)).astype(int)
    # Deep-winter-holiday window (Dec 24 .. Jan 6): load is strongly depressed.
    f["xmas_window"] = (((local.month == 12) & (local.day >= 24)) | ((local.month == 1) & (local.day <= 6))).astype(int)
    # Cyclical encodings so the tree splits do not fight ordinal wrap-around.
    f["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    return f


def build_matrix(s: pd.Series, idx: pd.DatetimeIndex) -> pd.DataFrame:
    f = calendar_features(idx)
    # Level anchors known for the whole horizon (they point into the past):
    #   364 days = exactly 52 weeks, so same hour & weekday one year prior.
    #   7 days  = same hour & weekday one week prior.
    f["lag_364d"] = s.reindex(idx - pd.Timedelta(days=364)).to_numpy()
    f["lag_7d"] = s.reindex(idx - pd.Timedelta(days=7)).to_numpy()
    f["lag_14d"] = s.reindex(idx - pd.Timedelta(days=14)).to_numpy()
    return f


def main() -> None:
    s = load_data()
    train_idx = s.index[s.index < H_START]
    fut_idx = pd.date_range(H_START, H_END, freq="h", tz="UTC")

    X_train = build_matrix(s, train_idx)
    y_train = s.loc[train_idx]
    X_fut = build_matrix(s, fut_idx)

    # Drop rows whose 364-day lag is unavailable (first year of history).
    ok = X_train["lag_364d"].notna() & X_train["lag_7d"].notna() & X_train["lag_14d"].notna()
    X_train, y_train = X_train[ok], y_train[ok]

    model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=64,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=0,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    pred_lgb = pd.Series(model.predict(X_fut), index=fut_idx, name="lgbm")

    actual = s.loc[fut_idx]
    snaive = s.reindex(fut_idx - pd.Timedelta(days=7)).to_numpy()  # last-week baseline
    snaive = pd.Series(snaive, index=fut_idx, name="snaive")

    def scores(pred: pd.Series) -> dict[str, float]:
        err = pred.to_numpy() - actual.to_numpy()
        return {
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err**2))),
            "MAPE_%": float(np.mean(np.abs(err) / actual.to_numpy()) * 100),
        }

    print("=== Backtest scores, Jan 1-7 2020 (168 h) ===")
    for name, p in [("seasonal-naive (last week)", snaive), ("LightGBM", pred_lgb)]:
        sc = scores(p)
        print(f"{name:28s}  MAE={sc['MAE']:6.0f}  RMSE={sc['RMSE']:6.0f}  MAPE={sc['MAPE_%']:.2f}%")

    out = pd.DataFrame(
        {
            "utc_timestamp": fut_idx,
            "forecast_lgbm_MW": pred_lgb.to_numpy().round(1),
            "actual_MW": actual.to_numpy(),
        }
    )
    out["abs_pct_err"] = (out.forecast_lgbm_MW - out.actual_MW).abs() / out.actual_MW * 100
    out.to_csv("forecast_jan2020.csv", index=False)
    print("\nWrote forecast_jan2020.csv")

    # Daily MAPE breakdown.
    daily = out.copy()
    daily["date"] = daily.utc_timestamp.dt.tz_convert("Europe/Berlin").dt.date
    print("\nDaily MAPE (LightGBM):")
    for d, g in daily.groupby("date"):
        print(f"  {d}  MAPE={g.abs_pct_err.mean():5.2f}%  mean_load={g.actual_MW.mean():6.0f} MW")

    # Plot.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 5))
    local = fut_idx.tz_convert("Europe/Berlin")
    ax.plot(local, actual.to_numpy(), label="Actual", color="#111", lw=1.8)
    ax.plot(local, pred_lgb.to_numpy(), label="LightGBM forecast", color="#c44", lw=1.6)
    ax.plot(local, snaive.to_numpy(), label="Seasonal-naive", color="#888", lw=1.0, ls="--")
    ax.set_title("German hourly electricity load — forecast vs actual, 1-7 Jan 2020")
    ax.set_ylabel("Load (MW)")
    ax.set_xlabel("Local time (Europe/Berlin)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("forecast_jan2020.png", dpi=110)
    print("Wrote forecast_jan2020.png")


if __name__ == "__main__":
    main()
