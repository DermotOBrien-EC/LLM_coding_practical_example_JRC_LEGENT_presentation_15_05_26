from __future__ import annotations
import numpy as np
import pandas as pd
import holidays
from sklearn.ensemble import HistGradientBoostingRegressor

TZ = "Europe/Berlin"
YEARS = range(2014, 2021)
DE_HOL = holidays.Germany(years=YEARS)  # nationwide subdivision-less -> only national holidays
# Epiphany (Jan 6) is a holiday only in BW, BY, ST. Track it separately.
DE_BY = holidays.Germany(years=YEARS, subdiv="BY")

# Precompute holiday date sets once (vectorised lookups downstream).
NAT_DATES = {pd.Timestamp(d) for d in DE_HOL}
REG_DATES = {pd.Timestamp(d) for d in DE_BY} - NAT_DATES  # e.g. Epiphany Jan 6
PRE_DATES = {d - pd.Timedelta(days=1) for d in NAT_DATES}   # day before a national holiday
POST_DATES = {d + pd.Timedelta(days=1) for d in NAT_DATES}  # day after a national holiday


def load_data() -> pd.DataFrame:
    df = pd.read_csv("opsd_de_load.csv", parse_dates=["utc_timestamp"])
    df = df.rename(columns={"DE_load_actual_entsoe_transparency": "load"})
    df = df.set_index("utc_timestamp").sort_index()
    return df


def make_features(idx_utc: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar features derived from LOCAL (Europe/Berlin) time."""
    loc = idx_utc.tz_convert(TZ)
    dates = loc.normalize().tz_localize(None)
    f = pd.DataFrame(index=idx_utc)
    f["hour"] = loc.hour
    f["dow"] = loc.dayofweek
    f["is_weekend"] = (loc.dayofweek >= 5).astype(int)
    doy = loc.dayofyear
    f["yr_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["yr_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["year"] = loc.year
    dnorm = pd.DatetimeIndex(dates)  # normalized tz-naive daily timestamps
    f["is_holiday"] = dnorm.isin(NAT_DATES).astype(int)
    f["is_regional_holiday"] = dnorm.isin(REG_DATES).astype(int)  # e.g. Epiphany Jan 6
    f["is_pre_holiday"] = dnorm.isin(PRE_DATES).astype(int)   # day before a national holiday
    f["is_post_holiday"] = dnorm.isin(POST_DATES).astype(int)  # day after a national holiday
    return f


CAT = ["hour", "dow", "is_weekend", "is_holiday", "is_regional_holiday",
       "is_pre_holiday", "is_post_holiday"]


def fit_predict(train: pd.DataFrame, fc_index: pd.DatetimeIndex) -> np.ndarray:
    Xtr = make_features(train.index)
    ytr = train["load"].values
    model = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_depth=None,
        max_leaf_nodes=63, min_samples_leaf=40,
        l2_regularization=1.0, random_state=0,
    )
    model.fit(Xtr, ytr)
    Xfc = make_features(fc_index)
    return model.predict(Xfc)


# ---- Baselines ----
def seasonal_naive_week(df: pd.DataFrame, fc_index: pd.DatetimeIndex) -> np.ndarray:
    """Load from same hour one week earlier."""
    return df["load"].reindex(fc_index - pd.Timedelta(days=7)).values


def seasonal_naive_year(df: pd.DataFrame, fc_index: pd.DatetimeIndex) -> np.ndarray:
    """Same calendar week last year, aligned by day-of-week (shift ~364 days)."""
    return df["load"].reindex(fc_index - pd.Timedelta(days=364)).values


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE%": float(np.mean(np.abs(err / y_true)) * 100),
        "meanbias": float(np.mean(err)),
    }


def week_index(year: int) -> pd.DatetimeIndex:
    # "First week of January" defined in German local time (CET), then to UTC.
    loc = pd.date_range(f"{year}-01-01 00:00", f"{year}-01-07 23:00", freq="h", tz=TZ)
    return loc.tz_convert("UTC")


def main() -> None:
    df = load_data()

    # ---- Backtest on first weeks of Jan 2017, 2018, 2019 ----
    print("=== BACKTEST: forecast Jan 1-7, train only on prior data ===\n")
    rows = []
    for yr in (2017, 2018, 2019):
        fc = week_index(yr)
        train = df.loc[: fc[0] - pd.Timedelta(hours=1)]
        yt = df["load"].reindex(fc).values
        preds = {
            "GBM": fit_predict(train, fc),
            "naive_week": seasonal_naive_week(df, fc),
            "naive_year": seasonal_naive_year(df, fc),
        }
        for name, yp in preds.items():
            m = metrics(yt, yp)
            m.update(year=yr, model=name)
            rows.append(m)
    bt = pd.DataFrame(rows)[["year", "model", "MAE", "RMSE", "MAPE%", "meanbias"]]
    print(bt.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    print("\nMean across backtest years:")
    agg = bt.groupby("model")[["MAE", "RMSE", "MAPE%"]].mean()
    print(agg.to_string(float_format=lambda x: f"{x:,.2f}"))

    # ---- Final forecast: Jan 1-7, 2020 ----
    print("\n=== FINAL FORECAST: Jan 1-7, 2020 (train = all data < 2020-01-01) ===\n")
    fc = week_index(2020)
    train = df.loc[: fc[0] - pd.Timedelta(hours=1)]
    yhat = fit_predict(train, fc)
    y_actual = df["load"].reindex(fc).values  # held out, present in file

    out = pd.DataFrame({
        "utc_timestamp": fc,
        "local_time": fc.tz_convert(TZ).strftime("%Y-%m-%d %H:%M %a"),
        "forecast_MW": np.round(yhat, 1),
        "actual_MW": y_actual,
    })
    out["abs_err"] = (out["forecast_MW"] - out["actual_MW"]).abs().round(1)
    out.to_csv("forecast_jan2020.csv", index=False)

    m = metrics(y_actual, yhat)
    print("Held-out accuracy vs actuals (168 h):")
    for k, v in m.items():
        print(f"  {k:9s} {v:,.2f}")
    # baselines on 2020 for reference
    for name, yp in {"naive_week": seasonal_naive_week(df, fc),
                     "naive_year": seasonal_naive_year(df, fc)}.items():
        mm = metrics(y_actual, yp)
        print(f"  [{name}] MAPE% {mm['MAPE%']:.2f}  MAE {mm['MAE']:,.0f}")

    # daily summary
    out["day"] = fc.tz_convert(TZ).strftime("%a %m-%d")
    daily = out.groupby("day", sort=False).apply(
        lambda g: pd.Series({
            "fc_mean": g["forecast_MW"].mean(),
            "act_mean": g["actual_MW"].mean(),
            "MAPE%": (g["abs_err"] / g["actual_MW"]).mean() * 100,
        }), include_groups=False)
    print("\nDaily means (MW) and MAPE:")
    print(daily.to_string(float_format=lambda x: f"{x:,.1f}"))
    print("\nWrote forecast_jan2020.csv (168 hourly rows).")

    # save arrays for plotting
    np.savez("_fc.npz", utc=fc.astype("int64").values, yhat=yhat, yact=y_actual)


if __name__ == "__main__":
    main()
