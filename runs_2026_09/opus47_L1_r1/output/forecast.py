from __future__ import annotations

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

CSV = "/Users/doob/dev/energy_forecast_ws/37e6d0/project/runs/37e6d0/opsd_de_load.csv"


def load() -> pd.Series:
    df = pd.read_csv(CSV, parse_dates=["utc_timestamp"])
    df = df.set_index("utc_timestamp").sort_index()
    s = df["DE_load_actual_entsoe_transparency"].astype(float)
    return s.interpolate(limit=3)


def build_features(idx: pd.DatetimeIndex, de_hol: holidays.HolidayBase) -> pd.DataFrame:
    local = idx.tz_convert("Europe/Berlin")
    doy = local.dayofyear.to_numpy()
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    month = local.month.to_numpy()
    is_hol = np.array([d.date() in de_hol for d in local]).astype(int)
    # Day before / after holiday, and turn-of-year bridge days (typically low-load)
    is_hol_next = np.roll(is_hol, -24)
    is_hol_prev = np.roll(is_hol, 24)
    is_hol_next[-24:] = 0
    is_hol_prev[:24] = 0
    day = local.day.to_numpy()
    is_xmas_ny = (((month == 12) & (day >= 24)) | ((month == 1) & (day <= 6))).astype(int)

    return pd.DataFrame(
        {
            "hour": hour,
            "dow": dow,
            "month": month,
            "doy": doy,
            "is_weekend": (dow >= 5).astype(int),
            "is_hol": is_hol,
            "is_hol_prev": is_hol_prev,
            "is_hol_next": is_hol_next,
            "is_xmas_ny": is_xmas_ny,
            "sin_hour": np.sin(2 * np.pi * hour / 24),
            "cos_hour": np.cos(2 * np.pi * hour / 24),
            "sin_year": np.sin(2 * np.pi * doy / 365.25),
            "cos_year": np.cos(2 * np.pi * doy / 365.25),
            "sin_year2": np.sin(4 * np.pi * doy / 365.25),
            "cos_year2": np.cos(4 * np.pi * doy / 365.25),
        },
        index=idx,
    )


def main() -> None:
    y = load()
    de_hol = holidays.country_holidays("DE", years=range(2015, 2021))

    fc_start = pd.Timestamp("2020-01-01 00:00", tz="UTC")
    fc_end = pd.Timestamp("2020-01-07 23:00", tz="UTC")

    y_train = y[y.index < fc_start].dropna()
    X_train = build_features(y_train.index, de_hol)

    cat_cols = ["hour", "dow", "month"]
    model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        num_leaves=64,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        objective="regression_l1",
        random_state=0,
        verbose=-1,
    )
    model.fit(X_train, y_train.values, categorical_feature=cat_cols)

    fc_idx = pd.date_range(fc_start, fc_end, freq="h", tz="UTC")
    X_fc = build_features(fc_idx, de_hol)
    yhat = model.predict(X_fc)

    y_true = y.reindex(fc_idx)
    out = pd.DataFrame({"actual_MW": y_true.values, "forecast_MW": yhat}, index=fc_idx)
    out.index.name = "utc_timestamp"
    out.to_csv("/Users/doob/dev/energy_forecast_ws/37e6d0/project/runs/37e6d0/forecast_2020w1.csv")

    err = out["forecast_MW"] - out["actual_MW"]
    mae = err.abs().mean()
    rmse = float(np.sqrt((err**2).mean()))
    mape = (err.abs() / out["actual_MW"]).mean() * 100

    daily = out.copy()
    daily["date"] = daily.index.tz_convert("Europe/Berlin").date
    day_summary = daily.groupby("date").agg(
        actual_mean=("actual_MW", "mean"),
        forecast_mean=("forecast_MW", "mean"),
        actual_peak=("actual_MW", "max"),
        forecast_peak=("forecast_MW", "max"),
    )
    day_summary["mape_%"] = daily.groupby("date").apply(
        lambda d: (d["forecast_MW"] - d["actual_MW"]).abs().mean() / d["actual_MW"].mean() * 100,
        include_groups=False,
    )

    print(f"Train rows: {len(y_train):,} (2015-01-01 .. 2019-12-31 UTC)")
    print(f"Horizon   : {fc_start} .. {fc_end} ({len(out)} h)")
    print()
    print(f"MAE  : {mae:8.0f} MW")
    print(f"RMSE : {rmse:8.0f} MW")
    print(f"MAPE : {mape:8.2f} %")
    print(f"Peak forecast : {out['forecast_MW'].max():8.0f} MW at {out['forecast_MW'].idxmax()}")
    print(f"Peak actual   : {out['actual_MW'].max():8.0f} MW at {out['actual_MW'].idxmax()}")
    print()
    print("Daily summary (Europe/Berlin dates):")
    print(day_summary.round(0).to_string())


if __name__ == "__main__":
    main()
