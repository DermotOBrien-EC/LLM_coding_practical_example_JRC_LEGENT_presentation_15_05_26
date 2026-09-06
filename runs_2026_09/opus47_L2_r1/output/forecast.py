"""Forecast German hourly electricity load for 2020-01-01 to 2020-01-07 (168 h).

Trains a LightGBM regressor on 2015-01 to 2019-12 data using calendar features
and a same-hour-last-week lag, predicts the target week, then plots the forecast
against the actual load and prints error metrics.
"""

from __future__ import annotations

from pathlib import Path

import holidays
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "opsd_de_load.csv"
PLOT_PATH = HERE / "forecast_2020_w01.png"

FORECAST_START = pd.Timestamp("2020-01-01 00:00", tz="UTC")
FORECAST_END = pd.Timestamp("2020-01-07 23:00", tz="UTC")


def load_series() -> pd.Series:
    df = pd.read_csv(CSV_PATH, parse_dates=["utc_timestamp"])
    s = (
        df.rename(columns={"DE_load_actual_entsoe_transparency": "load"})
        .set_index("utc_timestamp")["load"]
        .sort_index()
    )
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC")
    return s.dropna()


def build_features(idx: pd.DatetimeIndex, lag_168: pd.Series) -> pd.DataFrame:
    de_holidays = holidays.country_holidays("DE", years=range(idx.year.min(), idx.year.max() + 1))
    local = idx.tz_convert("Europe/Berlin")
    return pd.DataFrame(
        {
            "hour": local.hour,
            "dayofweek": local.dayofweek,
            "month": local.month,
            "day": local.day,
            "dayofyear": local.dayofyear,
            "weekofyear": local.isocalendar().week.to_numpy(),
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "is_holiday": np.array([d.date() in de_holidays for d in local]).astype(int),
            "year": local.year,
            "lag_168h": lag_168.to_numpy(),
        },
        index=idx,
    )


def main() -> None:
    load = load_series()

    target = load.loc[FORECAST_START:FORECAST_END]
    if len(target) != 168:
        raise RuntimeError(f"expected 168 target hours, got {len(target)}")

    train_end = FORECAST_START - pd.Timedelta(hours=1)
    train_load = load.loc[:train_end]

    lag = load.shift(168)
    train_idx = train_load.index[train_load.index >= train_load.index[0] + pd.Timedelta(hours=168)]
    X_train = build_features(train_idx, lag.loc[train_idx])
    y_train = train_load.loc[train_idx].to_numpy()

    X_test = build_features(target.index, lag.loc[target.index])
    y_test = target.to_numpy()

    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=64,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        random_state=0,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    err = y_test - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / y_test)) * 100)
    mean_load = float(np.mean(y_test))

    print(f"Training samples: {len(X_train):,}  ({train_idx[0]} to {train_idx[-1]})")
    print(f"Forecast window : {target.index[0]} to {target.index[-1]}  ({len(target)} h)")
    print(f"Mean actual load: {mean_load:,.0f} MW")
    print(f"MAE             : {mae:,.0f} MW")
    print(f"RMSE            : {rmse:,.0f} MW")
    print(f"MAPE            : {mape:.2f} %")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(target.index, y_test, label="Actual", color="black", linewidth=1.6)
    ax.plot(target.index, y_pred, label="Forecast", color="tab:red", linewidth=1.4, linestyle="--")
    ax.set_title("Germany hourly load, 1 to 7 January 2020: forecast vs actual")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Load (MW)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"Plot saved to  : {PLOT_PATH}")


if __name__ == "__main__":
    main()
