"""Forecast German hourly electricity load for 2020-01-01..2020-01-07.

Trains a LightGBM regressor on calendar + lagged features using every hour
before 2020-01-01, produces a 168-hour forecast for the target week, plots
forecast against actuals, and reports MAE / RMSE / MAPE.
"""
from __future__ import annotations

from pathlib import Path

import holidays
import lightgbm as lgb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "opsd_de_load.csv"
PLOT_PATH = HERE / "forecast_vs_actual.png"

FORECAST_START = pd.Timestamp("2020-01-01 00:00", tz="UTC")
FORECAST_END = pd.Timestamp("2020-01-07 23:00", tz="UTC")
HORIZON_HOURS = 168


def load_series() -> pd.Series:
    df = pd.read_csv(CSV_PATH, parse_dates=["utc_timestamp"])
    df = df.rename(columns={"DE_load_actual_entsoe_transparency": "load"})
    df = df.set_index("utc_timestamp").sort_index()
    df["load"] = df["load"].interpolate("time", limit=6)
    return df["load"]


def build_features(index: pd.DatetimeIndex, load: pd.Series) -> pd.DataFrame:
    de_holidays = holidays.country_holidays("DE", years=range(2015, 2021))
    local = index.tz_convert("Europe/Berlin")
    feats = pd.DataFrame(index=index)
    feats["hour"] = local.hour
    feats["dow"] = local.dayofweek
    feats["month"] = local.month
    feats["day_of_year"] = local.dayofyear
    feats["is_weekend"] = (local.dayofweek >= 5).astype(int)
    feats["is_holiday"] = np.array(
        [1 if d.date() in de_holidays else 0 for d in local]
    )
    # Cyclical encodings help the tree separate close-to-midnight hours.
    feats["hour_sin"] = np.sin(2 * np.pi * feats["hour"] / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * feats["hour"] / 24)
    feats["doy_sin"] = np.sin(2 * np.pi * feats["day_of_year"] / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * feats["day_of_year"] / 365.25)
    # Lag features that are safe for a 168h-ahead forecast: use only lags
    # that reach back at least one week, so every target hour has a known
    # value at forecast time.
    for lag in (168, 336, 504):
        feats[f"lag_{lag}h"] = load.reindex(index - pd.Timedelta(hours=lag)).values
    return feats


def main() -> None:
    load = load_series()

    target_index = pd.date_range(FORECAST_START, FORECAST_END, freq="h", tz="UTC")
    assert len(target_index) == HORIZON_HOURS

    train_index = load.index[load.index < FORECAST_START]
    X_train = build_features(train_index, load)
    y_train = load.loc[train_index]

    # Drop rows where any lag is missing (the first three weeks of the record).
    mask = X_train.notna().all(axis=1) & y_train.notna()
    X_train, y_train = X_train.loc[mask], y_train.loc[mask]

    X_test = build_features(target_index, load)
    y_test = load.reindex(target_index)

    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=64,
        min_data_in_leaf=40,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        random_state=0,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    forecast = pd.Series(model.predict(X_test), index=target_index, name="forecast")

    err = forecast - y_test
    mae = err.abs().mean()
    rmse = float(np.sqrt((err ** 2).mean()))
    mape = (err.abs() / y_test).mean() * 100
    peak_actual = y_test.max()
    peak_forecast = forecast.max()

    print(f"Forecast window : {FORECAST_START} .. {FORECAST_END} ({HORIZON_HOURS} h)")
    print(f"Training hours  : {len(y_train):,}")
    print(f"MAE             : {mae:8.1f} MW")
    print(f"RMSE            : {rmse:8.1f} MW")
    print(f"MAPE            : {mape:8.2f} %")
    print(f"Peak actual     : {peak_actual:8.0f} MW")
    print(f"Peak forecast   : {peak_forecast:8.0f} MW")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.index, y_test.values, label="Actual", color="#1f2937", linewidth=1.8)
    ax.plot(
        forecast.index,
        forecast.values,
        label="Forecast",
        color="#d97706",
        linewidth=1.6,
        linestyle="--",
    )
    ax.set_title(
        "German hourly load, 2020-01-01 to 2020-01-07\n"
        f"MAE {mae:,.0f} MW  |  RMSE {rmse:,.0f} MW  |  MAPE {mape:.2f}%"
    )
    ax.set_ylabel("Load (MW)")
    ax.set_xlabel("Time (UTC)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    print(f"Plot written to : {PLOT_PATH}")


if __name__ == "__main__":
    main()
