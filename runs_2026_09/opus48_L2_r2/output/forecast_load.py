"""Forecast German hourly electricity load for the first week of January 2020.

Trains a gradient-boosted regression on calendar and seasonal-lag features
using all history before 2020-01-01, forecasts the 168 hours from
2020-01-01 00:00 to 2020-01-07 23:00 (UTC), plots the forecast against the
actual load, and reports accuracy against both the actuals and a
seasonal-naive baseline (same hour one week earlier).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_FILE = Path(__file__).with_name("opsd_de_load.csv")
PLOT_FILE = Path(__file__).with_name("forecast_jan2020.png")

FORECAST_START = pd.Timestamp("2020-01-01 00:00", tz="UTC")
FORECAST_END = pd.Timestamp("2020-01-07 23:00", tz="UTC")  # inclusive, 168 hours

WEEK_HOURS = 168
YEAR_HOURS = 364 * 24  # 364 days keeps the day-of-week aligned across the lag


def german_public_holidays(years: range) -> set[pd.Timestamp]:
    """Nationwide German public holidays (dates only, UTC midnight)."""
    days: set[pd.Timestamp] = set()
    for year in years:
        # Fixed-date nationwide holidays.
        fixed = [(1, 1), (5, 1), (10, 3), (12, 25), (12, 26)]
        for month, day in fixed:
            days.add(pd.Timestamp(year=year, month=month, day=day, tz="UTC"))
        # Easter (Anonymous Gregorian algorithm) and the movable feasts.
        a = year % 19
        b, c = divmod(year, 100)
        d, e = divmod(b, 4)
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i, k = divmod(c, 4)
        m = (32 + 2 * e + 2 * i - h - k) % 7
        n = (a + 11 * h + 22 * m) // 451
        month, day = divmod(h + m - 7 * n + 114, 31)
        easter = pd.Timestamp(year=year, month=month, day=day + 1, tz="UTC")
        for offset in (-2, 1, 39, 50):  # Good Fri, Easter Mon, Ascension, Whit Mon
            days.add(easter + pd.Timedelta(days=offset))
    return days


def build_features(load: pd.Series) -> pd.DataFrame:
    """Calendar and seasonal-lag features, all knowable at forecast time."""
    idx = load.index
    holidays = german_public_holidays(range(idx.year.min(), idx.year.max() + 1))
    is_holiday = pd.Series(idx.normalize().isin(holidays), index=idx).astype(int)

    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    feats = pd.DataFrame(
        {
            "hour": hour,
            "dayofweek": idx.dayofweek.to_numpy(),
            "month": idx.month.to_numpy(),
            "is_weekend": (idx.dayofweek >= 5).astype(int),
            "is_holiday": is_holiday.to_numpy(),
            # Cyclical encodings so the model sees hour 23 next to hour 0.
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            # Seasonal lags available for the whole horizon from pre-2020 data.
            "lag_1w": load.shift(WEEK_HOURS).to_numpy(),
            "lag_2w": load.shift(2 * WEEK_HOURS).to_numpy(),
            "lag_1y": load.shift(YEAR_HOURS).to_numpy(),
        },
        index=idx,
    )
    return feats


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - actual
    return {
        "MAPE_%": float(np.mean(np.abs(err / actual)) * 100),
        "MAE_MW": float(np.mean(np.abs(err))),
        "RMSE_MW": float(np.sqrt(np.mean(err**2))),
        "max_abs_err_MW": float(np.max(np.abs(err))),
        "bias_MW": float(np.mean(err)),
    }


def main() -> None:
    df = pd.read_csv(DATA_FILE, parse_dates=["utc_timestamp"])
    load = (
        df.set_index("utc_timestamp")["DE_load_actual_entsoe_transparency"]
        .sort_index()
        .asfreq("h")  # enforce a gap-free hourly index so lags line up
    )
    if load.isna().any():
        load = load.interpolate(limit=6)  # bridge any short gaps

    feats = build_features(load)
    horizon = (feats.index >= FORECAST_START) & (feats.index <= FORECAST_END)
    train = feats.index < FORECAST_START

    x_train = feats.loc[train].dropna()
    y_train = load.loc[x_train.index]
    x_test = feats.loc[horizon]
    y_test = load.loc[horizon]

    assert len(x_test) == WEEK_HOURS, f"expected 168 forecast hours, got {len(x_test)}"

    model = HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.05,
        max_leaf_nodes=63,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=0,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    # Seasonal-naive baseline: the same hour one week earlier.
    baseline = load.shift(WEEK_HOURS).loc[horizon].to_numpy()

    actual = y_test.to_numpy()
    model_m = metrics(actual, pred)
    base_m = metrics(actual, baseline)

    print(f"Training rows: {len(x_train):,}  "
          f"({x_train.index.min().date()} to {x_train.index.max().date()})")
    print(f"Forecast horizon: {FORECAST_START:%Y-%m-%d %H:%M} to "
          f"{FORECAST_END:%Y-%m-%d %H:%M} UTC ({len(x_test)} hours)\n")
    header = f"{'metric':<16}{'model':>14}{'naive (1-week)':>18}"
    print(header)
    print("-" * len(header))
    for key in model_m:
        print(f"{key:<16}{model_m[key]:>14,.1f}{base_m[key]:>18,.1f}")
    skill = (1 - model_m["RMSE_MW"] / base_m["RMSE_MW"]) * 100
    print(f"\nModel RMSE skill vs naive baseline: {skill:+.1f}%")
    print(f"Mean actual load over the week: {actual.mean():,.0f} MW")

    # Plot.
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(x_test.index, actual, color="#1b2a4a", lw=2.2, label="Actual")
    ax.plot(x_test.index, pred, color="#e8703a", lw=2.0, label="Forecast (model)")
    ax.plot(x_test.index, baseline, color="#9aa5b1", lw=1.3, ls="--",
            label="Naive (1-week) baseline")
    ax.set_title("German electricity load: forecast vs actual, 1-7 Jan 2020")
    ax.set_ylabel("Load (MW)")
    ax.set_xlabel("Time (UTC)")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.text(0.012, 0.04,
            f"Model MAPE {model_m['MAPE_%']:.2f}%   MAE {model_m['MAE_MW']:,.0f} MW"
            f"   RMSE {model_m['RMSE_MW']:,.0f} MW",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=130)
    print(f"\nPlot written to {PLOT_FILE.name}")


if __name__ == "__main__":
    main()
