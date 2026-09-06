"""Forecast German hourly electricity load for the first week of January 2020.

Trains on all historical data before 2020-01-01, forecasts the 168 hours from
2020-01-01 00:00 to 2020-01-07 23:59 (UTC), plots the forecast against the
actual values, and reports accuracy.

The forecast is a genuine week-ahead forecast: the model sees no load values
from inside the target week. It relies on calendar structure (hour-of-day,
day-of-week, position in the year) plus a German public-holiday flag, all of
which are known in advance. That holiday flag is what lets the model anticipate
the sharp New Year's Day (1 Jan) dip.
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

DATA_FILE = Path("opsd_de_load.csv")
PLOT_FILE = Path("forecast_vs_actual.png")
FORECAST_START = pd.Timestamp("2020-01-01", tz="UTC")
FORECAST_END = pd.Timestamp("2020-01-08", tz="UTC")  # exclusive; 168 hours


def easter(year: int) -> pd.Timestamp:
    """Gregorian Easter Sunday (Anonymous / Meeus algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return pd.Timestamp(year=year, month=month, day=day + 1)


def german_holidays(years: range) -> set[pd.Timestamp]:
    """Nationwide German public holidays (dates, tz-naive) for the given years."""
    days: set[pd.Timestamp] = set()
    for y in years:
        e = easter(y)
        fixed = [
            pd.Timestamp(y, 1, 1),    # Neujahr
            pd.Timestamp(y, 5, 1),    # Tag der Arbeit
            pd.Timestamp(y, 10, 3),   # Tag der Deutschen Einheit
            pd.Timestamp(y, 12, 25),  # 1. Weihnachtstag
            pd.Timestamp(y, 12, 26),  # 2. Weihnachtstag
        ]
        movable = [
            e - pd.Timedelta(days=2),   # Karfreitag
            e + pd.Timedelta(days=1),   # Ostermontag
            e + pd.Timedelta(days=39),  # Christi Himmelfahrt
            e + pd.Timedelta(days=50),  # Pfingstmontag
        ]
        days.update(fixed + movable)
    return {d.normalize() for d in days}


def make_features(index: pd.DatetimeIndex, holidays: set[pd.Timestamp]) -> pd.DataFrame:
    """Calendar features known ahead of time for any timestamp."""
    local = index.tz_convert("Europe/Berlin")  # load follows local clock/behaviour
    dates = local.normalize().tz_localize(None)
    is_holiday = dates.isin(holidays)
    doy = local.dayofyear
    feats = pd.DataFrame(
        {
            "hour": local.hour,
            "dayofweek": local.dayofweek,
            "month": local.month,
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "is_holiday": is_holiday.astype(int),
            # smooth yearly cycle (captures the winter-peak / summer-trough shape)
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=index,
    )
    return feats


def main() -> None:
    df = pd.read_csv(DATA_FILE, parse_dates=["utc_timestamp"])
    df = df.rename(columns={"DE_load_actual_entsoe_transparency": "load"})
    df = df.set_index("utc_timestamp").sort_index()
    df.index = pd.DatetimeIndex(df.index)  # already tz-aware UTC

    holidays = german_holidays(range(2014, 2021))

    train = df.loc[df.index < FORECAST_START]
    actual = df.loc[(df.index >= FORECAST_START) & (df.index < FORECAST_END), "load"]
    assert len(actual) == 168, f"expected 168 target hours, got {len(actual)}"

    X_train = make_features(train.index, holidays)
    y_train = train["load"].to_numpy()
    X_test = make_features(actual.index, holidays)

    model = HistGradientBoostingRegressor(
        loss="absolute_error",  # optimise for MAE, robust to load spikes
        max_iter=600,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=1.0,
        random_state=0,
    )
    model.fit(X_train, y_train)
    forecast = pd.Series(model.predict(X_test), index=actual.index, name="forecast")

    # Naive reference: average load by (day-of-week, hour) over the training years.
    tr = train.copy()
    local_tr = tr.index.tz_convert("Europe/Berlin")
    tr["dow"] = local_tr.dayofweek
    tr["hr"] = local_tr.hour
    profile = tr.groupby(["dow", "hr"])["load"].mean()
    local_te = actual.index.tz_convert("Europe/Berlin")
    naive = pd.Series(
        profile.loc[list(zip(local_te.dayofweek, local_te.hour))].to_numpy(),
        index=actual.index,
    )

    def scores(a: np.ndarray, p: np.ndarray) -> dict[str, float]:
        err = p - a
        return {
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err**2))),
            "MAPE": float(np.mean(np.abs(err / a)) * 100),
            "peak_err": float(p[np.argmax(a)] - a.max()),  # error at the weekly peak hour
        }

    m_model = scores(actual.to_numpy(), forecast.to_numpy())
    m_naive = scores(actual.to_numpy(), naive.to_numpy())

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(13, 6))
    idx = actual.index.tz_convert("Europe/Berlin")
    ax.plot(idx, actual.to_numpy() / 1000, color="#111827", lw=2.2, label="Actual")
    ax.plot(idx, forecast.to_numpy() / 1000, color="#2563eb", lw=2.0, ls="--",
            label=f"Forecast (MAPE {m_model['MAPE']:.1f}%)")
    ax.fill_between(idx, actual.to_numpy() / 1000, forecast.to_numpy() / 1000,
                    color="#2563eb", alpha=0.10)
    for d in pd.date_range("2020-01-01", "2020-01-07", tz="Europe/Berlin"):
        ax.axvline(d, color="#e5e7eb", lw=1, zorder=0)
    ax.axvspan(pd.Timestamp("2020-01-01", tz="Europe/Berlin"),
               pd.Timestamp("2020-01-02", tz="Europe/Berlin"),
               color="#fca5a5", alpha=0.18, label="New Year's Day (holiday)")
    ax.set_title("German electricity load — forecast vs actual, 1–7 Jan 2020",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Load (GW)")
    ax.set_xlabel("Local time (Europe/Berlin)")
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=idx.tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b", tz=idx.tz))
    ax.grid(axis="y", color="#f3f4f6")
    ax.legend(loc="upper left", framealpha=0.9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=130)

    # ---- report ----
    print(f"Trained on {len(train):,} hourly observations "
          f"({train.index.min().date()} to {train.index.max().date()}).")
    print(f"Forecast horizon: 168 hours, {FORECAST_START.date()} to "
          f"{(FORECAST_END - pd.Timedelta(hours=1)).date()}.\n")
    print("Accuracy on the target week (lower is better):")
    print(f"  {'metric':<8}{'model':>12}{'naive profile':>16}")
    for k in ("MAE", "RMSE"):
        print(f"  {k:<8}{m_model[k]:>10,.0f} MW{m_naive[k]:>13,.0f} MW")
    print(f"  {'MAPE':<8}{m_model['MAPE']:>11.2f} %{m_naive['MAPE']:>14.2f} %")
    acc = 100 - m_model["MAPE"]
    print(f"\nThe model's mean absolute percentage error is {m_model['MAPE']:.2f}% "
          f"— i.e. about {acc:.1f}% accurate on average across the 168 hours.")
    print(f"Average load that week was {actual.mean():,.0f} MW; "
          f"typical hourly miss was {m_model['MAE']:,.0f} MW.")
    print(f"Plot written to {PLOT_FILE.resolve()}")


if __name__ == "__main__":
    main()
