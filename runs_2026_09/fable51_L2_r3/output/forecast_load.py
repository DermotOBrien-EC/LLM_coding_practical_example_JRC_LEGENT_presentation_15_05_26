"""Week-ahead forecast of German hourly electricity load.

Trains a gradient-boosted tree model on Open Power System Data hourly load
(2015 onwards), forecasts the 168 hours from 2020-01-01 00:00 to
2020-01-07 23:00 UTC without peeking at any actual inside that week, plots
the forecast against the actuals, and reports accuracy against two naive
baselines. The same procedure is re-run for the first week of January 2019
as an out-of-sample sanity check, so the 2020 accuracy can be judged as
typical or not.

Usage:
    python forecast_load.py [--data opsd_de_load.csv] [--out .]
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

# The model is small (35k rows, 21 features) and fits in seconds on one core.
# With OpenMP free to use every core, sklearn's boosting spent ~60 s per 50
# iterations on a 15-core macOS box (spin-wait contention), so one thread is
# both faster and deterministic. Must be set before sklearn is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

LOCAL_TZ = "Europe/Berlin"
HORIZON_HOURS = 168
# Lags in hours. Every lag is >= 168 so each one is already observed at the
# forecast cutoff for every hour of the week being forecast.
LAG_HOURS = (168, 336, 504, 672, 8736, 8760)

# Reference palette from the dataviz skill (light mode).
COLOR_ACTUAL = "#2a78d6"
COLOR_FORECAST = "#eb6834"
COLOR_BASELINE = "#8a8984"
COLOR_CONTEXT = "#c3c2b7"
COLOR_TEXT = "#0b0b0b"
COLOR_TEXT_2 = "#52514e"
COLOR_GRID = "#e6e5e1"
COLOR_SURFACE = "#fcfcfb"
COLOR_HOLIDAY = "#f0efec"


# --------------------------------------------------------------------------
# German public holidays (computed, no external package)
# --------------------------------------------------------------------------
def easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) / 451
    month = (h + l - 7 * int(m) + 114) // 31
    day = (h + l - 7 * int(m) + 114) % 31 + 1
    return date(year, month, day)


def national_holidays(year: int) -> set[date]:
    e = easter_sunday(year)
    days = {
        date(year, 1, 1),
        e - timedelta(days=2),  # Good Friday
        e + timedelta(days=1),  # Easter Monday
        date(year, 5, 1),
        e + timedelta(days=39),  # Ascension
        e + timedelta(days=50),  # Whit Monday
        date(year, 10, 3),
        date(year, 12, 25),
        date(year, 12, 26),
    }
    if year == 2017:
        days.add(date(year, 10, 31))  # Reformation 500th anniversary, nationwide once
    return days


def regional_holidays(year: int) -> set[date]:
    """Holidays observed in only some states; they still dent national load."""
    e = easter_sunday(year)
    nov23 = date(year, 11, 23)
    buss_und_bettag = nov23 - timedelta(days=(nov23.weekday() - 2) % 7 or 7)
    return {
        date(year, 1, 6),  # Epiphany (BW, BY, ST)
        e + timedelta(days=60),  # Corpus Christi
        date(year, 8, 15),  # Assumption (parts of BY, SL)
        date(year, 10, 31),  # Reformation Day (northern and eastern states)
        date(year, 11, 1),  # All Saints (western and southern states)
        buss_und_bettag,  # Repentance Day (SN)
    }


def bridge_days(year: int) -> set[date]:
    """Not holidays, but most of industry is off: Christmas Eve, New Year's Eve."""
    return {date(year, 12, 24), date(year, 12, 31)}


# --------------------------------------------------------------------------
# Data and features
# --------------------------------------------------------------------------
def load_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["utc_timestamp"])
    s = df.set_index("utc_timestamp")["DE_load_actual_entsoe_transparency"]
    s = s.sort_index()
    s.index = s.index.tz_convert("UTC")
    expected = pd.date_range(s.index[0], s.index[-1], freq="h", tz="UTC")
    if not s.index.equals(expected):
        raise ValueError("series is not a complete regular hourly index")
    if s.isna().any():
        raise ValueError(f"series has {int(s.isna().sum())} missing values")
    return s.rename("load")


def build_features(s: pd.Series) -> pd.DataFrame:
    local = s.index.tz_convert(LOCAL_TZ)
    local_date = pd.Series(local.date, index=s.index)
    years = range(local.year.min() - 1, local.year.max() + 2)
    nat = set().union(*(national_holidays(y) for y in years))
    reg = set().union(*(regional_holidays(y) for y in years))
    bridge = set().union(*(bridge_days(y) for y in years))

    f = pd.DataFrame(index=s.index)
    f["hour"] = local.hour
    f["dow"] = local.dayofweek
    f["month"] = local.month
    doy = local.dayofyear.to_numpy()
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["is_weekend"] = (local.dayofweek >= 5).astype(int)
    f["hol_nat"] = local_date.isin(nat).astype(int).to_numpy()
    f["hol_reg"] = local_date.isin(reg).astype(int).to_numpy()
    f["bridge"] = local_date.isin(bridge).astype(int).to_numpy()

    # Distance in days to the nearest national holiday, clipped to a week each
    # side; captures the low-load run-up to Christmas and the ramp after New Year.
    hol_dates = np.array(sorted(nat), dtype="datetime64[D]")
    d = np.array(local_date.to_numpy(), dtype="datetime64[D]")
    pos = np.searchsorted(hol_dates, d)
    next_hol = hol_dates[np.clip(pos, 0, len(hol_dates) - 1)]
    prev_hol = hol_dates[np.clip(pos - 1, 0, len(hol_dates) - 1)]
    f["days_to_hol"] = np.clip((next_hol - d).astype(int), 0, 7)
    f["days_since_hol"] = np.clip((d - prev_hol).astype(int), 0, 7)

    for lag in LAG_HOURS:
        f[f"lag_{lag}"] = s.shift(lag).to_numpy()
    # Average load over the most recent fully observed week.
    f["prev_week_mean"] = s.shift(HORIZON_HOURS).rolling(HORIZON_HOURS).mean().to_numpy()
    # Whether the hour one week earlier fell on a holiday or bridge day, so the
    # model can discount a lag taken from the Christmas trough.
    for col in ("hol_nat", "hol_reg", "bridge"):
        f[f"{col}_lag168"] = f[col].shift(HORIZON_HOURS)
    return f


# --------------------------------------------------------------------------
# Forecasting and scoring
# --------------------------------------------------------------------------
@dataclass
class WeekForecast:
    cutoff: pd.Timestamp
    frame: pd.DataFrame  # index: target hours; columns: actual, forecast, naive_week, naive_year
    n_train: int


def forecast_week(s: pd.Series, features: pd.DataFrame, cutoff: pd.Timestamp) -> WeekForecast:
    """Train on everything strictly before `cutoff`, forecast the next 168 hours."""
    target_index = pd.date_range(cutoff, periods=HORIZON_HOURS, freq="h", tz="UTC")
    if not target_index.isin(s.index).all():
        raise ValueError("target week is not fully covered by the data")

    train_mask = (features.index < cutoff) & features.notna().all(axis=1)
    x_train = features.loc[train_mask]
    y_train = s.loc[train_mask]
    x_test = features.loc[target_index]
    if x_test.isna().any().any():
        raise ValueError("feature leakage guard: a target-week feature is unobserved")

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=800,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=0,
    )
    model.fit(x_train, y_train)

    frame = pd.DataFrame(
        {
            "actual": s.loc[target_index],
            "forecast": model.predict(x_test),
            "naive_week": x_test["lag_168"].to_numpy(),
            "naive_year": x_test["lag_8736"].to_numpy(),
        },
        index=target_index,
    )
    return WeekForecast(cutoff=cutoff, frame=frame, n_train=int(train_mask.sum()))


def score(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    err = pred - actual
    return {
        "mae_mw": float(err.abs().mean()),
        "rmse_mw": float(np.sqrt((err**2).mean())),
        "mape_pct": float((err.abs() / actual).mean() * 100),
        "bias_mw": float(err.mean()),
        "max_abs_err_mw": float(err.abs().max()),
    }


def daily_mape(frame: pd.DataFrame) -> pd.Series:
    """MAPE per UTC day, matching how the forecast window itself is defined."""
    utc_day = frame.index.strftime("%a %d %b")
    ape = (frame["forecast"] - frame["actual"]).abs() / frame["actual"] * 100
    return ape.groupby(utc_day, sort=False).mean()


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def plot_week(s: pd.Series, wf: WeekForecast, out_png: Path) -> None:
    frame = wf.frame
    local = frame.index.tz_convert(LOCAL_TZ)
    context_start = wf.cutoff - pd.Timedelta(days=3)
    context = s.loc[context_start : wf.cutoff]
    context_local = context.index.tz_convert(LOCAL_TZ)
    err = frame["forecast"] - frame["actual"]

    fig, (ax, ax_err) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True, height_ratios=[3, 1.2], facecolor=COLOR_SURFACE
    )
    for a in (ax, ax_err):
        a.set_facecolor(COLOR_SURFACE)
        a.grid(True, color=COLOR_GRID, linewidth=0.8)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(COLOR_GRID)
        a.tick_params(colors=COLOR_TEXT_2, labelsize=9)

    # Shade holiday days in the forecast week.
    nat = national_holidays(local[0].year)
    reg = regional_holidays(local[0].year)
    for day in sorted(set(local.date)):
        if day in nat or day in reg:
            start = pd.Timestamp(day, tz=LOCAL_TZ)
            label = "public holiday" if day in nat else "regional holiday"
            ax.axvspan(start, start + pd.Timedelta(days=1), color=COLOR_HOLIDAY, zorder=0)
            ax.text(
                start + pd.Timedelta(hours=12),
                0.985,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8.5,
                color=COLOR_TEXT_2,
            )

    ax.plot(
        context_local,
        context.to_numpy(),
        color=COLOR_CONTEXT,
        linewidth=2,
        label="Actual, before cutoff",
    )
    ax.plot(
        local,
        frame["naive_week"],
        color=COLOR_BASELINE,
        linewidth=1.2,
        linestyle=(0, (3, 2)),
        label="Same hour last week (naive)",
    )
    ax.plot(local, frame["actual"], color=COLOR_ACTUAL, linewidth=2, label="Actual")
    ax.plot(local, frame["forecast"], color=COLOR_FORECAST, linewidth=2, label="Forecast")
    ax.axvline(wf.cutoff.tz_convert(LOCAL_TZ), color=COLOR_TEXT_2, linewidth=1, linestyle=":")
    ax.text(
        wf.cutoff.tz_convert(LOCAL_TZ) - pd.Timedelta(hours=2),
        0.03,
        "forecast cutoff",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=COLOR_TEXT_2,
    )
    ax.set_ylabel("Load (MW)", color=COLOR_TEXT_2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=COLOR_TEXT_2, ncols=2)

    m = score(frame["actual"], frame["forecast"])
    ax.set_title(
        f"German electricity load, week of {local[0]:%d %b %Y}: forecast vs actual\n"
        f"MAPE {m['mape_pct']:.2f} %, MAE {m['mae_mw']:,.0f} MW, RMSE {m['rmse_mw']:,.0f} MW",
        loc="left",
        fontsize=12,
        color=COLOR_TEXT,
    )

    ax_err.fill_between(local, err, 0, color=COLOR_FORECAST, alpha=0.25, linewidth=0)
    ax_err.plot(local, err, color=COLOR_FORECAST, linewidth=1.5)
    ax_err.axhline(0, color=COLOR_TEXT_2, linewidth=1)
    ax_err.set_ylabel("Forecast minus actual (MW)", color=COLOR_TEXT_2, fontsize=9)
    ax_err.xaxis.set_major_locator(mdates.DayLocator(tz=LOCAL_TZ))
    ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b", tz=LOCAL_TZ))
    ax_err.set_xlim(context_local[0], local[-1] + pd.Timedelta(hours=1))
    ax_err.set_xlabel(f"Time ({LOCAL_TZ})", color=COLOR_TEXT_2, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def report(name: str, wf: WeekForecast) -> dict[str, object]:
    frame = wf.frame
    rows = {
        "model": score(frame["actual"], frame["forecast"]),
        "naive_week": score(frame["actual"], frame["naive_week"]),
        "naive_year": score(frame["actual"], frame["naive_year"]),
    }
    print(f"\n=== {name}: cutoff {wf.cutoff:%Y-%m-%d %H:%M} UTC, {wf.n_train:,} training hours ===")
    print(
        f"{'':12s}{'MAE MW':>10s}{'RMSE MW':>10s}{'MAPE %':>9s}{'bias MW':>10s}{'max|err| MW':>13s}"
    )
    for key, m in rows.items():
        print(
            f"{key:12s}{m['mae_mw']:10,.0f}{m['rmse_mw']:10,.0f}{m['mape_pct']:9.2f}"
            f"{m['bias_mw']:10,.0f}{m['max_abs_err_mw']:13,.0f}"
        )
    print("Model MAPE by UTC day:")
    for day, v in daily_mape(frame).items():
        print(f"  {day}: {v:.2f} %")
    return {
        "cutoff_utc": wf.cutoff.isoformat(),
        "n_train_hours": wf.n_train,
        "metrics": rows,
        "daily_mape_pct": {k: float(v) for k, v in daily_mape(frame).items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("opsd_de_load.csv"))
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    s = load_series(args.data)
    features = build_features(s)
    print(f"Loaded {len(s):,} hourly values, {s.index[0]:%Y-%m-%d} to {s.index[-1]:%Y-%m-%d} (UTC)")
    print(f"Features: {', '.join(features.columns)}")

    results: dict[str, object] = {}
    check = forecast_week(s, features, pd.Timestamp("2019-01-01 00:00", tz="UTC"))
    results["jan_2019_check"] = report("Sanity check, first week of January 2019", check)

    main_week = forecast_week(s, features, pd.Timestamp("2020-01-01 00:00", tz="UTC"))
    results["jan_2020"] = report("Target, first week of January 2020", main_week)

    csv_path = args.out / "forecast_2020_w1.csv"
    png_path = args.out / "forecast_2020_w1.png"
    json_path = args.out / "forecast_metrics.json"
    out = main_week.frame.copy()
    out["error"] = out["forecast"] - out["actual"]
    out.index.name = "utc_timestamp"
    out.round(1).to_csv(csv_path)
    plot_week(s, main_week, png_path)
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {csv_path}, {png_path}, {json_path}")


if __name__ == "__main__":
    main()
