"""Week-ahead forecast of German hourly electricity load.

Forecasts the 168 hours 2020-01-01 00:00 to 2020-01-07 23:00 (UTC, the
dataset's own clock) from data up to and including 2019-12-31 23:00, the last
observed hour before the target week, using a gradient
boosted tree model on calendar, holiday and weekly-lag features.

Every lag is at least 168 hours, so the whole week is predicted directly from
values known at the forecast origin: no recursive feeding of predictions.
The same procedure is backtested on the first weeks of 2017, 2018 and 2019 so
the error on 2020 can be judged against what the method normally achieves.

Run:  python forecast_load.py
Outputs (next to this file): forecast_2020_w1.png, forecast_2020_w1.csv,
forecast_report.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import holidays
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "opsd_de_load.csv"
TARGET = "load"
LOCAL_TZ = "Europe/Berlin"
HORIZON_H = 168
FORECAST_START = pd.Timestamp("2020-01-01 00:00", tz="UTC")
BACKTEST_STARTS = [pd.Timestamp(f"{y}-01-01 00:00", tz="UTC") for y in (2017, 2018, 2019)]

# Weekly lags in UTC hours. All >= HORIZON_H so they are known for the whole target
# week. Around a DST change a UTC lag lands one local-clock hour off; no January
# target week crosses one.
WEEKLY_LAGS = (168, 336, 504, 672)
# 52 weeks: the same weekday and UTC hour one year earlier.
YEAR_LAG = 52 * 168

# Palette from the dataviz reference instance (light mode).
C_SURFACE = "#fcfcfb"
C_TEXT = "#0b0b0b"
C_TEXT_2 = "#52514e"
C_GRID = "#e6e5e1"
C_ACTUAL = "#2a78d6"
C_FORECAST = "#eb6834"


@dataclass(frozen=True)
class Metrics:
    mae_mw: float
    rmse_mw: float
    mape_pct: float
    max_abs_err_mw: float

    @classmethod
    def compute(cls, actual: pd.Series, predicted: pd.Series) -> Metrics:
        err = predicted.to_numpy(dtype=float) - actual.to_numpy(dtype=float)
        return cls(
            mae_mw=float(np.mean(np.abs(err))),
            rmse_mw=float(np.sqrt(np.mean(err**2))),
            mape_pct=float(np.mean(np.abs(err) / actual.to_numpy(dtype=float)) * 100),
            max_abs_err_mw=float(np.max(np.abs(err))),
        )


@dataclass(frozen=True)
class WeekResult:
    start: pd.Timestamp
    actual: pd.Series
    model: pd.Series
    naive_last_week: pd.Series
    naive_last_year: pd.Series

    @property
    def model_metrics(self) -> Metrics:
        return Metrics.compute(self.actual, self.model)

    @property
    def naive_week_metrics(self) -> Metrics:
        return Metrics.compute(self.actual, self.naive_last_week)

    @property
    def naive_year_metrics(self) -> Metrics:
        return Metrics.compute(self.actual, self.naive_last_year)


def load_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["utc_timestamp"])
    s = df.set_index("utc_timestamp")["DE_load_actual_entsoe_transparency"].rename(TARGET)
    s.index = pd.DatetimeIndex(s.index, name="utc_timestamp").tz_convert("UTC")
    full = pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC")
    if len(full) != len(s) or s.isna().any():
        raise ValueError("series must be hourly, gap-free and without NaNs")
    return s.astype(float)


def build_features(s: pd.Series) -> pd.DataFrame:
    """Calendar, holiday and lag features for every hour of the series.

    Calendar features use local (Berlin) time because demand follows the local
    clock, while the row index stays in UTC.
    """
    local = s.index.tz_convert(LOCAL_TZ)
    years = range(local.year.min(), local.year.max() + 1)
    de_holidays = holidays.country_holidays("DE", years=years)
    local_dates = pd.Series(local.date, index=s.index)

    f = pd.DataFrame(index=s.index)
    f["hour"] = local.hour
    f["dow"] = local.dayofweek
    f["is_weekend"] = (local.dayofweek >= 5).astype(int)
    f["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    f["is_holiday"] = local_dates.map(lambda d: int(d in de_holidays)).to_numpy()
    # Christmas to Epiphany: schools and many workplaces are closed, and 6 Jan is a
    # holiday in the three southern states. Nationwide holiday flags miss this dip.
    f["xmas_period"] = (
        ((local.month == 12) & (local.day >= 24)) | ((local.month == 1) & (local.day <= 6))
    ).astype(int)
    # Slow drift in the overall load level, in days since the start of the data.
    f["trend_days"] = (s.index - s.index.min()).total_seconds() / 86400.0

    for lag in WEEKLY_LAGS:
        f[f"lag_{lag}"] = s.shift(lag)
    f["lag_week_mean"] = f[[f"lag_{lag}" for lag in WEEKLY_LAGS]].mean(axis=1)
    f[f"lag_{YEAR_LAG}"] = s.shift(YEAR_LAG)
    # Whether the hour one week earlier was itself a holiday, so the model can
    # discount a lag drawn from Christmas week when predicting a working week.
    f["is_holiday_lag_168"] = f["is_holiday"].shift(168)
    f["xmas_period_lag_168"] = f["xmas_period"].shift(168)
    f[TARGET] = s
    return f


def fit_predict_week(features: pd.DataFrame, start: pd.Timestamp) -> pd.Series:
    """Train on every complete row before `start`; predict the following 168 hours."""
    end = start + pd.Timedelta(hours=HORIZON_H - 1)
    train = features.loc[features.index < start].dropna()
    test = features.loc[start:end]
    if len(test) != HORIZON_H:
        raise ValueError(f"expected {HORIZON_H} target rows from {start}, got {len(test)}")
    x_cols = [c for c in features.columns if c != TARGET]
    if test[x_cols].isna().any().any():
        raise ValueError("target week has missing feature values")

    model = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=0,
        deterministic=True,
        # One thread: on this machine the OpenMP build spends 10x longer on 4 threads.
        n_jobs=1,
        verbose=-1,
    )
    model.fit(train[x_cols], train[TARGET])
    pred = model.predict(test[x_cols])
    return pd.Series(np.asarray(pred, dtype=float), index=test.index, name="forecast")


def evaluate_week(features: pd.DataFrame, start: pd.Timestamp) -> WeekResult:
    pred = fit_predict_week(features, start)
    end = start + pd.Timedelta(hours=HORIZON_H - 1)
    week = features.loc[start:end]
    return WeekResult(
        start=start,
        actual=week[TARGET],
        model=pred,
        naive_last_week=week["lag_168"].rename("naive_last_week"),
        naive_last_year=week[f"lag_{YEAR_LAG}"].rename("naive_last_year"),
    )


def plot_week(result: WeekResult, out_path: Path) -> None:
    idx = result.actual.index
    fig, (ax, ax_err) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    fig.patch.set_facecolor(C_SURFACE)
    m = result.model_metrics

    ax.plot(idx, result.actual, color=C_ACTUAL, linewidth=2, label="Actual")
    ax.plot(idx, result.model, color=C_FORECAST, linewidth=2, label="Forecast")
    for label, series, color in (
        ("Actual", result.actual, C_ACTUAL),
        ("Forecast", result.model, C_FORECAST),
    ):
        ax.annotate(
            label,
            xy=(idx[-1], series.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=color,
            fontsize=10,
            va="center",
        )
    ax.set_ylabel("Load (MW)", color=C_TEXT)
    ax.set_title(
        "German hourly electricity load, 1 to 7 January 2020 (UTC)\n"
        f"Week-ahead forecast from 31 Dec 2019: MAE {m.mae_mw:,.0f} MW, "
        f"MAPE {m.mape_pct:.1f}%",
        loc="left",
        color=C_TEXT,
        fontsize=12,
    )
    ax.legend(loc="upper left", frameon=False)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))

    err = result.model - result.actual
    ax_err.axhline(0, color=C_TEXT_2, linewidth=1)
    ax_err.fill_between(idx, 0, err, color=C_FORECAST, alpha=0.25, linewidth=0)
    ax_err.plot(idx, err, color=C_FORECAST, linewidth=1.5)
    ax_err.set_ylabel("Forecast minus actual (MW)", color=C_TEXT, fontsize=9)
    ax_err.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v / 1000:+.0f}k"))

    for a in (ax, ax_err):
        a.set_facecolor(C_SURFACE)
        a.grid(True, color=C_GRID, linewidth=0.8)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(C_GRID)
        a.tick_params(colors=C_TEXT_2, labelsize=9)
        a.xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))  # type: ignore[no-untyped-call]
        a.xaxis.set_major_formatter(mdates.DateFormatter("%a %d Jan", tz="UTC"))  # type: ignore[no-untyped-call]
    ax_err.set_xlabel("Time (UTC)", color=C_TEXT)
    ax.set_xlim(idx[0], idx[-1] + pd.Timedelta(hours=12))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_SURFACE)
    plt.close(fig)


def metrics_row(label: str, m: Metrics) -> str:
    return (
        f"| {label} | {m.mae_mw:,.0f} | {m.rmse_mw:,.0f} | {m.mape_pct:.2f} | "
        f"{m.max_abs_err_mw:,.0f} |"
    )


def write_report(final: WeekResult, backtests: list[WeekResult], out_path: Path) -> None:
    header = (
        "| Week | MAE (MW) | RMSE (MW) | MAPE (%) | Max abs error (MW) |\n|---|---|---|---|---|"
    )
    lines = [
        "# Week-ahead load forecast: 1 to 7 January 2020",
        "",
        "Model: LightGBM on calendar, holiday and weekly-lag features, trained on data up",
        "to and including 2019-12-31 23:00 UTC (the last observed hour). All 168 hours",
        "are predicted directly from that data (no recursion).",
        "",
        "## Target week (2020)",
        "",
        header,
        metrics_row("Model", final.model_metrics),
        metrics_row("Naive: same hour last week", final.naive_week_metrics),
        metrics_row("Naive: same weekday and hour last year", final.naive_year_metrics),
        "",
        "## Per-day model error, 2020",
        "",
        "| Day (UTC) | MAE (MW) | Mean bias (MW) | Mean actual (MW) |",
        "|---|---|---|---|",
    ]
    err = final.model - final.actual
    for day, grp in err.groupby(err.index.date):
        lines.append(
            f"| {day} | {grp.abs().mean():,.0f} | {grp.mean():+,.0f} | "
            f"{final.actual.loc[grp.index].mean():,.0f} |"
        )
    lines += ["", "## Backtests: first week of earlier years, same procedure", "", header]
    for r in backtests:
        y = r.start.year
        lines.append(metrics_row(f"Model {y}", r.model_metrics))
        lines.append(metrics_row(f"Naive last week {y}", r.naive_week_metrics))
        lines.append(metrics_row(f"Naive last year {y}", r.naive_year_metrics))
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    series = load_series(DATA_PATH)
    features = build_features(series)

    backtests = [evaluate_week(features, start) for start in BACKTEST_STARTS]
    for r in backtests:
        print(f"backtest {r.start.year}: {json.dumps(r.model_metrics.__dict__)}")

    final = evaluate_week(features, FORECAST_START)
    print(f"forecast 2020: {json.dumps(final.model_metrics.__dict__)}")
    print(f"naive last week 2020: {json.dumps(final.naive_week_metrics.__dict__)}")
    print(f"naive last year 2020: {json.dumps(final.naive_year_metrics.__dict__)}")

    out = pd.DataFrame({"actual_mw": final.actual, "forecast_mw": final.model})
    out.index.name = "utc_timestamp"
    out.to_csv(HERE / "forecast_2020_w1.csv", float_format="%.1f")
    plot_week(final, HERE / "forecast_2020_w1.png")
    write_report(final, backtests, HERE / "forecast_report.md")


if __name__ == "__main__":
    main()
