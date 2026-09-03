#!/usr/bin/env python3
"""Forecast German hourly electricity load for the first week of January 2020.

The forecast is a single-origin, 168-hour-ahead prediction: everything is
predicted from one cut-off (the last hour of 2019, local time) without ever
seeing an actual value from the target week. Accuracy is reported against two
naive baselines and against the same pipeline run on earlier January weeks, so
the headline number can be read in context rather than in isolation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import holidays
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CSV_PATH: Final = Path(__file__).parent / "opsd_de_load.csv"
OUT_DIR: Final = Path(__file__).parent / "outputs"
LOAD_COL: Final = "DE_load_actual_entsoe_transparency"
TZ: Final = "Europe/Berlin"

HORIZON: Final = 168  # hours forecast from a single origin

# Only lags >= HORIZON are knowable at the origin for every hour of the week.
# 8736 h = 52 weeks (same hour, same weekday, a year earlier).
# 8760 h = 365 days (same hour, same calendar date, a year earlier).
LAGS: Final[tuple[int, ...]] = (168, 169, 170, 336, 504, 672, 8736, 8760)

CATEGORICAL: Final[list[str]] = ["hour_of_week", "dayofweek", "month"]


def _make_model() -> lgb.LGBMRegressor:
    """The gradient-boosting model. Fixed hyperparameters, never tuned on the target week."""
    return lgb.LGBMRegressor(
        objective="l2",
        learning_rate=0.03,
        n_estimators=1500,
        num_leaves=64,
        min_child_samples=30,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )


# Backtest origins (local dates whose 00:00 starts the forecast week), then the
# real target. Earlier origins exist to show the 2020 number is not a lucky draw.
BACKTEST_STARTS: Final[tuple[str, ...]] = ("2017-01-01", "2018-01-01", "2019-01-01")
TARGET_START: Final = "2020-01-01"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_series(path: Path = CSV_PATH) -> pd.Series:
    """Read the OPSD file as a load series indexed in German local time."""
    frame = pd.read_csv(path, parse_dates=["utc_timestamp"])
    series = frame.set_index("utc_timestamp")[LOAD_COL].astype(float)
    series.index = series.index.tz_convert(TZ)
    series = series.sort_index()
    series.name = "load_mw"

    # Lags below are positional shifts, which only equal time shifts on a gap-free
    # hourly index. Assert rather than assume.
    deltas = series.index.to_series().diff().dropna().unique()
    if len(deltas) != 1 or deltas[0] != pd.Timedelta(hours=1):
        raise ValueError(f"expected a gap-free hourly index, got steps {deltas}")
    if series.isna().any():
        raise ValueError("load series contains missing values")
    return series


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def _german_holiday_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Nationwide holiday flags plus the share of states observing each day.

    The state share matters for this particular week: 6 January (Epiphany) is a
    public holiday in Baden-Wuerttemberg, Bavaria and Saxony-Anhalt but nowhere
    else, so it depresses load without being a nationwide holiday.
    """
    years = sorted({int(y) for y in index.year.unique()})
    years = list(range(min(years) - 1, max(years) + 2))

    national = holidays.country_holidays("DE", years=years)
    # 'Augsburg' is a city subdivision, not a Bundesland; drop it so the share is
    # a share of the 16 states.
    states = [s for s in holidays.country_holidays("DE").subdivisions if len(s) == 2]
    per_state = [holidays.country_holidays("DE", subdiv=s, years=years) for s in states]

    dates = pd.Series(index.date, index=index)
    unique_dates = pd.Index(sorted(set(dates)))

    is_nat = pd.Series([d in national for d in unique_dates], index=unique_dates)
    share = pd.Series(
        [sum(d in cal for cal in per_state) / len(per_state) for d in unique_dates],
        index=unique_dates,
    )

    one_day = pd.Timedelta(days=1)
    prev_nat = pd.Series(
        [(d - one_day).date() in national for d in pd.to_datetime(unique_dates)],
        index=unique_dates,
    )
    next_nat = pd.Series(
        [(d + one_day).date() in national for d in pd.to_datetime(unique_dates)],
        index=unique_dates,
    )

    out = pd.DataFrame(index=index)
    out["is_holiday"] = dates.map(is_nat).astype(float).to_numpy()
    out["holiday_state_share"] = dates.map(share).astype(float).to_numpy()
    out["holiday_yesterday"] = dates.map(prev_nat).astype(float).to_numpy()
    out["holiday_tomorrow"] = dates.map(next_nat).astype(float).to_numpy()

    weekday = index.dayofweek.to_numpy()
    out["is_bridge_day"] = (
        (weekday < 5)
        & (out["is_holiday"].to_numpy() == 0)
        & ((out["holiday_yesterday"].to_numpy() == 1) | (out["holiday_tomorrow"].to_numpy() == 1))
    ).astype(float)
    return out


def _workday_run_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Where each day sits in its contiguous run of working days.

    The German Brueckentag effect: a midweek public holiday strands a short run
    of working days against the weekend and much of the workforce takes the whole
    block off, so load stays low on days that are not themselves holidays.
    ``is_bridge_day`` only fires on days adjacent to a holiday, which catches
    Thu 2 Jan 2020 but not Fri 3 Jan. Run length generalises that.

    Kept as a separate feature block because it was added after the fact and
    accepted on the 2017-2019 backtest weeks (see experiment_bridge.py).
    """
    days = pd.DatetimeIndex(sorted({pd.Timestamp(d) for d in index.date}))
    span = pd.date_range(days.min() - pd.Timedelta(days=10), days.max() + pd.Timedelta(days=10))
    years = list(range(int(span.year.min()), int(span.year.max()) + 1))
    national = holidays.country_holidays("DE", years=years)
    working = np.array([d.dayofweek < 5 and d.date() not in national for d in span])

    run_len = np.zeros(len(span))
    days_into = np.zeros(len(span))
    days_left = np.zeros(len(span))
    i = 0
    while i < len(span):
        if not working[i]:
            i += 1
            continue
        j = i
        while j < len(span) and working[j]:
            j += 1
        for k in range(i, j):
            run_len[k], days_into[k], days_left[k] = j - i, k - i + 1, j - k
        i = j

    table = pd.DataFrame(
        {"workday_run_len": run_len, "days_into_run": days_into, "days_left_in_run": days_left},
        index=pd.Index(span.date),
    )
    return table.loc[index.date].set_index(index)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Everything knowable about a timestamp without seeing any load value."""
    out = pd.DataFrame(index=index)
    hour = index.hour.to_numpy()
    weekday = index.dayofweek.to_numpy()
    doy = index.dayofyear.to_numpy()
    days_in_year = np.where(np.asarray(index.is_leap_year), 366, 365)

    out["hour"] = hour
    out["dayofweek"] = weekday
    out["hour_of_week"] = weekday * 24 + hour
    out["month"] = index.month.to_numpy()
    out["dayofyear"] = doy
    out["is_weekend"] = (weekday >= 5).astype(float)

    # Annual shape (heating/daylight), daily shape.
    frac_year = doy / days_in_year
    for k in (1, 2, 3):
        out[f"year_sin{k}"] = np.sin(2 * np.pi * k * frac_year)
        out[f"year_cos{k}"] = np.cos(2 * np.pi * k * frac_year)
    for k in (1, 2):
        out[f"day_sin{k}"] = np.sin(2 * np.pi * k * hour / 24)
        out[f"day_cos{k}"] = np.cos(2 * np.pi * k * hour / 24)

    # Signed distance in days from 1 January, the dominant effect in this window.
    # 99 is an out-of-season sentinel; trees split on it happily.
    out["days_from_newyear"] = np.where(
        doy <= 20, doy - 1, np.where(doy >= 350, doy - days_in_year - 1, 99)
    ).astype(float)

    return pd.concat([out, _german_holiday_features(index), _workday_run_features(index)], axis=1)


def lag_features(load: pd.Series, origin: pd.Timestamp) -> pd.DataFrame:
    """Lagged-load features built from history that stops at ``origin``.

    Masking to NaN after the origin makes leakage structurally impossible: a lag
    shorter than the horizon would surface as a missing value for the forecast
    week, not as a peek at the answer.
    """
    if min(LAGS) < HORIZON:
        raise ValueError(f"lags {LAGS} shorter than horizon {HORIZON} would leak")

    history = load.mask(load.index > origin)
    out = pd.DataFrame(index=load.index)

    for lag in LAGS:
        out[f"lag_{lag}"] = history.shift(lag)

    # Level and shape of the most recent fully-known week, and of the same week a
    # year earlier. Both windows end at or before the origin for every target hour.
    shifted_week = history.shift(HORIZON)
    out["roll168_mean_lag168"] = shifted_week.rolling(168, min_periods=120).mean()
    out["roll24_mean_lag168"] = shifted_week.rolling(24, min_periods=18).mean()
    out["roll168_std_lag168"] = shifted_week.rolling(168, min_periods=120).std()

    shifted_year = history.shift(8736)
    out["roll168_mean_lag8736"] = shifted_year.rolling(168, min_periods=120).mean()

    # Year-on-year level ratio: lets the model rescale last year's shape to this
    # year's level without needing to extrapolate a trend (trees cannot).
    out["yoy_level_ratio"] = out["roll168_mean_lag168"] / out["roll168_mean_lag8736"]
    return out


def build_matrix(load: pd.Series, origin: pd.Timestamp) -> pd.DataFrame:
    features = pd.concat([calendar_features(load.index), lag_features(load, origin)], axis=1)
    for col in CATEGORICAL:
        features[col] = features[col].astype("category")
    return features


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    """Accuracy of one 168-hour forecast."""

    label: str
    mae_mw: float
    rmse_mw: float
    mape_pct: float
    bias_mw: float
    max_abs_err_mw: float
    peak_err_mw: float
    r2: float

    @classmethod
    def compute(cls, label: str, actual: pd.Series, forecast: pd.Series) -> Metrics:
        err = (forecast - actual).to_numpy(dtype=float)
        act = actual.to_numpy(dtype=float)
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((act - act.mean()) ** 2))
        peak_hour = int(np.argmax(act))
        return cls(
            label=label,
            mae_mw=float(np.mean(np.abs(err))),
            rmse_mw=float(np.sqrt(np.mean(err**2))),
            mape_pct=float(np.mean(np.abs(err / act)) * 100),
            bias_mw=float(np.mean(err)),
            max_abs_err_mw=float(np.max(np.abs(err))),
            peak_err_mw=float(err[peak_hour]),
            r2=1.0 - ss_res / ss_tot,
        )


# --------------------------------------------------------------------------
# Forecast for one origin
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WeekResult:
    """One forecast week: the series involved and every model's accuracy."""

    start: pd.Timestamp
    origin: pd.Timestamp
    frame: pd.DataFrame  # actual + one column per model
    metrics: list[Metrics]

    @property
    def model(self) -> Metrics:
        return next(m for m in self.metrics if m.label == "LightGBM")


def forecast_week(load: pd.Series, start: str) -> WeekResult:
    """Train on everything before ``start`` and forecast the following 168 hours."""
    week_start = pd.Timestamp(start, tz=TZ)
    week_index = pd.date_range(week_start, periods=HORIZON, freq="h", tz=TZ)
    missing = week_index.difference(load.index)
    if len(missing):
        raise ValueError(f"target week is not fully present in the data: {missing[:3]}")

    origin = week_start - pd.Timedelta(hours=1)
    features = build_matrix(load, origin)

    train_mask = load.index <= origin
    x_train, y_train = features.loc[train_mask], load.loc[train_mask]
    x_test = features.loc[week_index]
    actual = load.loc[week_index]

    model = _make_model()
    model.fit(x_train, y_train, categorical_feature=CATEGORICAL)
    prediction = pd.Series(model.predict(x_test), index=week_index, name="LightGBM")

    # Baselines, from the same masked history for the same reason.
    history = load.mask(load.index > origin)
    naive_week = pd.Series(
        history.shift(168).loc[week_index].to_numpy(), index=week_index, name="Same hour last week"
    )
    naive_year = pd.Series(
        history.shift(8736).loc[week_index].to_numpy(),
        index=week_index,
        name="Same hour, 52 weeks ago",
    )

    frame = pd.DataFrame({"actual": actual})
    for series in (prediction, naive_week, naive_year):
        frame[str(series.name)] = series

    metrics = [
        Metrics.compute(str(s.name), actual, s) for s in (prediction, naive_week, naive_year)
    ]
    return WeekResult(start=week_start, origin=origin, frame=frame, metrics=metrics)


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    """Chart chrome for one colour mode."""

    name: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    actual: str
    forecast: str
    naive: str
    wash: str


LIGHT: Final = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    actual="#2a78d6",
    forecast="#eb6834",
    naive="#1baf7a",
    wash="#f0efec",
)

DARK: Final = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    actual="#3987e5",
    forecast="#d95926",
    naive="#199e70",
    wash="#383835",
)


def _style_axes(ax: Axes, theme: Theme) -> None:
    ax.set_facecolor(theme.surface)
    ax.grid(axis="y", color=theme.grid, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.axis)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=theme.muted, labelsize=9, length=0)


def plot_week(result: WeekResult, theme: Theme, path: Path) -> None:
    """Actual vs forecast over the week, with the hourly error beneath it."""
    frame = result.frame
    hours = np.arange(len(frame))
    actual = frame["actual"].to_numpy()
    forecast = frame["LightGBM"].to_numpy()
    naive = frame["Same hour last week"].to_numpy()
    error = forecast - actual
    model = result.model

    fig: Figure
    fig, (ax, ax_err) = plt.subplots(
        2,
        1,
        figsize=(13, 7.6),
        height_ratios=(3, 1),
        sharex=True,
        gridspec_kw={"hspace": 0.14},
    )
    fig.patch.set_facecolor(theme.surface)
    for axis in (ax, ax_err):
        _style_axes(axis, theme)

    # Day bands: shade the holidays so the two dips are explained, not mysterious.
    days = pd.Series(frame.index.date).unique()
    holiday_note = {0: "New Year's Day", 5: "Epiphany (BW, BY, ST)"}
    for i, day in enumerate(days):
        left = i * 24
        if i in holiday_note:
            ax.axvspan(left, left + 24, color=theme.wash, zorder=0, linewidth=0)
        if i:
            for axis in (ax, ax_err):
                axis.axvline(left, color=theme.grid, linewidth=0.8, zorder=1)

    ax.plot(
        hours,
        naive,
        color=theme.naive,
        linewidth=1.4,
        linestyle=(0, (4, 3)),
        zorder=2,
        label="Baseline: same hour last week",
    )
    ax.plot(hours, actual, color=theme.actual, linewidth=2.0, zorder=4, label="Actual")
    ax.plot(hours, forecast, color=theme.forecast, linewidth=2.0, zorder=3, label="Forecast")

    # Direct labels at the right edge, so identity never rests on colour alone.
    for value, colour, text in (
        (actual[-1], theme.actual, "Actual"),
        (forecast[-1], theme.forecast, "Forecast"),
    ):
        ax.annotate(
            text,
            xy=(len(hours) - 1, value),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            fontweight="medium",
            color=colour,
        )

    # Headroom below the data so the band captions never sit on the traces.
    low, high = float(np.min([actual.min(), forecast.min(), naive.min()])), ax.get_ylim()[1]
    ax.set_ylim(low - (high - low) * 0.10, high)

    for i, note in holiday_note.items():
        ax.annotate(
            note,
            xy=(i * 24 + 12, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=theme.muted,
        )

    ax.set_ylabel("Load (MW)", color=theme.ink_secondary, fontsize=10)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:,.0f} GW")
    # Legend above the plot area: inside it, it would sit on the first day's band.
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        fontsize=9.5,
        labelcolor=theme.ink_secondary,
        handlelength=2.2,
        ncols=3,
        borderaxespad=0,
    )
    ax.set_title(
        f"German electricity load, {result.start:%-d}–{frame.index[-1]:%-d %B %Y}",
        color=theme.ink,
        fontsize=15,
        fontweight="semibold",
        loc="left",
        pad=52,
    )
    ax.annotate(
        f"168-hour forecast from a single origin at {result.origin:%-d %b %Y %H:%M %Z}  ·  "
        f"MAPE {model.mape_pct:.2f}%  ·  MAE {model.mae_mw:,.0f} MW",
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 30),
        textcoords="offset points",
        fontsize=10,
        color=theme.ink_secondary,
    )

    # Error panel: one series, so sign is read off the zero baseline, not colour.
    ax_err.fill_between(hours, 0, error, color=theme.forecast, alpha=0.22, linewidth=0, zorder=2)
    ax_err.plot(hours, error, color=theme.forecast, linewidth=1.4, zorder=3)
    ax_err.axhline(0, color=theme.axis, linewidth=1.0, zorder=4)
    ax_err.set_ylabel("Forecast − actual", color=theme.ink_secondary, fontsize=10)
    ax_err.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:+,.0f} GW" if v else "0")
    limit = float(np.max(np.abs(error))) * 1.25
    ax_err.set_ylim(-limit, limit)

    ax_err.set_xlim(-1, len(hours))
    ax_err.set_xticks([i * 24 + 12 for i in range(len(days))])
    ax_err.set_xticklabels([pd.Timestamp(d).strftime("%a %-d %b") for d in days])
    ax_err.set_xlabel("German local time (CET)", color=theme.ink_secondary, fontsize=10)

    fig.text(
        0.5,
        0.015,
        "Data: Open Power System Data, ENTSO-E transparency platform. "
        "Model trained only on data up to the forecast origin.",
        ha="center",
        fontsize=8.5,
        color=theme.muted,
    )
    fig.subplots_adjust(left=0.07, right=0.90, top=0.84, bottom=0.10)
    fig.savefig(path, dpi=160, facecolor=theme.surface)
    plt.close(fig)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _table(rows: list[Metrics]) -> str:
    header = f"{'model':<26}{'MAPE %':>9}{'MAE MW':>10}{'RMSE MW':>10}{'bias MW':>10}{'R2':>8}"
    lines = [header, "-" * len(header)]
    for m in rows:
        lines.append(
            f"{m.label:<26}{m.mape_pct:>9.2f}{m.mae_mw:>10,.0f}"
            f"{m.rmse_mw:>10,.0f}{m.bias_mw:>+10,.0f}{m.r2:>8.3f}"
        )
    return "\n".join(lines)


def daily_breakdown(result: WeekResult) -> pd.DataFrame:
    frame = result.frame
    err = frame["LightGBM"] - frame["actual"]
    grouped = pd.DataFrame(
        {
            "actual_mean_mw": frame["actual"],
            "forecast_mean_mw": frame["LightGBM"],
            "abs_err": err.abs(),
            "ape": (err / frame["actual"]).abs() * 100,
        }
    ).groupby(frame.index.date)
    out = grouped.agg(
        actual_mean_mw=("actual_mean_mw", "mean"),
        forecast_mean_mw=("forecast_mean_mw", "mean"),
        mae_mw=("abs_err", "mean"),
        mape_pct=("ape", "mean"),
    )
    out.index = pd.Index(
        [pd.Timestamp(d).strftime("%a %d %b") for d in out.index], name="day (local)"
    )
    return out.round(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=TARGET_START, help="first local day of the target week")
    parser.add_argument("--no-backtest", action="store_true", help="skip earlier January weeks")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    load = load_series()
    print(
        f"Loaded {len(load):,} hourly observations, "
        f"{load.index[0]:%Y-%m-%d %H:%M %Z} to {load.index[-1]:%Y-%m-%d %H:%M %Z}\n"
    )

    backtests: list[WeekResult] = []
    if not args.no_backtest:
        print("Backtest: identical pipeline on earlier January weeks")
        for start in BACKTEST_STARTS:
            week = forecast_week(load, start)
            backtests.append(week)
            print(
                f"  {week.start:%Y-%m-%d}  LightGBM MAPE {week.model.mape_pct:5.2f}%   "
                f"MAE {week.model.mae_mw:6,.0f} MW"
            )
        print()

    result = forecast_week(load, args.start)
    print(f"Target week: {result.start:%Y-%m-%d} to {result.frame.index[-1]:%Y-%m-%d} ({TZ})")
    print(f"Forecast origin: {result.origin:%Y-%m-%d %H:%M %Z} (no later data used)\n")
    print(_table(result.metrics))
    print("\nPer-day accuracy of the LightGBM forecast:")
    print(daily_breakdown(result).to_string())

    stem = f"forecast_{result.start:%Y-%m-%d}"
    result.frame.assign(error_mw=result.frame["LightGBM"] - result.frame["actual"]).to_csv(
        OUT_DIR / f"{stem}.csv", float_format="%.1f"
    )
    for theme in (LIGHT, DARK):
        suffix = "" if theme.name == "light" else f"_{theme.name}"
        plot_week(result, theme, OUT_DIR / f"{stem}{suffix}.png")

    payload = {
        "target_week_start": str(result.start),
        "forecast_origin": str(result.origin),
        "horizon_hours": HORIZON,
        "training_observations": int((load.index <= result.origin).sum()),
        "target_week": [asdict(m) for m in result.metrics],
        "backtest": {str(w.start.date()): asdict(w.model) for w in backtests},
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {stem}.csv, {stem}.png, {stem}_dark.png and metrics.json to {OUT_DIR}/")


if __name__ == "__main__":
    main()
