"""Forecast German hourly electricity load for the first week of January 2020.

Trains on Open Power System Data hourly load (2015-01-01 .. 2019-12-31), forecasts
the 168 hours of 2020-01-01 .. 2020-01-07 (Europe/Berlin local days), scores the
forecast against the recorded actuals and plots the two against each other.

The forecast is a true 168-hour-ahead forecast: every predictor is either a calendar
value known in advance or a lag of at least 168 hours, so no value from inside the
horizon is ever an input.

Usage:
    python forecast_de_load.py             # backtest, fit, forecast, plot, report
    python forecast_de_load.py --no-backtest   # skip model selection, use the default
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import holidays
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "opsd_de_load.csv"
VALUE_COL = "DE_load_actual_entsoe_transparency"
LOCAL_TZ = "Europe/Berlin"

HORIZON_H = 168
FORECAST_START = pd.Timestamp("2020-01-01 00:00", tz=LOCAL_TZ)
BACKTEST_YEAR = 2020  # backtest origins are the 52 weeks *before* this year

# Approximate 2019 resident population per federal state, in millions. Used only to
# weight the load impact of holidays that are regional rather than national (Epiphany
# on 6 January, for example, is a holiday in BW/BY/ST only, ~32% of the population).
STATE_POP_M: dict[str, float] = {
    "BW": 11.10,
    "BY": 13.12,
    "BE": 3.67,
    "BB": 2.52,
    "HB": 0.68,
    "HH": 1.85,
    "HE": 6.27,
    "MV": 1.61,
    "NI": 7.99,
    "NW": 17.93,
    "RP": 4.09,
    "SL": 0.99,
    "SN": 4.07,
    "ST": 2.19,
    "SH": 2.90,
    "TH": 2.13,
}

CALENDAR_FEATURES = [
    "hour",
    "dayofweek",
    "dayofyear",
    "month",
    "daytype",
    "is_holiday",
    "holiday_pop_share",
    "yearend_offset",
]
LAG_FEATURES = [
    "lag_168",
    "lag_336",
    "lag_504",
    "lag_364d",
    "lag_365d",
    "roll_mean_prev_week",
    "level_28d",
]
CATEGORICAL = ["daytype", "yearend_offset"]

FEATURE_SETS: dict[str, list[str]] = {
    "calendar": CALENDAR_FEATURES,
    "calendar+lags": CALENDAR_FEATURES + LAG_FEATURES,
}

LGBM_PARAMS: dict[str, Any] = dict(
    objective="l2",
    n_estimators=900,
    learning_rate=0.045,
    num_leaves=64,
    min_child_samples=25,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=0,
    n_jobs=1,  # 10x faster than -1 here: threading overhead dominates on 35k rows
    verbose=-1,
    deterministic=True,
    force_row_wise=True,
)

# dataviz reference palette (validated: all-pairs CVD dE 9.2, normal-vision dE 24.0)
C_ACTUAL = "#2a78d6"  # slot 1 blue
C_FORECAST = "#eb6834"  # slot 2 orange
C_BASELINE = "#1baf7a"  # slot 3 aqua
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"


# --------------------------------------------------------------------------- data


def load_load_series(csv_path: Path = CSV_PATH) -> pd.Series:
    """Return the hourly load series (MW) indexed by Europe/Berlin local time."""
    frame = pd.read_csv(csv_path, parse_dates=["utc_timestamp"], index_col="utc_timestamp")
    series = frame[VALUE_COL].astype("float64").sort_index()
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    series = series.tz_convert(LOCAL_TZ)
    series.index.name = "timestamp"
    series.name = "load_mw"

    expected = pd.date_range(series.index[0], series.index[-1], freq="h")
    if not series.index.equals(expected):
        raise ValueError("load series is not a gap-free, on-the-hour, duplicate-free index")
    if series.isna().any():
        raise ValueError("load series contains missing values")
    return series


# ----------------------------------------------------------------------- features


def _holiday_tables(years: range) -> tuple[set[dt.date], dict[dt.date, float]]:
    """National holiday dates, and per-date share of the population on holiday."""
    national = set(holidays.Germany(years=list(years)).keys())

    total_pop = sum(STATE_POP_M.values())
    share: dict[dt.date, float] = {}
    for state, pop in STATE_POP_M.items():
        for day in holidays.Germany(subdiv=state, years=list(years)):
            share[day] = share.get(day, 0.0) + pop / total_pop
    return national, share


def _yearend_offset(day: dt.date) -> int:
    """Day offset from the adjacent 1 January, as a small categorical code.

    Codes 0..26 cover 20 December (offset -12) through 15 January (offset +14, where
    1 January is offset 0); 27 means
    "an ordinary day, not in the Christmas / New Year period". This is the feature
    that lets the model learn the shape of the industrial shutdown week, which a
    plain day-of-week feature cannot express.
    """
    jan1_this = dt.date(day.year, 1, 1)
    jan1_next = dt.date(day.year + 1, 1, 1)
    forward = (day - jan1_this).days  # >= 0
    backward = (jan1_next - day).days  # > 0
    if forward <= 14:
        return forward + 12
    if backward <= 12:
        return 12 - backward
    return 27


def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar predictors: all knowable arbitrarily far ahead of time."""
    years = range(index[0].year - 1, index[-1].year + 2)
    national, pop_share = _holiday_tables(years)

    dates = pd.Series(index.date, index=index)
    is_holiday = dates.map(lambda d: d in national).astype("int8")

    feats = pd.DataFrame(index=index)
    feats["hour"] = index.hour.astype("int16")
    feats["dayofweek"] = index.dayofweek.astype("int16")
    feats["dayofyear"] = index.dayofyear.astype("int16")
    feats["month"] = index.month.astype("int16")
    feats["is_holiday"] = is_holiday
    feats["holiday_pop_share"] = dates.map(lambda d: pop_share.get(d, 0.0)).astype("float64")
    feats["yearend_offset"] = dates.map(_yearend_offset).astype("int16")

    daytype = np.where(
        is_holiday.to_numpy() == 1,
        3,
        np.where(index.dayofweek == 6, 2, np.where(index.dayofweek == 5, 1, 0)),
    )
    feats["daytype"] = daytype.astype("int16")
    return feats


def build_lag_features(series: pd.Series) -> pd.DataFrame:
    """Lagged load. Every lag is >= 168 h, so all are observable at the forecast origin.

    For a forecast issued for hours [origin, origin+167], the 168-hour lag of the last
    forecast hour is origin-1 -- the final observed value. Nothing here peeks ahead.
    """
    feats = pd.DataFrame(index=series.index)
    feats["lag_168"] = series.shift(168)
    feats["lag_336"] = series.shift(336)
    feats["lag_504"] = series.shift(504)
    feats["lag_364d"] = series.shift(364 * 24)  # weekday-aligned, ~1 year back
    feats["lag_365d"] = series.shift(365 * 24)  # date-aligned, ~1 year back
    feats["roll_mean_prev_week"] = series.shift(168).rolling(168).mean()
    feats["level_28d"] = series.shift(168).rolling(672).mean()
    return feats


def build_design_matrix(series: pd.Series) -> pd.DataFrame:
    cal = build_calendar_features(series.index)
    lags = build_lag_features(series)
    return pd.concat([cal, lags], axis=1)


# ------------------------------------------------------------------------- models


def fit_forecast_lgbm(
    series: pd.Series,
    design: pd.DataFrame,
    origin: pd.Timestamp,
    horizon: int,
    features: list[str],
) -> pd.Series:
    """Train on everything strictly before `origin`, predict the next `horizon` hours."""
    target_index = pd.date_range(origin, periods=horizon, freq="h", tz=LOCAL_TZ)

    train_x = design.loc[design.index < origin, features]
    train_y = series.loc[series.index < origin]
    usable = train_x.notna().all(axis=1)
    train_x, train_y = train_x[usable], train_y[usable]

    cats = [c for c in CATEGORICAL if c in features]
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(train_x, train_y, categorical_feature=cats)

    pred = model.predict(design.loc[target_index, features])
    return pd.Series(np.asarray(pred, dtype="float64"), index=target_index, name="forecast")


def naive_last_week(series: pd.Series, origin: pd.Timestamp, horizon: int) -> pd.Series:
    idx = pd.date_range(origin, periods=horizon, freq="h", tz=LOCAL_TZ)
    return pd.Series(series.shift(168).loc[idx].to_numpy(), index=idx, name="naive_last_week")


def naive_last_year(series: pd.Series, origin: pd.Timestamp, horizon: int) -> pd.Series:
    idx = pd.date_range(origin, periods=horizon, freq="h", tz=LOCAL_TZ)
    return pd.Series(series.shift(365 * 24).loc[idx].to_numpy(), index=idx, name="naive_last_year")


# ------------------------------------------------------------------------ scoring


@dataclass(frozen=True)
class Scores:
    mae: float
    rmse: float
    mape: float
    bias: float
    max_abs_err: float

    def as_row(self) -> dict[str, float]:
        return {
            "MAE (MW)": self.mae,
            "RMSE (MW)": self.rmse,
            "MAPE (%)": self.mape,
            "Bias (MW)": self.bias,
            "Max |err| (MW)": self.max_abs_err,
        }


def score(actual: pd.Series, pred: pd.Series) -> Scores:
    if not actual.index.equals(pred.index):
        raise ValueError("actual and predicted series must share an identical index")
    err = pred.to_numpy() - actual.to_numpy()
    return Scores(
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err**2))),
        mape=float(np.mean(np.abs(err / actual.to_numpy())) * 100.0),
        bias=float(np.mean(err)),
        max_abs_err=float(np.max(np.abs(err))),
    )


def rolling_backtest(
    series: pd.Series, design: pd.DataFrame, year: int, horizon: int = HORIZON_H
) -> pd.DataFrame:
    """Refit and forecast at 52 weekly origins in the year before `year`.

    Model selection happens here and only here; the target week is never involved.
    """
    first = pd.Timestamp(f"{year - 1}-01-01 00:00", tz=LOCAL_TZ)
    origins = [first + pd.Timedelta(weeks=w) for w in range(52)]

    rows: list[dict[str, object]] = []
    for i, origin in enumerate(origins, start=1):
        actual = series.loc[origin : origin + pd.Timedelta(hours=horizon - 1)]
        if len(actual) != horizon:
            continue
        preds: dict[str, pd.Series] = {
            name: fit_forecast_lgbm(series, design, origin, horizon, feats)
            for name, feats in FEATURE_SETS.items()
        }
        preds["naive_last_week"] = naive_last_week(series, origin, horizon)
        for name, pred in preds.items():
            if pred.isna().any():
                raise ValueError(f"model {name!r} produced NaN at origin {origin}")
            rows.append({"origin": origin, "model": name, **score(actual, pred).as_row()})
        print(f"  backtest origin {i:2d}/52  {origin:%Y-%m-%d}", flush=True)

    frame = pd.DataFrame(rows)
    coverage = frame.groupby("model").size()
    if coverage.nunique() != 1:
        raise ValueError(f"models cover unequal origin counts, means not comparable:\n{coverage}")
    return frame


# ------------------------------------------------------------------------ plotting


def _style_axis(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_forecast(
    actual: pd.Series,
    forecast: pd.Series,
    baseline: pd.Series,
    scores: Scores,
    out_path: Path,
) -> None:
    """Two panels on one time axis: the forecast against actuals, and its error."""
    fig, (ax, ax_err) = plt.subplots(
        2,
        1,
        figsize=(13.0, 7.6),
        height_ratios=[3, 1],
        sharex=True,
        dpi=160,
        gridspec_kw={"hspace": 0.16},
    )
    fig.patch.set_facecolor(SURFACE)

    gw = lambda s: s / 1000.0  # noqa: E731 - local unit shorthand

    for a in (ax, ax_err):
        _style_axis(a)
        for day in pd.date_range(actual.index[0], actual.index[-1], freq="D"):
            a.axvline(day, color=GRID, linewidth=0.8, zorder=0)

    ax.plot(
        baseline.index,
        gw(baseline),
        color=C_BASELINE,
        linewidth=1.3,
        linestyle=(0, (5, 3)),
        label="Naive: same hour last week",
        zorder=2,
    )
    ax.plot(actual.index, gw(actual), color=C_ACTUAL, linewidth=2.0, label="Actual", zorder=4)
    ax.plot(
        forecast.index,
        gw(forecast),
        color=C_FORECAST,
        linewidth=2.0,
        label="Forecast (LightGBM)",
        zorder=3,
    )

    ax.set_ylabel("Load (GW)", color=INK_MUTED, fontsize=10)
    ax.set_title(
        "German electricity load, 1-7 January 2020: 168-hour forecast vs. actual",
        color=INK,
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=26,
    )
    ax.text(
        0.0,
        1.035,
        f"Trained on 2015-2019. MAPE {scores.mape:.2f}%  ·  MAE {scores.mae:,.0f} MW"
        f"  ·  RMSE {scores.rmse:,.0f} MW",
        transform=ax.transAxes,
        color=INK_MUTED,
        fontsize=10.5,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=9.5, ncol=3, labelcolor=INK_MUTED)

    # Direct labels at the right edge so identity is never colour-alone. The two
    # series can finish within a few hundred MW of each other, so nudge them apart
    # rather than letting the text overlap.
    ends = sorted(
        [
            ("Actual", C_ACTUAL, actual.iloc[-1] / 1000.0),
            ("Forecast", C_FORECAST, forecast.iloc[-1] / 1000.0),
        ],
        key=lambda e: e[2],
    )
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    gap = ends[1][2] - ends[0][2]
    if gap < 0.05 * span:  # would collide: spread symmetrically about their midpoint
        mid = (ends[0][2] + ends[1][2]) / 2.0
        label_y = [mid - 0.03 * span, mid + 0.03 * span]
    else:
        label_y = [ends[0][2], ends[1][2]]

    last_x = actual.index[-1]
    for (label, colour, y_data), y_text in zip(ends, label_y):
        ax.plot(
            [last_x], [y_data], marker="o", markersize=4.5, color=colour, clip_on=False, zorder=6
        )
        ax.annotate(
            label,
            xy=(last_x, y_text),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            color=colour,
            fontsize=9.5,
            fontweight="bold",
            annotation_clip=False,
            zorder=6,
        )

    err_gw = (forecast - actual) / 1000.0
    ax_err.axhline(0, color=INK_MUTED, linewidth=1.0, zorder=3)
    ax_err.fill_between(err_gw.index, 0, err_gw, color=C_FORECAST, alpha=0.22, zorder=2)
    ax_err.plot(err_gw.index, err_gw, color=C_FORECAST, linewidth=1.6, zorder=4)
    ax_err.set_ylabel("Error (GW)", color=INK_MUTED, fontsize=10)
    ax_err.text(
        0.0,
        1.06,
        "Forecast minus actual  ·  above zero = over-forecast",
        transform=ax_err.transAxes,
        color=INK_MUTED,
        fontsize=9.5,
    )

    ax_err.xaxis.set_major_locator(mdates.DayLocator())
    ax_err.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%-d %b"))
    ax_err.set_xlim(actual.index[0], actual.index[-1])

    # Mark the holidays that drive most of the week's difficulty: New Year's Day is
    # national, Epiphany on 6 January covers only BW/BY/ST (~19% of the population).
    national, pop_share = _holiday_tables(range(2019, 2022))
    top = ax.get_ylim()[1]
    for day in pd.date_range(actual.index[0], actual.index[-1], freq="D"):
        share = pop_share.get(day.date(), 0.0)
        if share == 0.0:
            continue
        is_national = day.date() in national
        ax.axvspan(
            day,
            day + pd.Timedelta(days=1),
            color=GRID,
            alpha=0.55 if is_national else 0.3,
            zorder=0,
        )
        note = "public holiday" if is_national else f"regional holiday ({share:.0%} of Germany)"
        ax.text(
            day + pd.Timedelta(hours=12),
            top,
            note,
            ha="center",
            va="top",
            fontsize=8.5,
            color=INK_MUTED,
        )

    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------- run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-backtest", action="store_true", help="skip rolling-origin model selection"
    )
    args = parser.parse_args()

    series = load_load_series()
    design = build_design_matrix(series)
    print(
        f"Loaded {len(series):,} hourly observations, "
        f"{series.index[0]:%Y-%m-%d %H:%M %Z} to {series.index[-1]:%Y-%m-%d %H:%M %Z}"
    )

    chosen = "calendar+lags"
    bt: pd.DataFrame | None = None
    if not args.no_backtest:
        print(
            f"\nRolling-origin backtest, 52 weekly origins across {BACKTEST_YEAR - 1} "
            f"(each refits on data before its origin):"
        )
        bt = rolling_backtest(series, design, BACKTEST_YEAR)
        summary = bt.groupby("model")[["MAE (MW)", "RMSE (MW)", "MAPE (%)"]].mean()
        summary = summary.sort_values("MAPE (%)")
        print("\nBacktest means over 52 weeks:")
        print(summary.round(2).to_string())
        bt.to_csv(HERE / "backtest_2019.csv", index=False)

        candidates = summary.loc[summary.index.isin(FEATURE_SETS)]
        chosen = str(candidates.index[0])
        print(
            f"\nSelected feature set: '{chosen}' "
            f"(lowest backtest MAPE; the 2020 test week played no part in this)."
        )

    origin = FORECAST_START
    actual = series.loc[origin : origin + pd.Timedelta(hours=HORIZON_H - 1)]
    if len(actual) != HORIZON_H:
        raise ValueError(f"expected {HORIZON_H} actual hours, found {len(actual)}")

    print(
        f"\nTraining on {(series.index < origin).sum():,} hours "
        f"(through {series.loc[series.index < origin].index[-1]:%Y-%m-%d %H:%M}); "
        f"forecasting {HORIZON_H} hours from {origin:%Y-%m-%d %H:%M %Z}."
    )

    forecast = fit_forecast_lgbm(series, design, origin, HORIZON_H, FEATURE_SETS[chosen])
    baselines = {
        "Naive: same hour last week": naive_last_week(series, origin, HORIZON_H),
        "Naive: same hour last year": naive_last_year(series, origin, HORIZON_H),
    }

    print("\n" + "=" * 72)
    print("ACCURACY, 168 hours of 2020-01-01 .. 2020-01-07 (Europe/Berlin)")
    print("=" * 72)
    table = pd.DataFrame(
        {f"LightGBM ({chosen})": score(actual, forecast).as_row()}
        | {name: score(actual, pred).as_row() for name, pred in baselines.items()}
    ).T
    print(table.round(2).to_string())

    daily = pd.DataFrame({"actual": actual, "forecast": forecast})
    daily["abs_pct_err"] = (daily["forecast"] - daily["actual"]).abs() / daily["actual"] * 100
    by_day = daily.groupby(daily.index.date).agg(
        actual_mean_MW=("actual", "mean"),
        forecast_mean_MW=("forecast", "mean"),
        MAPE_pct=("abs_pct_err", "mean"),
    )
    by_day.insert(0, "weekday", [pd.Timestamp(d).day_name()[:3] for d in by_day.index])
    if bt is not None:
        ref = bt[bt["model"] == chosen].set_index("origin")["MAPE (%)"]
        test_mape = score(actual, forecast).mape
        pct = float((ref < test_mape).mean() * 100)
        print(
            f"\nContext from the {BACKTEST_YEAR - 1} backtest of the same model "
            f"({len(ref)} weeks): mean {ref.mean():.2f}%, median {ref.median():.2f}%, "
            f"worst {ref.max():.2f}%."
        )
        jan_analogue = float(ref.iloc[0])
        print(
            f"  Equivalent first-week-of-January backtest: {jan_analogue:.2f}%. "
            f"This week: {test_mape:.2f}% ({test_mape - jan_analogue:+.2f} pp)."
        )
        print(
            f"  That is the {pct:.0f}th percentile of that year's 52 weeks "
            f"(100th would be the worst week of the year)."
        )
        print(
            "  Caveat: the backtest mean was used to pick the feature set, so it is a "
            "selection statistic, not an unbiased out-of-sample estimate."
        )

    print("\nPer-day breakdown:")
    print(by_day.round(2).to_string())

    out_csv = HERE / "forecast_2020_week1.csv"
    daily.drop(columns="abs_pct_err").assign(error_MW=lambda d: d["forecast"] - d["actual"]).to_csv(
        out_csv
    )

    out_png = HERE / "forecast_2020_week1.png"
    plot_forecast(
        actual, forecast, baselines["Naive: same hour last week"], score(actual, forecast), out_png
    )
    print(f"\nWrote {out_png.name} and {out_csv.name}")


if __name__ == "__main__":
    main()
