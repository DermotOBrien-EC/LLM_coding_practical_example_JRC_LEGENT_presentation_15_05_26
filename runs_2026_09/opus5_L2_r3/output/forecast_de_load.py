"""Forecast German hourly electricity load for a 168-hour window.

The forecast is *ex ante*: at the forecast origin the model sees only load
values strictly before the window, so every lag feature is at least
``HORIZON`` hours old. Nothing inside the window is fed back in.

Run ``python forecast_de_load.py --help`` for options.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Iterable, Sequence

import holidays
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LOCAL_TZ: Final[str] = "Europe/Berlin"
HORIZON: Final[int] = 168
TARGET_COL: Final[str] = "DE_load_actual_entsoe_transparency"

# Load is not proportional to population (industry weighs more), but population
# share is a serviceable proxy for how much a state-only holiday dents national
# load. Figures are 2019 Destatis state populations in millions.
BUNDESLAND_POP: Final[dict[str, float]] = {
    "NW": 17.93,
    "BY": 13.08,
    "BW": 11.07,
    "NI": 7.99,
    "HE": 6.27,
    "RP": 4.09,
    "SN": 4.07,
    "BE": 3.67,
    "SH": 2.90,
    "BB": 2.52,
    "ST": 2.19,
    "TH": 2.13,
    "HH": 1.85,
    "MV": 1.61,
    "SL": 0.99,
    "HB": 0.68,
}

# Every lag is >= HORIZON, which is what makes a 168-hour-ahead forecast honest.
LAGS: Final[tuple[int, ...]] = (168, 169, 170, 172, 176, 192, 336, 504, 672, 8736, 8760)

DISPLAY_NAMES: Final[dict[str, str]] = {
    "seasonal_naive": "Seasonal naive",
    "harmonic_regression": "Harmonic regression",
}


@dataclass(frozen=True)
class Theme:
    """Chart colours for one mode. Both sets pass the categorical palette checks
    (lightness band, chroma floor, all-pairs CVD separation, contrast) against
    their own surface; the dark set is the same three hues re-stepped, not an
    inversion. Slot order is fixed and never cycled."""

    surface: str
    text: str
    muted: str
    grid: str
    band: str
    series: tuple[str, str, str]


THEMES: Final[dict[str, Theme]] = {
    "light": Theme(
        surface="#fcfcfb",
        text="#0b0b0b",
        muted="#52514e",
        grid="#e3e2df",
        band="#eeede9",
        series=("#2a78d6", "#eb6834", "#1baf7a"),
    ),
    "dark": Theme(
        surface="#1a1a19",
        text="#ffffff",
        muted="#c3c2b7",
        grid="#34332f",
        band="#262523",
        series=("#3987e5", "#d95926", "#199e70"),
    ),
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_series(csv_path: Path) -> pd.Series:
    """Read the OPSD CSV into a gap-free, UTC-indexed hourly load series."""
    frame = pd.read_csv(csv_path, parse_dates=["utc_timestamp"])
    series = (
        frame.set_index("utc_timestamp")[TARGET_COL].sort_index().rename("load_mw").astype(float)
    )
    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    else:
        series.index = series.index.tz_convert("UTC")

    if series.index.has_duplicates:
        raise ValueError("duplicate timestamps in input")
    expected = pd.date_range(series.index[0], series.index[-1], freq="h", tz="UTC")
    if not series.index.equals(expected):
        raise ValueError(
            f"index is not a gap-free hourly grid: {len(series)} rows, {len(expected)} expected"
        )
    if series.isna().any():
        raise ValueError(f"{int(series.isna().sum())} missing load values")
    return series


# --------------------------------------------------------------------------- #
# Calendar features (all derived from German local time)
# --------------------------------------------------------------------------- #
class HolidayCalendar:
    """German public holidays, national and state-weighted."""

    def __init__(self) -> None:
        self._national: holidays.HolidayBase = holidays.country_holidays("DE")
        self._states: dict[str, holidays.HolidayBase] = {
            code: holidays.country_holidays("DE", subdiv=code) for code in BUNDESLAND_POP
        }
        self._total_pop: float = sum(BUNDESLAND_POP.values())
        self._share_cache: dict[dt.date, float] = {}

    def is_national(self, day: dt.date) -> bool:
        return day in self._national

    def population_share(self, day: dt.date) -> float:
        """Share of the German population for which ``day`` is a public holiday."""
        cached = self._share_cache.get(day)
        if cached is not None:
            return cached
        covered = sum(pop for code, pop in BUNDESLAND_POP.items() if day in self._states[code])
        share = covered / self._total_pop
        self._share_cache[day] = share
        return share


def _fourier(frame: dict[str, np.ndarray], name: str, phase: np.ndarray, orders: int) -> None:
    for k in range(1, orders + 1):
        frame[f"{name}_sin{k}"] = np.sin(2.0 * np.pi * k * phase)
        frame[f"{name}_cos{k}"] = np.cos(2.0 * np.pi * k * phase)


def calendar_features(index: pd.DatetimeIndex, cal: HolidayCalendar) -> pd.DataFrame:
    """Calendar/holiday regressors. Known arbitrarily far ahead, so leak-free."""
    local = index.tz_convert(LOCAL_TZ)
    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    doy = local.dayofyear.to_numpy()
    days = np.array([d.date() for d in local])

    national = np.array([cal.is_national(d) for d in days], dtype=float)
    share = np.array([cal.population_share(d) for d in days], dtype=float)
    share_prev = np.array([cal.population_share(d - dt.timedelta(days=1)) for d in days])
    share_next = np.array([cal.population_share(d + dt.timedelta(days=1)) for d in days])

    month = local.month.to_numpy()
    day_of_month = local.day.to_numpy()
    # The Christmas/New Year industrial lull: many plants simply shut down, so
    # load in this stretch is unlike any ordinary winter week.
    lull = ((month == 12) & (day_of_month >= 24)) | ((month == 1) & (day_of_month <= 6))

    cols: dict[str, np.ndarray] = {
        "hour": hour.astype(float),
        "dayofweek": dow.astype(float),
        "dayofyear": doy.astype(float),
        "month": month.astype(float),
        "is_weekend": (dow >= 5).astype(float),
        "is_saturday": (dow == 5).astype(float),
        "is_sunday": (dow == 6).astype(float),
        "is_national_holiday": national,
        "holiday_pop_share": share,
        "holiday_pop_share_prev_day": share_prev,
        "holiday_pop_share_next_day": share_next,
        "is_xmas_lull": lull.astype(float),
        "days_into_lull": np.where(
            lull, np.where(month == 12, day_of_month - 24, day_of_month + 7), -1.0
        ),
        # A workday wedged between a holiday and a weekend: many take it off.
        "is_bridge_day": (
            (dow < 5) & (national == 0.0) & ((share_prev > 0.5) | (share_next > 0.5))
        ).astype(float),
        "year_frac": (local.year.to_numpy() - 2015).astype(float),
    }
    _fourier(cols, "daily", hour / 24.0, 4)
    _fourier(cols, "weekly", (dow * 24 + hour) / 168.0, 3)
    _fourier(cols, "yearly", doy / 365.25, 3)
    return pd.DataFrame(cols, index=index)


# --------------------------------------------------------------------------- #
# Lag features
# --------------------------------------------------------------------------- #
def lag_features(
    history: pd.Series,
    target_index: pd.DatetimeIndex,
    cal: HolidayCalendar,
    lags: Sequence[int] = LAGS,
) -> pd.DataFrame:
    """Lagged-load regressors for ``target_index``, using only ``history``.

    ``history`` must already be truncated to the data available at the forecast
    origin; this function extends it with NaN over the target window, so a
    future value cannot enter a feature even by accident.
    """
    too_recent = [lag for lag in lags if lag < HORIZON]
    if too_recent:
        raise ValueError(f"lags shorter than the {HORIZON}h horizon would leak: {too_recent}")

    span = pd.date_range(
        min(history.index[0], target_index[0]),
        max(history.index[-1], target_index[-1]),
        freq="h",
        tz="UTC",
    )
    base = history.reindex(span)  # positions at/after the origin are NaN

    cols: dict[str, pd.Series] = {
        f"lag_{lag}h": base.shift(lag).reindex(target_index) for lag in lags
    }

    shifted = base.shift(HORIZON)
    for window, label in ((24, "1d"), (168, "1w"), (672, "4w")):
        roll = shifted.rolling(window, min_periods=max(2, window // 2))
        cols[f"roll_mean_{label}"] = roll.mean().reindex(target_index)
    cols["roll_std_1w"] = shifted.rolling(168, min_periods=84).std().reindex(target_index)

    # A level anchor that ignores holidays and the Christmas lull, so the model
    # still knows the *ordinary* recent level when the origin sits in the lull.
    span_cal = calendar_features(span, cal)
    ordinary = base.where(
        (span_cal["is_national_holiday"].to_numpy() == 0.0)
        & (span_cal["is_xmas_lull"].to_numpy() == 0.0)
        & (span_cal["is_weekend"].to_numpy() == 0.0)
    )
    cols["ordinary_level_8w"] = (
        ordinary.shift(HORIZON).rolling(1344, min_periods=200).mean().reindex(target_index)
    )
    frame = pd.DataFrame(cols, index=target_index)
    frame["lag_168h_vs_ordinary"] = frame["lag_168h"] - frame["ordinary_level_8w"]
    return frame


def design_matrix(
    history: pd.Series, target_index: pd.DatetimeIndex, cal: HolidayCalendar
) -> pd.DataFrame:
    return pd.concat(
        [calendar_features(target_index, cal), lag_features(history, target_index, cal)],
        axis=1,
    )


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
ModelFn = Callable[[pd.Series, pd.DatetimeIndex, HolidayCalendar], pd.Series]


def seasonal_naive(
    history: pd.Series, target_index: pd.DatetimeIndex, cal: HolidayCalendar
) -> pd.Series:
    """Last week's value at the same hour. The standard load-forecasting yardstick."""
    del cal
    values = history.reindex(target_index - pd.Timedelta(hours=HORIZON)).to_numpy()
    return pd.Series(values, index=target_index, name="seasonal_naive")


def _harmonic_columns(cal_frame: pd.DataFrame) -> pd.DataFrame:
    """Daily shape interacted with day type, plus weekly/yearly harmonics."""
    daily = [c for c in cal_frame.columns if c.startswith("daily_")]
    weekly = [c for c in cal_frame.columns if c.startswith("weekly_")]
    yearly = [c for c in cal_frame.columns if c.startswith("yearly_")]

    holiday_like = np.maximum(
        cal_frame["is_national_holiday"].to_numpy(), cal_frame["is_sunday"].to_numpy()
    )
    saturday = cal_frame["is_saturday"].to_numpy() * (1.0 - holiday_like)
    workday = 1.0 - np.maximum(holiday_like, saturday)

    out: dict[str, np.ndarray] = {}
    for label, mask in (("work", workday), ("sat", saturday), ("off", holiday_like)):
        out[f"daytype_{label}"] = mask
        for col in daily:
            out[f"{col}_x_{label}"] = cal_frame[col].to_numpy() * mask
    for col in weekly + yearly:
        out[col] = cal_frame[col].to_numpy()
    out["holiday_pop_share"] = cal_frame["holiday_pop_share"].to_numpy()
    out["is_xmas_lull"] = cal_frame["is_xmas_lull"].to_numpy()
    out["days_into_lull"] = cal_frame["days_into_lull"].to_numpy()
    out["is_bridge_day"] = cal_frame["is_bridge_day"].to_numpy()
    out["trend"] = cal_frame["year_frac"].to_numpy()
    return pd.DataFrame(out, index=cal_frame.index)


def harmonic_regression(
    history: pd.Series, target_index: pd.DatetimeIndex, cal: HolidayCalendar
) -> pd.Series:
    """Least-squares harmonic model on log load: transparent statistical benchmark."""
    train = history.iloc[-3 * 8760 :] if len(history) > 3 * 8760 else history
    x_train = _harmonic_columns(calendar_features(train.index, cal)).to_numpy()
    x_test = _harmonic_columns(calendar_features(target_index, cal)).to_numpy()
    y_train = np.log(train.to_numpy())

    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])
    coef, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    return pd.Series(np.exp(x_test @ coef), index=target_index, name="harmonic_regression")


@dataclass(frozen=True)
class LgbmConfig:
    n_estimators: int = 700
    num_leaves: int = 63
    learning_rate: float = 0.05
    min_child_samples: int = 40
    # LightGBM's default (-1) oversubscribes this machine's efficiency cores and
    # runs ~2.5x slower than a modest thread count on a table this size.
    n_jobs: int = 4

    @property
    def label(self) -> str:
        return f"lgbm(n={self.n_estimators},leaves={self.num_leaves})"


def make_lgbm(config: LgbmConfig, seed: int = 0) -> ModelFn:
    """Gradient-boosted trees, fitted directly at the 168-hour horizon."""

    def fit_predict(
        history: pd.Series, target_index: pd.DatetimeIndex, cal: HolidayCalendar
    ) -> pd.Series:
        train_x = design_matrix(history, history.index, cal)
        train_y = np.log(history.to_numpy())
        usable = train_x.notna().all(axis=1).to_numpy()
        model = lgb.LGBMRegressor(
            n_estimators=config.n_estimators,
            num_leaves=config.num_leaves,
            learning_rate=config.learning_rate,
            min_child_samples=config.min_child_samples,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=config.n_jobs,
            force_col_wise=True,
            verbose=-1,
        )
        model.fit(train_x.loc[usable], train_y[usable])
        test_x = design_matrix(history, target_index, cal)
        return pd.Series(np.exp(model.predict(test_x)), index=target_index, name=config.label)

    return fit_predict


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    err = predicted.to_numpy() - actual.to_numpy()
    act = actual.to_numpy()
    return {
        "MAPE_pct": float(np.mean(np.abs(err / act)) * 100.0),
        "MAE_MW": float(np.mean(np.abs(err))),
        "RMSE_MW": float(np.sqrt(np.mean(err**2))),
        "bias_MW": float(np.mean(err)),
        "max_abs_err_MW": float(np.max(np.abs(err))),
        "peak_err_pct": float((predicted.max() - actual.max()) / actual.max() * 100.0),
    }


def window_index(start_local: pd.Timestamp, hours: int = HORIZON) -> pd.DatetimeIndex:
    start_utc = start_local.tz_localize(LOCAL_TZ).tz_convert("UTC")
    return pd.date_range(start_utc, periods=hours, freq="h", tz="UTC")


def fold_regime(origin: pd.Timestamp, cal: HolidayCalendar) -> str:
    """Label a window by whether it contains a national holiday.

    A New Year week and an ordinary October week are different forecasting
    problems, so a model is picked on folds that match the target's regime.
    """
    days = {t.date() for t in window_index(origin).tz_convert(LOCAL_TZ)}
    return "holiday" if any(cal.is_national(d) for d in days) else "ordinary"


def backtest(
    series: pd.Series,
    origins: Iterable[pd.Timestamp],
    models: dict[str, ModelFn],
    cal: HolidayCalendar,
) -> pd.DataFrame:
    """Rolling-origin evaluation. Each fold trains only on data before its origin."""
    rows: list[dict[str, object]] = []
    for origin in origins:
        print(f"  fold {origin:%Y-%m-%d} ...", flush=True)
        target = window_index(origin)
        if target[-1] > series.index[-1]:
            continue
        history = series.loc[series.index < target[0]]
        actual = series.reindex(target)
        for name, fn in models.items():
            pred = fn(history, target, cal)
            print(f"      {name:32s} MAPE {metrics(actual, pred)['MAPE_pct']:5.2f}%", flush=True)
            rows.append(
                {"origin": origin.date().isoformat(), "model": name, **metrics(actual, pred)}
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_forecast(
    actual: pd.Series,
    forecasts: dict[str, pd.Series],
    cal: HolidayCalendar,
    out_path: Path,
    theme: str,
    headline: str,
) -> None:
    style = THEMES[theme]
    surface, text, muted = style.surface, style.text, style.muted
    grid_c, band_c = style.grid, style.band
    series_colors = style.series

    local_index = actual.index.tz_convert(LOCAL_TZ)
    x = local_index.to_pydatetime()

    fig, (ax, ax_res) = plt.subplots(
        2,
        1,
        figsize=(14.6, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.12},
    )
    fig.patch.set_facecolor(surface)

    named = list(forecasts.items())
    if len(named) > len(series_colors) - 1:
        raise ValueError(
            f"{len(named)} forecast series exceeds the {len(series_colors) - 1} "
            "validated hues; cycling hues would make two series share a colour"
        )
    colors = {"actual": series_colors[0]}
    for i, (name, _) in enumerate(named):
        colors[name] = series_colors[1 + i]

    # Shade days that are a public holiday somewhere in Germany.
    days = pd.DatetimeIndex(sorted({t.normalize() for t in local_index}))
    for day in days:
        share = cal.population_share(day.date())
        if share <= 0.0:
            continue
        ax.axvspan(
            day.to_pydatetime(),
            (day + pd.Timedelta(days=1)).to_pydatetime(),
            color=band_c,
            zorder=0,
            lw=0,
        )
        label = "public holiday" if share > 0.9 else f"holiday in {share:.0%} of DE"
        ax.text(
            (day + pd.Timedelta(hours=12)).to_pydatetime(),
            0.965,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=muted,
        )

    for a in (ax, ax_res):
        a.set_facecolor(surface)
        a.grid(True, color=grid_c, lw=0.8, zorder=1)
        a.set_axisbelow(True)
        for spine in ("top", "right"):
            a.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            a.spines[spine].set_color(grid_c)
        a.tick_params(colors=muted, labelsize=9)

    ax.plot(x, actual.to_numpy(), lw=2.0, color=colors["actual"], label="Actual", zorder=5)
    dashes = [(0, (6, 2)), (0, (1.5, 2))]
    for i, (name, pred) in enumerate(named):
        ax.plot(
            x,
            pred.to_numpy(),
            lw=2.0,
            color=colors[name],
            label=name,
            ls=dashes[i % len(dashes)],
            zorder=4,
        )

    # Direct labels at the right edge, so identity never rests on colour alone.
    for label, values in [("Actual", actual)] + named:
        ax.annotate(
            label,
            xy=(x[-1], float(values.to_numpy()[-1])),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=colors["actual" if label == "Actual" else label],
            fontweight="600",
        )

    ax.set_ylabel("Load (MW)", color=muted, fontsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title(
        "German electricity load, 1-7 January 2020",
        color=text,
        fontsize=15,
        fontweight="600",
        loc="left",
        pad=56,
    )
    ax.text(
        0.0,
        1.09,
        headline,
        transform=ax.transAxes,
        color=muted,
        fontsize=10.5,
        ha="left",
        va="bottom",
    )
    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        frameon=False,
        fontsize=9,
        ncols=len(named) + 1,
        handlelength=2.6,
        columnspacing=1.8,
    )
    for txt in legend.get_texts():
        txt.set_color(muted)

    primary_name, primary_pred = named[0]
    residual = primary_pred.to_numpy() - actual.to_numpy()
    ax_res.axhline(0.0, color=muted, lw=1.0, zorder=3)
    ax_res.fill_between(x, 0.0, residual, color=colors[primary_name], alpha=0.35, lw=0, zorder=2)
    ax_res.plot(x, residual, lw=1.6, color=colors[primary_name], zorder=4)
    ax_res.set_ylabel("Error (MW)", color=muted, fontsize=10)
    ax_res.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax_res.text(
        0.0,
        1.03,
        f"{primary_name} minus actual  ·  above zero = over-forecast",
        transform=ax_res.transAxes,
        color=muted,
        fontsize=9,
        ha="left",
        va="bottom",
    )

    ax_res.xaxis.set_major_locator(mdates.DayLocator())  # type: ignore[no-untyped-call]
    ax_res.xaxis.set_major_formatter(
        mdates.DateFormatter("%a\n%d %b")  # type: ignore[no-untyped-call]
    )
    ax_res.xaxis.set_minor_locator(
        mdates.HourLocator(byhour=(6, 12, 18))  # type: ignore[no-untyped-call]
    )
    ax_res.set_xlim(x[0], x[-1])
    ax_res.set_xlabel(f"Local time ({LOCAL_TZ})", color=muted, fontsize=10)

    fig.subplots_adjust(left=0.065, right=0.85, top=0.85, bottom=0.10)
    fig.savefig(out_path, dpi=160, facecolor=surface)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=here / "opsd_de_load.csv")
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="First local day of the forecast window (default: 2020-01-01).",
    )
    parser.add_argument("--outdir", type=Path, default=here / "output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=4, help="LightGBM threads.")
    parser.add_argument(
        "--reuse-validation",
        action="store_true",
        help="Reuse output/validation_folds.csv instead of refitting the folds.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip model selection and use the default LightGBM config.",
    )
    return parser.parse_args(argv)


def validation_origins(window_start: pd.Timestamp) -> list[pd.Timestamp]:
    """Folds strictly before the forecast window: New Year weeks plus ordinary ones."""
    candidates = [
        pd.Timestamp("2018-01-01"),
        pd.Timestamp("2019-01-01"),  # same seasonal trap
        pd.Timestamp("2018-03-05"),
        pd.Timestamp("2018-06-04"),
        pd.Timestamp("2018-10-01"),
        pd.Timestamp("2019-03-04"),
        pd.Timestamp("2019-06-03"),
        pd.Timestamp("2019-10-07"),
        pd.Timestamp("2019-11-04"),
    ]
    return [c for c in candidates if window_index(c)[-1] < window_start]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    cal = HolidayCalendar()

    series = load_series(args.csv)
    target = window_index(pd.Timestamp(args.start))
    if target[-1] > series.index[-1]:
        raise SystemExit(f"forecast window extends past the data ({series.index[-1]})")

    history = series.loc[series.index < target[0]]
    actual = series.reindex(target)

    print("=" * 78)
    print("GERMAN HOURLY LOAD FORECAST")
    print("=" * 78)
    print(
        f"data          : {series.index[0]:%Y-%m-%d} .. {series.index[-1]:%Y-%m-%d} UTC"
        f"  ({len(series):,} hourly observations, no gaps)"
    )
    print(
        f"training data : up to {history.index[-1].tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M %Z}"
        f"  ({len(history):,} hours)"
    )
    print(
        f"forecast      : {target[0].tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M} .. "
        f"{target[-1].tz_convert(LOCAL_TZ):%Y-%m-%d %H:%M} local ({len(target)} hours)"
    )
    print("mode          : ex ante, single 168h origin, no actuals from inside the window\n")

    # ---- model selection on folds that end before the forecast window ----- #
    configs = [
        LgbmConfig(n_estimators=400, num_leaves=31, n_jobs=args.n_jobs),
        LgbmConfig(n_estimators=800, num_leaves=63, n_jobs=args.n_jobs),
        LgbmConfig(n_estimators=1200, num_leaves=127, n_jobs=args.n_jobs),
    ]
    chosen = configs[1]
    selected_name = "LightGBM"
    selection_basis = "default config (backtest skipped)"
    if not args.skip_backtest:
        origins = validation_origins(target[0])
        candidates: dict[str, ModelFn] = {
            "seasonal_naive": seasonal_naive,
            "harmonic_regression": harmonic_regression,
        }
        for cfg in configs:
            candidates[cfg.label] = make_lgbm(cfg, args.seed)

        cached = args.outdir / "validation_folds.csv"
        if args.reuse_validation and cached.exists():
            print(f"Reusing validation folds from {cached}")
            results = pd.read_csv(cached)
            stale = set(candidates) ^ set(results["model"].unique())
            if stale:
                raise SystemExit(
                    f"{cached} was produced by a different candidate set "
                    f"(mismatched: {sorted(stale)}). Delete it and re-run without "
                    "--reuse-validation."
                )
        else:
            print(
                f"Validating on {len(origins)} held-out weeks before the window "
                f"({', '.join(o.strftime('%Y-%m-%d') for o in origins)}) ..."
            )
            results = backtest(series, origins, candidates, cal)
            results.to_csv(cached, index=False)

        results["regime"] = [fold_regime(pd.Timestamp(o), cal) for o in results["origin"]]
        target_regime = fold_regime(pd.Timestamp(args.start), cal)

        def rank(frame: pd.DataFrame) -> pd.DataFrame:
            return (
                frame.groupby("model")[["MAPE_pct", "MAE_MW", "RMSE_MW"]]
                .mean()
                .sort_values("MAPE_pct")
            )

        overall = rank(results)
        print("\nValidation, all folds (mean MAPE %):")
        print(overall.round(2).to_string())

        matched = results.loc[results["regime"] == target_regime]
        n_matched = matched["origin"].nunique()
        print(
            f"\nThe target window is a '{target_regime}' week, so the model is "
            f"picked on the {n_matched} matching fold(s) only:"
        )
        regime_rank = rank(matched)
        print(regime_rank.round(2).to_string())

        best = regime_rank.index[0]
        by_label = {cfg.label: cfg for cfg in configs}
        if best in by_label:
            chosen = by_label[best]
            print(f"\nSelected: {best}")
        else:
            selected_name = DISPLAY_NAMES[best]
            print(
                f"\nSelected: {best} — it beat every LightGBM config on the "
                f"{n_matched} regime-matched fold(s)."
            )
        selection_basis = f"{n_matched} '{target_regime}' fold(s) before the window"

    # ---- final fit and forecast ------------------------------------------- #
    print("\nFitting on all data before the window and forecasting 168 hours ...")
    forecasts: dict[str, pd.Series] = {
        "LightGBM": make_lgbm(chosen, args.seed)(history, target, cal),
        "Harmonic regression": harmonic_regression(history, target, cal),
        "Seasonal naive": seasonal_naive(history, target, cal),
    }

    scores = {name: metrics(actual, pred) for name, pred in forecasts.items()}
    table = pd.DataFrame(scores).T.sort_values("MAPE_pct")
    print("\n" + "=" * 78)
    print(
        f"ACCURACY on {target[0].tz_convert(LOCAL_TZ):%d %b} - "
        f"{target[-1].tz_convert(LOCAL_TZ):%d %b %Y} (168 hours, never seen in training)"
    )
    print("=" * 78)
    print(table.round(2).to_string())
    print(
        f"\nPrimary model: {selected_name} (chosen on {selection_basis}, "
        "not on the week below). All three are reported so the choice is auditable."
    )

    primary = selected_name
    per_day = (
        pd.DataFrame(
            {"actual": actual.to_numpy(), "forecast": forecasts[primary].to_numpy()},
            index=actual.index.tz_convert(LOCAL_TZ),
        )
        .assign(
            ape=lambda d: (d.forecast - d.actual).abs() / d.actual * 100.0,
            err=lambda d: d.forecast - d.actual,
        )
        .groupby(lambda t: t.date())
        .agg(MAPE_pct=("ape", "mean"), bias_MW=("err", "mean"), actual_peak_MW=("actual", "max"))
    )
    print(f"\n{primary}, day by day:")
    print(per_day.round(2).to_string())

    # ---- artefacts --------------------------------------------------------- #
    out_table = pd.DataFrame(
        {"actual_MW": actual.to_numpy()}
        | {f"{name}_MW": pred.to_numpy() for name, pred in forecasts.items()},
        index=actual.index.tz_convert(LOCAL_TZ).rename("local_time"),
    )
    out_table[f"{primary}_error_MW"] = out_table[f"{primary}_MW"] - out_table["actual_MW"]
    out_table.round(1).to_csv(args.outdir / "forecast_vs_actual.csv")
    (args.outdir / "metrics.json").write_text(
        json.dumps(
            {
                "window_start_local": str(target[0].tz_convert(LOCAL_TZ)),
                "primary_model": primary,
                "lgbm_config": chosen.label,
                "selection_basis": selection_basis,
                "test": scores,
                # json object keys must be strings; the index holds date objects.
                "per_day": {
                    str(day): row for day, row in per_day.round(3).to_dict(orient="index").items()
                },
            },
            indent=2,
            default=str,
        )
    )

    headline = (
        f"{primary}, 168 h ahead  ·  MAPE {scores[primary]['MAPE_pct']:.2f}%  ·  "
        f"MAE {scores[primary]['MAE_MW']:,.0f} MW  ·  "
        f"seasonal-naive baseline {scores['Seasonal naive']['MAPE_pct']:.2f}%"
    )
    baseline = "Seasonal naive" if primary != "Seasonal naive" else "Harmonic regression"
    plot_series = {primary: forecasts[primary], baseline: forecasts[baseline]}
    for theme in ("light", "dark"):
        suffix = "" if theme == "light" else "_dark"
        plot_forecast(
            actual,
            plot_series,
            cal,
            args.outdir / f"forecast{suffix}.png",
            theme,
            headline,
        )

    print(f"\nWritten to {args.outdir}/:")
    for name in sorted(p.name for p in args.outdir.iterdir()):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
