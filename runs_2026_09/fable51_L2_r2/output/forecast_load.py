"""Forecast German hourly electricity load for the first week of January 2020.

The model is a single LightGBM regressor that predicts each of the 168 target
hours directly from information available at the forecast origin
(2019-12-31 23:00 UTC): calendar features in Berlin local time, German public
holiday flags, and the load at the same hour one, two, three and fifty-two
weeks earlier plus recent weekly/monthly level. No lag shorter than 168 hours
is used, so the whole week can be forecast in one shot without feeding
predictions back in.

Run:  python forecast_load.py
Outputs: forecast_jan2020.csv, forecast_jan2020.png, metrics printed to stdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import holidays
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

DATA_PATH = Path(__file__).with_name("opsd_de_load.csv")
LOCAL_TZ = "Europe/Berlin"
HORIZON_HOURS = 168
TEST_START = pd.Timestamp("2020-01-01 00:00", tz="UTC")
BACKTEST_START = pd.Timestamp("2019-01-01 00:00", tz="UTC")

GERMAN_STATES = [
    "BB", "BE", "BW", "BY", "HB", "HE", "HH", "MV",
    "NI", "NW", "RP", "SH", "SL", "SN", "ST", "TH",
]  # fmt: skip

# Chart colours: dataviz reference palette (light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES_ACTUAL = "#2a78d6"
SERIES_FORECAST = "#eb6834"


@dataclass(frozen=True)
class Metrics:
    mae_mw: float
    rmse_mw: float
    mape_pct: float
    max_abs_err_mw: float

    @staticmethod
    def from_series(actual: pd.Series, predicted: pd.Series) -> Metrics:
        err = predicted - actual
        return Metrics(
            mae_mw=float(err.abs().mean()),
            rmse_mw=float(np.sqrt((err**2).mean())),
            mape_pct=float((err.abs() / actual).mean() * 100),
            max_abs_err_mw=float(err.abs().max()),
        )

    def row(self, label: str) -> str:
        return (
            f"{label:<34} MAE {self.mae_mw:8.0f} MW   RMSE {self.rmse_mw:8.0f} MW   "
            f"MAPE {self.mape_pct:5.2f} %   max |err| {self.max_abs_err_mw:6.0f} MW"
        )


def load_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["utc_timestamp"], index_col="utc_timestamp")
    series = df["DE_load_actual_entsoe_transparency"].astype(float).rename("load_mw")
    series.index = pd.DatetimeIndex(series.index, name="utc_timestamp")
    expected = pd.date_range(series.index[0], series.index[-1], freq="h", tz="UTC")
    if not series.index.equals(expected) or series.isna().any():
        raise ValueError("expected a gap-free hourly series without missing values")
    return series


def holiday_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """National holiday flags plus the share of the 16 states observing a holiday.

    Epiphany (6 January) is a holiday in only three states, so a national flag
    alone would miss the dip it causes. The share is an equal-weight count, not
    population-weighted, which is close enough for the model to pick it up.
    """
    local_dates = pd.Index(index.tz_convert(LOCAL_TZ).date)
    years = sorted({d.year for d in local_dates} | {local_dates[-1].year + 1})
    national = holidays.country_holidays("DE", years=years)
    per_state = [holidays.country_holidays("DE", subdiv=st, years=years) for st in GERMAN_STATES]

    is_holiday = np.array([d in national for d in local_dates], dtype=float)
    state_share = np.array(
        [sum(d in cal for cal in per_state) / len(per_state) for d in local_dates], dtype=float
    )
    one_day = pd.Timedelta(days=1)
    day_before = np.array([(d + one_day) in national for d in local_dates], dtype=float)
    day_after = np.array([(d - one_day) in national for d in local_dates], dtype=float)
    month_day = np.array([(d.month, d.day) for d in local_dates])
    christmas_period = (
        ((month_day[:, 0] == 12) & (month_day[:, 1] >= 24))
        | ((month_day[:, 0] == 1) & (month_day[:, 1] <= 6))
    ).astype(float)

    return pd.DataFrame(
        {
            "is_holiday": is_holiday,
            "holiday_state_share": state_share,
            "day_before_holiday": day_before,
            "day_after_holiday": day_after,
            "christmas_period": christmas_period,
        },
        index=index,
    )


def build_features(series: pd.Series) -> pd.DataFrame:
    """Feature matrix aligned to `series`. Every column is known 168 hours ahead."""
    local = series.index.tz_convert(LOCAL_TZ)
    frame = pd.DataFrame(index=series.index)
    frame["hour"] = local.hour
    frame["day_of_week"] = local.dayofweek
    frame["day_of_year"] = local.dayofyear
    frame["is_weekend"] = (local.dayofweek >= 5).astype(float)
    frame = frame.join(holiday_features(series.index))

    for weeks in (1, 2, 3):
        frame[f"lag_{weeks}w"] = series.shift(HORIZON_HOURS * weeks)
    frame["lag_52w"] = series.shift(HORIZON_HOURS * 52)
    # Level features: rolling means that end exactly 168 hours before the target.
    lagged = series.shift(HORIZON_HOURS)
    frame["mean_prev_week"] = lagged.rolling(HORIZON_HOURS).mean()
    frame["mean_prev_4weeks"] = lagged.rolling(HORIZON_HOURS * 4).mean()
    # Same-hour profile from the week before the origin, as a ratio to its weekly mean,
    # tells the model how peaked the daily shape has recently been.
    frame["lag_1w_ratio"] = frame["lag_1w"] / frame["mean_prev_week"]
    return frame


def fit_and_forecast(series: pd.Series, features: pd.DataFrame, start: pd.Timestamp) -> pd.Series:
    """Train on everything strictly before `start`, forecast the next HORIZON_HOURS."""
    end = start + pd.Timedelta(hours=HORIZON_HOURS - 1)
    train_mask = (features.index < start) & features.notna().all(axis=1)
    x_train = features.loc[train_mask]
    y_train = series.loc[train_mask]
    x_test = features.loc[start:end]
    if len(x_test) != HORIZON_HOURS or x_test.isna().any().any():
        raise ValueError(f"forecast window starting {start} is incomplete")

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    predicted = np.asarray(model.predict(x_test), dtype=float)
    return pd.Series(predicted, index=x_test.index, name="forecast_mw")


def baselines(series: pd.Series, start: pd.Timestamp) -> dict[str, pd.Series]:
    end = start + pd.Timedelta(hours=HORIZON_HOURS - 1)
    window = series.loc[start:end].index
    return {
        "Seasonal naive (same hour, 1 week ago)": series.shift(HORIZON_HOURS).loc[window],
        "Same hour 52 weeks ago": series.shift(HORIZON_HOURS * 52).loc[window],
    }


def plot(actual: pd.Series, forecast: pd.Series, metrics: Metrics, out: Path) -> None:
    # Naive Berlin-local timestamps: matplotlib then needs no timezone handling at all.
    local_index = actual.index.tz_convert(LOCAL_TZ).tz_localize(None)
    plt.rcParams["font.family"] = "sans-serif"
    fig, (ax, ax_err) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
    )
    fig.patch.set_facecolor(SURFACE)

    ax.plot(local_index, actual.to_numpy(), color=SERIES_ACTUAL, lw=2, label="Actual load")
    ax.plot(
        local_index, forecast.to_numpy(), color=SERIES_FORECAST, lw=2, label="LightGBM forecast"
    )
    ax.set_ylabel("Load (MW)", color=INK_SECONDARY)
    ax.set_title(
        "German electricity load, 1 to 7 January 2020 (times in Europe/Berlin)\n"
        f"Forecast made at 2019-12-31 23:00 UTC for all 168 hours. "
        f"MAE {metrics.mae_mw:,.0f} MW, MAPE {metrics.mape_pct:.2f} %",
        loc="left",
        color=INK,
        fontsize=11,
    )
    ax.legend(loc="upper left", frameon=False, labelcolor=INK_SECONDARY)

    residual = forecast - actual
    ax_err.fill_between(local_index, 0, residual.to_numpy(), color=SERIES_FORECAST, alpha=0.15)
    ax_err.plot(local_index, residual.to_numpy(), color=SERIES_FORECAST, lw=1.5)
    ax_err.axhline(0, color=MUTED, lw=1)
    ax_err.set_ylabel("Forecast minus\nactual (MW)", color=INK_SECONDARY)

    for axis in (ax, ax_err):
        axis.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
        axis.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, length=0)
        axis.grid(axis="y", color=GRID, lw=1)
        axis.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        axis.grid(axis="x", color=GRID, lw=1)
    midnights = pd.date_range(local_index[0].ceil("D"), local_index[-1], freq="D")
    ax_err.set_xticks(midnights, [t.strftime("%a %d %b") for t in midnights])

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    series = load_series(DATA_PATH)
    features = build_features(series)

    print("Backtest on the same week one year earlier (train < 2019-01-01):")
    back_forecast = fit_and_forecast(series, features, BACKTEST_START)
    back_actual = series.loc[back_forecast.index]
    print("  " + Metrics.from_series(back_actual, back_forecast).row("LightGBM"))
    for label, base in baselines(series, BACKTEST_START).items():
        print("  " + Metrics.from_series(back_actual, base).row(label))

    print("\nTarget week, 2020-01-01 00:00 to 2020-01-07 23:00 UTC (train < 2020-01-01):")
    forecast = fit_and_forecast(series, features, TEST_START)
    actual = series.loc[forecast.index]
    metrics = Metrics.from_series(actual, forecast)
    print("  " + metrics.row("LightGBM"))
    for label, base in baselines(series, TEST_START).items():
        print("  " + Metrics.from_series(actual, base).row(label))

    daily = (forecast - actual).groupby(forecast.index.tz_convert(LOCAL_TZ).date)
    print("\nDaily error of the LightGBM forecast (Berlin dates):")
    for day, err in daily:
        print(f"  {day}  mean error {err.mean():+7.0f} MW   MAE {err.abs().mean():6.0f} MW")

    out_csv = Path(__file__).with_name("forecast_jan2020.csv")
    out_png = Path(__file__).with_name("forecast_jan2020.png")
    pd.DataFrame({"actual_mw": actual, "forecast_mw": forecast.round(0)}).to_csv(out_csv)
    plot(actual, forecast, metrics, out_png)
    print(f"\nWrote {out_csv.name} and {out_png.name}")


if __name__ == "__main__":
    main()
