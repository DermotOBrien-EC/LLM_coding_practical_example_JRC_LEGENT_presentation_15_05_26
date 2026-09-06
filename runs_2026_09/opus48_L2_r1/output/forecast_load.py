"""Forecast German hourly electricity load for the first week of January 2020.

Trains on OPSD hourly load history (2015 up to the forecast origin), produces a
genuine ex-ante 168-hour forecast for 2020-01-01..2020-01-07, plots it against
the actuals, and reports accuracy against a seasonal-naive baseline.

No look-ahead leakage: the model only uses features knowable at the forecast
origin (2019-12-31 23:00 UTC) for the entire horizon. That means calendar terms
plus lags of at least 168 hours (a sub-week lag would need actuals from inside
the forecast window, which we do not have when standing at the origin).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import holidays
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_FILE = Path(__file__).parent / "opsd_de_load.csv"
FORECAST_START = pd.Timestamp("2020-01-01 00:00", tz="UTC")
FORECAST_END = pd.Timestamp("2020-01-07 23:00", tz="UTC")  # inclusive, 168 hours
LOAD_COL = "DE_load_actual_entsoe_transparency"

FEATURES = [
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_holiday",
    "lag_168",
    "lag_336",
]

# dataviz palette (light surface): blue = actual, orange = forecast, aqua = naive
COL_ACTUAL = "#2a78d6"
COL_FORECAST = "#eb6834"
COL_NAIVE = "#1baf7a"
COL_GRID = "#d9d8d4"
COL_INK = "#0b0b0b"
COL_INK_SOFT = "#52514e"
SURFACE = "#fcfcfb"


@dataclass(frozen=True)
class Scores:
    """Accuracy of a forecast against the actuals."""

    mape: float  # mean absolute percentage error (%)
    mae: float  # mean absolute error (MW)
    rmse: float  # root mean squared error (MW)

    @classmethod
    def compute(cls, actual: np.ndarray, predicted: np.ndarray) -> Scores:
        err = predicted - actual
        return cls(
            mape=float(np.mean(np.abs(err / actual)) * 100.0),
            mae=float(np.mean(np.abs(err))),
            rmse=float(np.sqrt(np.mean(err**2))),
        )


def load_series(path: Path) -> pd.Series:
    """Read the CSV into an hourly, gap-free, UTC-indexed load series (MW)."""
    df = pd.read_csv(path, parse_dates=["utc_timestamp"])
    s = df.set_index("utc_timestamp")[LOAD_COL].sort_index()
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC")
    # Enforce a regular hourly grid and fill the handful of OPSD gaps so the
    # 168h lag features line up exactly.
    s = s.asfreq("h")
    n_missing = int(s.isna().sum())
    if n_missing:
        s = s.interpolate(method="time", limit_direction="both")
        print(f"Filled {n_missing} missing hourly value(s) by time interpolation.")
    return s


def build_features(s: pd.Series) -> pd.DataFrame:
    """Calendar + safe-lag feature matrix aligned to the load series."""
    idx = s.index
    de_holidays = holidays.Germany(years=range(idx.year.min(), idx.year.max() + 1))
    doy = idx.dayofyear.to_numpy(dtype=float)
    df = pd.DataFrame(
        {
            "load": s.to_numpy(),
            "hour": idx.hour,
            "dayofweek": idx.dayofweek,
            "month": idx.month,
            "dayofyear": doy,
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "is_weekend": (idx.dayofweek >= 5).astype(int),
            "is_holiday": np.array([d.date() in de_holidays for d in idx], dtype=int),
            "lag_168": s.shift(168).to_numpy(),
            "lag_336": s.shift(336).to_numpy(),
        },
        index=idx,
    )
    return df


def main() -> None:
    load = load_series(DATA_FILE)
    horizon = pd.date_range(FORECAST_START, FORECAST_END, freq="h", tz="UTC")
    if not horizon.isin(load.index).all():
        raise SystemExit("Forecast window is not fully present in the data.")

    feats = build_features(load)
    train = feats[feats.index < FORECAST_START].dropna(subset=FEATURES + ["load"])
    test = feats.loc[horizon]

    model = HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.05,
        max_depth=8,
        l2_regularization=1.0,
        random_state=0,
    )
    model.fit(train[FEATURES], train["load"])
    print(f"Trained on {len(train):,} hours "
          f"({train.index.min():%Y-%m-%d} to {train.index.max():%Y-%m-%d}).")

    forecast = pd.Series(model.predict(test[FEATURES]), index=horizon, name="forecast")
    actual = load.loc[horizon]
    naive = test["lag_168"]  # seasonal-naive: same hour one week earlier

    model_scores = Scores.compute(actual.to_numpy(), forecast.to_numpy())
    naive_scores = Scores.compute(actual.to_numpy(), naive.to_numpy())

    report(model_scores, naive_scores)
    save_outputs(actual, forecast, naive, model_scores, naive_scores)


def report(model: Scores, naive: Scores) -> None:
    print("\nForecast accuracy for 2020-01-01 .. 2020-01-07 (168 hours)")
    print("-" * 58)
    print(f"{'':16}{'MAPE':>9}{'MAE (MW)':>13}{'RMSE (MW)':>13}")
    print(f"{'Model (GBM)':16}{model.mape:>8.2f}%{model.mae:>13,.0f}{model.rmse:>13,.0f}")
    print(f"{'Naive (t-168h)':16}{naive.mape:>8.2f}%{naive.mae:>13,.0f}{naive.rmse:>13,.0f}")
    lift = (1 - model.mape / naive.mape) * 100
    print("-" * 58)
    print(f"Model beats the seasonal-naive baseline by {lift:.0f}% on MAPE.")


def save_outputs(
    actual: pd.Series,
    forecast: pd.Series,
    naive: pd.Series,
    model: Scores,
    naive_scores: Scores,
) -> None:
    out_csv = DATA_FILE.parent / "forecast_jan2020.csv"
    pd.DataFrame(
        {"actual_MW": actual, "forecast_MW": forecast, "naive_MW": naive}
    ).to_csv(out_csv, index_label="utc_timestamp")

    fig, ax = plt.subplots(figsize=(13, 6), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(naive.index, naive / 1000, color=COL_NAIVE, lw=1.4, ls="--",
            alpha=0.9, label="Naive (same hour, prev. week)", zorder=2)
    ax.plot(actual.index, actual / 1000, color=COL_ACTUAL, lw=2.2,
            label="Actual", zorder=4)
    ax.plot(forecast.index, forecast / 1000, color=COL_FORECAST, lw=2.2,
            label="Forecast (GBM)", zorder=3)

    ax.set_title("German electricity load: forecast vs actual, first week of January 2020",
                 fontsize=14, color=COL_INK, pad=12, loc="left")
    ax.set_ylabel("Load (GW)", color=COL_INK_SOFT, fontsize=11)
    ax.set_xlabel("")

    ax.xaxis.set_major_locator(mdates.DayLocator(tz="UTC"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%b %d", tz="UTC"))
    ax.grid(True, which="major", color=COL_GRID, lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COL_GRID)
    ax.tick_params(colors=COL_INK_SOFT)

    ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=COL_INK)
    ax.text(0.006, 0.97,
            f"Model MAPE {model.mape:.1f}%  ·  MAE {model.mae:,.0f} MW\n"
            f"Baseline MAPE {naive_scores.mape:.1f}%",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            color=COL_INK, linespacing=1.4)

    fig.tight_layout()
    out_png = DATA_FILE.parent / "forecast_jan2020.png"
    fig.savefig(out_png, facecolor=SURFACE)
    print(f"\nWrote {out_csv.name} and {out_png.name}")


if __name__ == "__main__":
    main()
