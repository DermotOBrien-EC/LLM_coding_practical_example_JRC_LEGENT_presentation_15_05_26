"""Produce and evaluate the 2020-01-01..07 German hourly load forecast."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecast.data import load_series
from forecast.evaluate import metrics
from forecast.intervals import apply_intervals, residual_quantiles
from forecast.models import (
    HORIZON,
    Ensemble,
    Forecaster,
    HolidayAdjustedNaive,
    LightGBMModel,
    LinearProfileModel,
    SeasonalNaive,
    YearAgoNaive,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
ORIGIN = pd.Timestamp("2020-01-01 00:00", tz="UTC")
TARGET = pd.date_range(ORIGIN, periods=HORIZON, freq="h", tz="UTC")
CALIBRATION_YEARS = [2016, 2017, 2018, 2019]


def build_selected() -> Forecaster:
    """The model chosen on prior-January validation folds only.

    Selection rule, fixed before scoring 2020: take the candidate that is best
    (or tied best) on both mean and worst-case MAPE across the four validation
    folds. A 0.5/0.25/0.25 log-space blend of LightGBM, the linear profile
    model and the year-ago naive won on both criteria over all four folds and
    was within 0.01pp of the best when the short-history 2016 fold was dropped.
    See outputs/validation_ensembles.csv.
    """
    return Ensemble(
        [LightGBMModel(), LinearProfileModel(), YearAgoNaive()], [0.5, 0.25, 0.25]
    )


def all_models() -> dict[str, object]:
    return {
        "seasonal_naive_168h": SeasonalNaive,
        "year_ago_naive_364d": YearAgoNaive,
        "holiday_adjusted_naive": HolidayAdjustedNaive,
        "linear_profile": LinearProfileModel,
        "lightgbm": LightGBMModel,
        "selected_ensemble": build_selected,
    }


def calibration_log_errors(series: pd.Series) -> pd.DataFrame:
    """Log-scale errors of the selected model on prior January weeks.

    Rows are folds, columns are hours-ahead, so interval width can widen with
    horizon the way the real error does.
    """
    rows: dict[int, np.ndarray] = {}
    for year in CALIBRATION_YEARS:
        start = pd.Timestamp(f"{year}-01-01 00:00", tz="UTC")
        index = pd.date_range(start, periods=HORIZON, freq="h", tz="UTC")
        history = series.loc[: start - pd.Timedelta(hours=1)]
        prediction = build_selected().fit(history).predict(index)
        actual = series.reindex(index)
        rows[year] = np.log(actual.to_numpy()) - np.log(prediction.to_numpy())
        print(
            f"  calibration fold {year}: MAPE {metrics(actual, prediction)['MAPE']:.2f}%",
            flush=True,
        )
    return pd.DataFrame(rows).T


def plot(actual: pd.Series, table: pd.DataFrame, others: pd.DataFrame) -> None:
    local_idx = actual.index.tz_convert("Europe/Berlin")
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.15]}
    )
    ax = axes[0]
    ax.fill_between(
        local_idx,
        table["q05_mw"],
        table["q95_mw"],
        color="#4C78A8",
        alpha=0.18,
        label="90% interval (empirical)",
    )
    ax.plot(local_idx, actual.to_numpy(), color="#333333", lw=2.0, label="Actual")
    ax.plot(
        local_idx,
        table["forecast_mw"],
        color="#4C78A8",
        lw=2.0,
        ls="--",
        label="Forecast (LightGBM+linear ensemble)",
    )
    ax.plot(
        local_idx,
        others["seasonal_naive_168h"],
        color="#E45756",
        lw=1.2,
        alpha=0.75,
        label="Seasonal naive (lag 168h)",
    )
    ax.set_ylabel("Load (MW)")
    ax.set_title(
        "German hourly electricity load, 1–7 January 2020\n"
        "168-hour ahead forecast, origin 2020-01-01 00:00 UTC"
    )
    ax.legend(loc="lower right", framealpha=0.92)
    ax.grid(alpha=0.25)

    err = table["forecast_mw"].to_numpy() - actual.to_numpy()
    axes[1].axhline(0, color="#888888", lw=1)
    axes[1].bar(local_idx, err, width=0.035, color=np.where(err >= 0, "#4C78A8", "#E45756"))
    axes[1].set_ylabel("Error (MW)")
    axes[1].set_xlabel("Local time (Europe/Berlin)")
    axes[1].grid(alpha=0.25)
    axes[1].xaxis.set_major_locator(mdates.DayLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
    fig.tight_layout()
    fig.savefig(OUT / "forecast_2020_week1.png", dpi=150)
    print(f"  wrote {OUT / 'forecast_2020_week1.png'}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    series = load_series(ROOT / "opsd_de_load.csv")
    history = series.loc[: ORIGIN - pd.Timedelta(hours=1)]
    actual = series.reindex(TARGET)
    print(f"history: {history.index[0]} .. {history.index[-1]}  ({len(history)} h)")

    print("\ncalibrating intervals on prior January weeks:")
    quantiles = residual_quantiles(calibration_log_errors(series))

    print("\nfitting final models on full history:")
    predictions: dict[str, pd.Series] = {}
    for label, factory in all_models().items():
        predictions[label] = factory().fit(history).predict(TARGET)
        print(f"  {label}: done", flush=True)
    frame = pd.DataFrame(predictions)

    table = apply_intervals(frame["selected_ensemble"], quantiles)
    table.insert(0, "local_time", table.index.tz_convert("Europe/Berlin"))
    table["actual_mw"] = actual.to_numpy()
    table["error_mw"] = table["forecast_mw"] - table["actual_mw"]
    table["abs_pct_error"] = (table["error_mw"] / table["actual_mw"]).abs() * 100
    table.round(2).to_csv(OUT / "forecast_2020_week1.csv", index_label="utc_timestamp")

    print("\n=== test-week accuracy, 2020-01-01..07 (168 h) ===")
    scores = pd.DataFrame({k: metrics(actual, v) for k, v in frame.items()}).T
    scores = scores.sort_values("MAPE")
    print(scores.round(2).to_string())
    scores.round(4).to_csv(OUT / "test_metrics.csv", index_label="model")

    covered = (table["actual_mw"] >= table["q05_mw"]) & (table["actual_mw"] <= table["q95_mw"])
    print(
        f"\n90% interval coverage on the test week: {covered.mean() * 100:.1f}% "
        f"({int(covered.sum())}/{len(covered)} hours)"
    )

    daily = pd.DataFrame(
        {
            "actual": actual.groupby(actual.index.tz_convert("Europe/Berlin").date).mean(),
            "forecast": frame["selected_ensemble"]
            .groupby(TARGET.tz_convert("Europe/Berlin").date)
            .mean(),
        }
    )
    daily["err_pct"] = (daily["forecast"] / daily["actual"] - 1) * 100
    print("\n=== daily mean load (MW) ===")
    print(daily.round(1).to_string())

    plot(actual, table, frame)
    print(f"  wrote {OUT / 'forecast_2020_week1.csv'}")


if __name__ == "__main__":
    main()
