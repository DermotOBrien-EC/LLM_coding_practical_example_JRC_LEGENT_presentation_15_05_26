"""
Forecast German hourly electricity load for 2020-01-01..2020-01-07 (168 hours).

Method: seasonal-DOW average over the same calendar week (Jan 1..7)
across the five preceding years (2015-2019), aligned by (day-of-week, hour).
Public holidays that always fall on Jan 1 and Jan 6 are handled by pooling
those calendar dates directly across the training years, since their DOW
shifts each year and their load profile is markedly different from a
normal weekday.

Train:   OPSD DE load, 2015-01-01 .. 2019-12-31
Predict: 2020-01-01 00:00 .. 2020-01-07 23:00 (UTC)
Score:   held-out actuals in the same file are used only for MAPE/MAE report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CSV = "opsd_de_load.csv"
FORECAST_START = pd.Timestamp("2020-01-01 00:00", tz="UTC")
FORECAST_END = pd.Timestamp("2020-01-07 23:00", tz="UTC")


def load_data() -> pd.Series:
    df = pd.read_csv(CSV, parse_dates=["utc_timestamp"])
    s = df.set_index("utc_timestamp")["DE_load_actual_entsoe_transparency"]
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC")
    return s.asfreq("h").rename("load_mw")


def build_forecast(load: pd.Series) -> pd.Series:
    train = load.loc[:"2019-12-31 23:00"].dropna()

    # Restrict to the first week of January in each training year.
    first_week = train[
        (train.index.month == 1) & (train.index.day.isin(range(1, 8)))
    ].copy()

    # Two components combined multiplicatively.
    #
    # 1. dow_profile[dow, hour]: normal-week shape from the rest of
    #    January (days 8..31 of Jan across 2015-2019), so New-Year-week
    #    anomalies do not contaminate the "typical" DOW×hour pattern.
    rest_of_jan = train[(train.index.month == 1) & (train.index.day >= 8)]
    dow_profile = (
        rest_of_jan.groupby([rest_of_jan.index.dayofweek, rest_of_jan.index.hour])
        .mean()
        .rename_axis(["dow", "hour"])
    )

    # 2. calendar_factor[day, hour]: how a given Jan-day-of-month
    #    (1..7) deviates from what its own DOW×hour would predict, pooled
    #    across 2015-2019. Captures Jan 1 holiday, Jan 2/3 bridge, Jan 6
    #    Epiphany, and the tail-of-week recovery, all as calendar effects.
    ref = first_week.index.map(
        lambda ts: dow_profile.loc[(ts.dayofweek, ts.hour)]
    )
    residual_ratio = pd.Series(first_week.values / ref.values, index=first_week.index)
    calendar_factor = (
        residual_ratio.groupby(
            [first_week.index.day, first_week.index.hour]
        )
        .mean()
        .rename_axis(["day", "hour"])
    )

    # 3. Level correction: German load has drifted down over 2015-2019.
    #    Use the most recent complete month (Dec 2019) versus the mean of
    #    all training Decembers as a scalar multiplier. Keeps the shape
    #    fixed but re-levels to the current operating point.
    dec = train[train.index.month == 12]
    level_factor = (
        dec.loc["2019-12"].mean() / dec.mean()
    )

    horizon = pd.date_range(FORECAST_START, FORECAST_END, freq="h", tz="UTC")
    yhat = pd.Series(index=horizon, dtype=float, name="forecast_mw")
    for ts in horizon:
        base = dow_profile.loc[(ts.dayofweek, ts.hour)]
        factor = calendar_factor.loc[(ts.day, ts.hour)]
        yhat.loc[ts] = base * factor * level_factor

    return yhat.round(1)


def score(yhat: pd.Series, load: pd.Series) -> dict:
    actual = load.reindex(yhat.index)
    err = yhat - actual
    ape = (err.abs() / actual).replace([np.inf, -np.inf], np.nan)
    return {
        "n": int(actual.notna().sum()),
        "MAE_MW": float(err.abs().mean()),
        "RMSE_MW": float(np.sqrt((err ** 2).mean())),
        "MAPE_pct": float(100 * ape.mean()),
        "bias_MW": float(err.mean()),
        "actual_mean_MW": float(actual.mean()),
        "forecast_mean_MW": float(yhat.mean()),
    }


def daily_summary(yhat: pd.Series, load: pd.Series) -> pd.DataFrame:
    actual = load.reindex(yhat.index)
    local = yhat.index.tz_convert("Europe/Berlin")
    key = pd.Index(local.date, name="date_CET")
    out = pd.DataFrame(
        {
            "forecast_mean_MW": yhat.groupby(key).mean().round(0),
            "actual_mean_MW": actual.groupby(key).mean().round(0),
            "forecast_peak_MW": yhat.groupby(key).max().round(0),
            "actual_peak_MW": actual.groupby(key).max().round(0),
        }
    )
    out["MAPE_pct"] = (
        100
        * (yhat - actual).abs().groupby(key).mean()
        / actual.groupby(key).mean()
    ).round(2)
    return out


def main() -> None:
    load = load_data()
    yhat = build_forecast(load)

    out_path = "forecast_2020_w01.csv"
    frame = yhat.to_frame()
    frame["actual_mw"] = load.reindex(yhat.index).round(1)
    frame.index.name = "utc_timestamp"
    frame.to_csv(out_path)

    metrics = score(yhat, load)
    print("Forecast horizon:", yhat.index.min(), "→", yhat.index.max(),
          f"({len(yhat)} hourly steps)")
    print("Written:", out_path)
    print()
    print("Skill vs held-out actuals:")
    for k, v in metrics.items():
        print(f"  {k:>18}: {v:,.2f}" if isinstance(v, float) else f"  {k:>18}: {v}")
    print()
    print("Daily rollup (CET):")
    print(daily_summary(yhat, load).to_string())


if __name__ == "__main__":
    main()
