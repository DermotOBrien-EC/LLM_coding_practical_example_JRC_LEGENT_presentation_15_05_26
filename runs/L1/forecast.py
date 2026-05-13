"""Forecast German hourly electricity load for the first week of January 2020.

Three simple baselines, all univariate on `DE_load_actual_entsoe_transparency`:

1. naive_w   — value 168 hours (1 week) before the forecast hour.
2. naive_y   — value 364 days (52 weeks, day-of-week aligned) before.
3. climo     — mean of (month=Jan, day-of-week, hour) over the training years,
               i.e. an hour-of-week climatology restricted to January.

The 2020-01-01..2020-01-07 actuals are already in the CSV, so we score each
forecast (MAE, RMSE, MAPE) and save a plot.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CSV = HERE / "opsd_de_load.csv"
TARGET_START = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
TARGET_END = pd.Timestamp("2020-01-08 00:00:00", tz="UTC")  # exclusive


def load_series() -> pd.Series:
    df = pd.read_csv(CSV, parse_dates=["utc_timestamp"])
    s = df.set_index("utc_timestamp")["DE_load_actual_entsoe_transparency"]
    s = s.asfreq("h")  # enforce regular hourly index
    assert s.isna().sum() == 0, "unexpected gaps in load series"
    return s.rename("load")


def forecast_naive_week(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    return s.shift(168).reindex(idx).rename("naive_w")


def forecast_naive_year(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    # 364 days = 52 * 7 so day-of-week is preserved.
    return s.shift(24 * 364).reindex(idx).rename("naive_y")


def forecast_climatology(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    train = s.loc[s.index < TARGET_START]
    grp_keys = pd.DataFrame(
        {
            "month": train.index.month,
            "dow": train.index.dayofweek,
            "hour": train.index.hour,
        },
        index=train.index,
    )
    table = train.groupby([grp_keys["month"], grp_keys["dow"], grp_keys["hour"]]).mean()
    keys = list(zip(idx.month, idx.dayofweek, idx.hour))
    return pd.Series(table.loc[keys].to_numpy(), index=idx, name="climo")


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    err = y_true - y_pred
    return {
        "MAE_MW": float(err.abs().mean()),
        "RMSE_MW": float(np.sqrt((err**2).mean())),
        "MAPE_pct": float((err.abs() / y_true).mean() * 100),
    }


def main() -> None:
    s = load_series()
    idx = pd.date_range(TARGET_START, TARGET_END, freq="h", inclusive="left")
    actual = s.reindex(idx).rename("actual")

    preds = pd.concat(
        [
            forecast_naive_week(s, idx),
            forecast_naive_year(s, idx),
            forecast_climatology(s, idx),
        ],
        axis=1,
    )
    out = pd.concat([actual, preds], axis=1)
    out.to_csv(HERE / "forecast.csv", index_label="utc_timestamp")

    scores = {col: metrics(actual, preds[col]) for col in preds.columns}
    score_df = pd.DataFrame(scores).T.round(2)
    print("Scores for 2020-01-01..2020-01-07 (168 hours):")
    print(score_df.to_string())
    score_df.to_csv(HERE / "metrics.csv")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(out.index, out["actual"], color="black", lw=1.6, label="actual")
    for col, style in [("naive_w", "--"), ("naive_y", ":"), ("climo", "-.")]:
        ax.plot(out.index, out[col], style, lw=1.1, label=col)
    ax.set_title("German hourly load — forecast vs actual, 2020-01-01..2020-01-07")
    ax.set_ylabel("Load (MW)")
    ax.set_xlabel("UTC time")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(HERE / "forecast.png", dpi=130)
    print(f"Wrote {HERE/'forecast.csv'}, {HERE/'metrics.csv'}, {HERE/'forecast.png'}")


if __name__ == "__main__":
    main()
