"""Forecast German hourly electricity load for 2020-01-01 to 2020-01-07.

Reads opsd_de_load.csv, trains a gradient-boosting model on the history
before 2020-01-01, predicts the next 168 hours, compares to actuals, and
writes forecast.csv, forecast.png, and metrics.csv next to this script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

HERE = Path(__file__).parent
DATA_PATH = HERE / "opsd_de_load.csv"
FORECAST_START = pd.Timestamp("2020-01-01", tz="UTC")
FORECAST_END = pd.Timestamp("2020-01-08", tz="UTC")  # exclusive: 168 hours
LAG_WEEK = 168       # same hour, one week earlier
LAG_YEAR = 24 * 365  # same hour, one year earlier (approx)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["utc_timestamp"])
    df = df.rename(columns={"DE_load_actual_entsoe_transparency": "load"})
    df = df.set_index("utc_timestamp").sort_index()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["lag_week"] = out["load"].shift(LAG_WEEK)
    out["lag_year"] = out["load"].shift(LAG_YEAR)
    return out


def main() -> None:
    df = load_data()
    feat = build_features(df).dropna()

    feature_cols = ["hour", "dayofweek", "month", "is_weekend", "lag_week", "lag_year"]
    train = feat.loc[feat.index < FORECAST_START]
    test = feat.loc[(feat.index >= FORECAST_START) & (feat.index < FORECAST_END)]
    assert len(test) == 168, f"expected 168 forecast hours, got {len(test)}"

    model = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        random_state=0,
    )
    model.fit(train[feature_cols], train["load"])
    pred = model.predict(test[feature_cols])

    actual = test["load"].to_numpy()
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / actual)) * 100)

    print(f"Forecast window: {FORECAST_START.date()} to {(FORECAST_END - pd.Timedelta(hours=1)).date()}  ({len(test)} hours)")
    print(f"MAE  : {mae:8.1f} MW")
    print(f"RMSE : {rmse:8.1f} MW")
    print(f"MAPE : {mape:8.2f} %")

    out = pd.DataFrame(
        {"timestamp": test.index, "actual_mw": actual, "predicted_mw": pred}
    )
    out.to_csv(HERE / "forecast.csv", index=False)
    pd.DataFrame(
        [{"metric": "MAE_MW", "value": mae},
         {"metric": "RMSE_MW", "value": rmse},
         {"metric": "MAPE_pct", "value": mape}]
    ).to_csv(HERE / "metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test.index, actual, label="Actual", color="black", linewidth=1.5)
    ax.plot(test.index, pred, label="Forecast", color="tab:orange", linewidth=1.5)
    ax.set_title(
        "German hourly electricity load — forecast vs actual\n"
        f"2020-01-01 to 2020-01-07   MAE={mae:.0f} MW   MAPE={mape:.2f}%"
    )
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Load (MW)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(HERE / "forecast.png", dpi=120)


if __name__ == "__main__":
    main()
