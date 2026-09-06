"""Forecast German hourly electricity load for 2020-01-01..2020-01-07.

Trains a LightGBM regressor on OPSD hourly load (2015 through 2019-12-31)
using calendar features, a German holiday flag, and lags at 168 h (one week)
and 8760 h (one year). Compares against a seasonal-naive baseline that reuses
the load 168 h earlier. Writes forecast.png and prints MAE, RMSE, and MAPE.
"""

from __future__ import annotations

from pathlib import Path

import holidays
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "opsd_de_load.csv"
LOAD_COL = "DE_load_actual_entsoe_transparency"
FORECAST_START = pd.Timestamp("2020-01-01 00:00", tz="UTC")
FORECAST_END = pd.Timestamp("2020-01-07 23:00", tz="UTC")
LAGS_HOURS = (168, 336, 8760)


def load_series() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["utc_timestamp"])
    df = df.set_index("utc_timestamp").sort_index()
    s = df[LOAD_COL].astype(float)
    s.name = "load_mw"
    return s


def build_features(target_idx: pd.DatetimeIndex, history: pd.Series) -> pd.DataFrame:
    de_holidays = holidays.country_holidays("DE")
    local = target_idx.tz_convert("Europe/Berlin")
    df = pd.DataFrame(index=target_idx)
    df["hour"] = local.hour
    df["dow"] = local.dayofweek
    df["month"] = local.month
    df["doy"] = local.dayofyear
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_holiday"] = np.fromiter(
        (d.date() in de_holidays for d in local), dtype=int, count=len(local)
    )
    two_pi = 2.0 * np.pi
    df["hour_sin"] = np.sin(two_pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(two_pi * df["hour"] / 24.0)
    df["doy_sin"] = np.sin(two_pi * df["doy"] / 365.25)
    df["doy_cos"] = np.cos(two_pi * df["doy"] / 365.25)
    for h in LAGS_HOURS:
        shifted = target_idx - pd.Timedelta(hours=h)
        df[f"lag_{h}h"] = history.reindex(shifted).to_numpy()
    return df


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    return {
        "MAE_MW": float(np.mean(np.abs(err))),
        "RMSE_MW": float(np.sqrt(np.mean(err**2))),
        "MAPE_%": float(np.mean(np.abs(err) / np.abs(y_true)) * 100.0),
    }


def main() -> None:
    load = load_series()

    train = load.loc[: FORECAST_START - pd.Timedelta(hours=1)]
    test = load.loc[FORECAST_START:FORECAST_END].dropna()
    assert len(test) == 168, f"Expected 168 test hours, got {len(test)}"

    X_train = build_features(train.index, load)
    y_train = train.reindex(X_train.index)
    mask = X_train.notna().all(axis=1) & y_train.notna()
    X_train, y_train = X_train.loc[mask], y_train.loc[mask]

    model = lgb.LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=64,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        random_state=0,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    X_test = build_features(test.index, load)
    forecast = pd.Series(model.predict(X_test), index=test.index, name="forecast_mw")

    baseline_vals = load.reindex(test.index - pd.Timedelta(hours=168)).to_numpy()
    baseline = pd.Series(baseline_vals, index=test.index, name="naive_mw")

    m_model = metrics(test.to_numpy(), forecast.to_numpy())
    m_naive = metrics(test.to_numpy(), baseline.to_numpy())

    print("First-week 2020 forecast for Germany, 168 hours")
    print(f"  Training window : {X_train.index.min()} to {X_train.index.max()}  "
          f"({len(X_train):,} hours)")
    print(f"  Mean actual load: {test.mean():,.0f} MW")
    print()
    print("Accuracy on the 168 test hours:")
    print(
        f"  LightGBM        MAE {m_model['MAE_MW']:>6,.0f} MW   "
        f"RMSE {m_model['RMSE_MW']:>6,.0f} MW   MAPE {m_model['MAPE_%']:.2f} %"
    )
    print(
        f"  Naive (t-168 h) MAE {m_naive['MAE_MW']:>6,.0f} MW   "
        f"RMSE {m_naive['RMSE_MW']:>6,.0f} MW   MAPE {m_naive['MAPE_%']:.2f} %"
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    idx_local = test.index.tz_convert("Europe/Berlin")
    ax.plot(idx_local, test.values, color="#1f77b4", label="Actual", linewidth=1.8)
    ax.plot(
        idx_local,
        forecast.values,
        color="#d62728",
        label=f"LightGBM  (MAPE {m_model['MAPE_%']:.1f} %)",
        linewidth=1.4,
    )
    ax.plot(
        idx_local,
        baseline.values,
        color="#7f7f7f",
        linestyle="--",
        label=f"Seasonal-naive t-168 h  (MAPE {m_naive['MAPE_%']:.1f} %)",
        linewidth=1.1,
    )
    ax.set_title("Germany hourly electricity load, 2020-01-01 to 2020-01-07")
    ax.set_ylabel("Load (MW)")
    ax.set_xlabel("Local time (Europe/Berlin)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.autofmt_xdate()
    fig.tight_layout()
    out = HERE / "forecast.png"
    fig.savefig(out, dpi=140)
    print(f"\nPlot written to {out}")


if __name__ == "__main__":
    main()
