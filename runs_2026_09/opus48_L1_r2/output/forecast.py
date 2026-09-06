from __future__ import annotations

import numpy as np
import pandas as pd
import holidays
from sklearn.ensemble import HistGradientBoostingRegressor

TARGET_START = pd.Timestamp("2020-01-01", tz="UTC")
TARGET_END = pd.Timestamp("2020-01-08", tz="UTC")  # exclusive


def load_data(path: str = "opsd_de_load.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["utc_timestamp"])
    df = df.rename(columns={"DE_load_actual_entsoe_transparency": "load"})
    df = df.sort_values("utc_timestamp").reset_index(drop=True)
    df = df.set_index("utc_timestamp")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar + holiday + observed-lag features. All are known for the
    target week when forecasting is done from an origin before 2020-01-01."""
    local = df.index.tz_convert("Europe/Berlin")
    out = pd.DataFrame(index=df.index)

    hour = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    month = local.month.to_numpy()
    doy = local.dayofyear.to_numpy()

    out["hour"] = hour
    out["dow"] = dow
    out["month"] = month
    out["is_weekend"] = (dow >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # German federal public holidays (national aggregate load).
    de_hol = holidays.Germany(years=range(2014, 2021))
    dates = pd.Series(local.date, index=df.index)
    is_hol = dates.map(lambda d: d in de_hol).astype(int)
    out["is_holiday"] = is_hol.to_numpy()
    # day adjacent to a holiday (bridge / reduced-activity days)
    out["is_holiday_adj"] = (
        (is_hol.shift(24, fill_value=0) == 1) | (is_hol.shift(-24, fill_value=0) == 1)
    ).astype(int).to_numpy()

    # slow multi-year trend (years since series start)
    out["trend_years"] = (df.index - df.index[0]).total_seconds().to_numpy() / (
        365.25 * 24 * 3600
    )

    # Fully-observed lags relative to the target week:
    #   168 h  -> Dec 25-31 2019 (known actuals)
    #   8736 h == 364 days -> same weekday one year earlier (known actuals)
    out["lag_168h"] = df["load"].shift(168).to_numpy()
    out["lag_364d"] = df["load"].shift(24 * 364).to_numpy()

    return out


def main() -> None:
    df = load_data()

    feats = build_features(df)
    y = df["load"]

    is_target = (df.index >= TARGET_START) & (df.index < TARGET_END)
    is_train = df.index < TARGET_START

    X = feats.copy()
    train_mask = is_train & X.notna().all(axis=1)
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[is_target], y[is_target]

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=600,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    # Baselines
    naive_year = feats.loc[is_target, "lag_364d"].to_numpy()  # same hour, 364d ago
    naive_week = feats.loc[is_target, "lag_168h"].to_numpy()  # one week ago

    def metrics(a: np.ndarray, f: np.ndarray) -> dict[str, float]:
        a = np.asarray(a, float)
        f = np.asarray(f, float)
        err = f - a
        return {
            "MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err**2))),
            "MAPE_%": float(np.mean(np.abs(err / a)) * 100),
            "bias": float(np.mean(err)),
        }

    actual = yte.to_numpy()
    print("=== First week of January 2020 — DE hourly load forecast ===")
    print(f"Train rows: {len(ytr):,}  ({Xtr.index.min()} -> {Xtr.index.max()})")
    print(f"Horizon: 168 h  ({Xte.index.min()} -> {Xte.index.max()})\n")

    for name, f in [
        ("GBM model", pred),
        ("Seasonal-naive (364d)", naive_year),
        ("Naive (1 week)", naive_week),
    ]:
        m = metrics(actual, f)
        print(
            f"{name:<24} MAPE {m['MAPE_%']:5.2f}%  MAE {m['MAE']:7.0f} MW  "
            f"RMSE {m['RMSE']:7.0f} MW  bias {m['bias']:+7.0f} MW"
        )

    out = pd.DataFrame(
        {
            "utc_timestamp": Xte.index,
            "local_time": Xte.index.tz_convert("Europe/Berlin"),
            "forecast_MW": np.round(pred, 1),
            "actual_MW": actual,
        }
    )
    out["abs_pct_err"] = (
        (out["forecast_MW"] - out["actual_MW"]).abs() / out["actual_MW"] * 100
    ).round(2)
    out.to_csv("forecast_jan2020_week1.csv", index=False)
    print("\nWrote forecast_jan2020_week1.csv")

    # Daily summary
    out["day"] = out["local_time"].dt.date
    daily = out.groupby("day").agg(
        forecast_MWh=("forecast_MW", "sum"),
        actual_MWh=("actual_MW", "sum"),
        mape=("abs_pct_err", "mean"),
    )
    daily["day_err_%"] = (
        (daily["forecast_MWh"] - daily["actual_MWh"]) / daily["actual_MWh"] * 100
    ).round(2)
    print("\nDaily totals (MWh) and error:")
    with pd.option_context("display.width", 120):
        print(daily.round(0))

    # feature importance via permutation would be costly; report training gain proxy
    np.save("_pred.npy", pred)
    np.save("_actual.npy", actual)
    out.to_pickle("_out.pkl")


if __name__ == "__main__":
    main()
