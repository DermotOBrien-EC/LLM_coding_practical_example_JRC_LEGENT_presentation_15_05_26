"""Forecast German hourly electricity load for 1 to 7 January 2020.

Direct 168-hour-ahead LightGBM model on calendar, holiday and weekly-lag
features. Every lag is at least 168 h so no recursion is needed. Validated on
the analogous week of 2019 before the final fit through 2019-12-31.
"""

from __future__ import annotations

from dataclasses import dataclass

import holidays
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TZ = "Europe/Berlin"
HORIZON = 168
LAGS = [168, 336, 504, 672, 8736]  # 1 to 4 weeks, and 52 weeks (same weekday)
SEED = 20200101


def load_series() -> pd.Series:
    df = pd.read_csv("opsd_de_load.csv", parse_dates=["utc_timestamp"], index_col="utc_timestamp")
    s = df["DE_load_actual_entsoe_transparency"].tz_convert(TZ)
    s.name = "load_mw"
    return s


def build_features(s: pd.Series) -> pd.DataFrame:
    idx = s.index
    de_hol = holidays.DE(years=range(idx.year.min(), idx.year.max() + 1))
    dates = idx.normalize().date
    f = pd.DataFrame(index=idx)
    f["hour"] = idx.hour
    f["dow"] = idx.dayofweek
    f["doy"] = idx.dayofyear
    f["month"] = idx.month
    f["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    f["is_holiday"] = np.array([d in de_hol for d in dates], dtype=int)
    # Christmas to Epiphany trough: industrial demand is depressed for the whole span.
    f["xmas_period"] = (
        ((idx.month == 12) & (idx.day >= 24)) | ((idx.month == 1) & (idx.day <= 6))
    ).astype(int)
    for lag in LAGS:
        f[f"lag_{lag}"] = s.shift(lag, freq="h").reindex(idx).to_numpy()
    f["y"] = s.to_numpy()
    return f


@dataclass
class Run:
    origin: pd.Timestamp
    pred: pd.Series
    p10: pd.Series
    p90: pd.Series
    naive_week: pd.Series
    naive_year: pd.Series
    actual: pd.Series


def fit_predict(f: pd.DataFrame, origin: pd.Timestamp) -> Run:
    end = origin + pd.Timedelta(hours=HORIZON - 1)
    train = f.loc[: origin - pd.Timedelta(hours=1)].dropna()
    test = f.loc[origin:end]
    assert len(test) == HORIZON, len(test)
    feats = [c for c in f.columns if c != "y"]
    cats = ["hour", "dow", "month"]

    def fit(objective: str, alpha: float | None = None) -> np.ndarray:
        params: dict[str, object] = {
            "objective": objective,
            "n_estimators": 1500,
            "learning_rate": 0.02,
            "num_leaves": 63,
            "min_child_samples": 40,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "random_state": SEED,
            "verbose": -1,
        }
        if alpha is not None:
            params["alpha"] = alpha
        m = lgb.LGBMRegressor(**params)
        m.fit(train[feats], train["y"], categorical_feature=cats)
        return np.asarray(m.predict(test[feats]))

    pred = pd.Series(fit("l2"), index=test.index, name="forecast_mw")
    p10 = pd.Series(fit("quantile", 0.1), index=test.index, name="p10_mw")
    p90 = pd.Series(fit("quantile", 0.9), index=test.index, name="p90_mw")
    return Run(
        origin=origin,
        pred=pred,
        p10=p10,
        p90=p90,
        naive_week=test["lag_168"].rename("naive_week_mw"),
        naive_year=test["lag_8736"].rename("naive_year_mw"),
        actual=test["y"].rename("actual_mw"),
    )


def metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    err = pred - actual
    return {
        "MAE_MW": float(err.abs().mean()),
        "RMSE_MW": float(np.sqrt((err**2).mean())),
        "MAPE_%": float((err.abs() / actual).mean() * 100),
        "bias_MW": float(err.mean()),
    }


def report(run: Run, label: str) -> None:
    rows = {
        "LightGBM": metrics(run.actual, run.pred),
        "Seasonal naive (t-168h)": metrics(run.actual, run.naive_week),
        "Year-ago same weekday (t-52w)": metrics(run.actual, run.naive_year),
    }
    cover = float(((run.actual >= run.p10) & (run.actual <= run.p90)).mean() * 100)
    print(f"\n== {label}: origin {run.origin} ==")
    print(pd.DataFrame(rows).T.round(1).to_string())
    print(f"p10-p90 band coverage: {cover:.0f}% of hours (target 80%)")
    daily = pd.DataFrame({"actual": run.actual, "forecast": run.pred})
    daily["ape_%"] = (daily["forecast"] - daily["actual"]).abs() / daily["actual"] * 100
    d = daily.groupby(daily.index.date).agg(
        actual_mean=("actual", "mean"), forecast_mean=("forecast", "mean"), mape=("ape_%", "mean")
    )
    print(d.round(1).to_string())


def plot(run: Run, path: str) -> None:
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    x = run.pred.index
    ax.fill_between(
        x, run.p10, run.p90, color=orange, alpha=0.18, linewidth=0, label="LightGBM p10 to p90"
    )
    ax.plot(
        x,
        run.naive_week,
        color=aqua,
        linewidth=1.4,
        linestyle="--",
        label="Seasonal naive (t-168h)",
    )
    ax.plot(x, run.pred, color=orange, linewidth=2, label="LightGBM forecast")
    ax.plot(x, run.actual, color=blue, linewidth=2, label="Actual (ENTSO-E)")
    ax.set_title(
        "Germany hourly electricity load, 1 to 7 January 2020 (Europe/Berlin)",
        loc="left",
        fontsize=12,
        color="#0b0b0b",
    )
    ax.set_ylabel("Load (GW)", color="#52514e")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=x.tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d Jan", tz=x.tz))
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, loc="upper left", ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    s = load_series()
    f = build_features(s)

    val = fit_predict(f, pd.Timestamp("2019-01-01 00:00", tz=TZ))
    report(val, "Validation week (held out)")

    fin = fit_predict(f, pd.Timestamp("2020-01-01 00:00", tz=TZ))
    report(fin, "Target week 1 to 7 Jan 2020 (model trained through 2019-12-31)")

    out = pd.concat([fin.pred, fin.p10, fin.p90, fin.naive_week, fin.actual], axis=1).round(0)
    out.insert(0, "utc_timestamp", out.index.tz_convert("UTC"))
    out.index.name = "local_timestamp"
    out.to_csv("forecast_jan2020_week1.csv")
    plot(fin, "forecast_jan2020_week1.png")
    print("\nwrote forecast_jan2020_week1.csv and forecast_jan2020_week1.png")


if __name__ == "__main__":
    main()
