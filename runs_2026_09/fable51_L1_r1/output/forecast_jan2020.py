"""Forecast German hourly load for 1-7 January 2020 (Europe/Berlin local time).

Training data: OPSD DE_load_actual_entsoe_transparency, hourly, 2015-01-01 to 2019-12-31.
Model: LightGBM on calendar/holiday features plus lags of >= 168 h, so the whole
168-hour horizon is predicted in one shot from data available at the forecast origin
(2020-01-01 00:00 local). Baselines: seasonal naive (same hour, previous week) and
weekday-aligned last year (lag 364 days). Backtest: the same week in 2017, 2018, 2019.
"""

from __future__ import annotations

import json
from pathlib import Path

import holidays
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parent
TZ = "Europe/Berlin"
HORIZON = 168

# Resident population per state, millions (Destatis 2019); weights Jan 6 style regional holidays.
STATE_POP: dict[str, float] = {
    "BW": 11.1, "BY": 13.1, "BE": 3.7, "BB": 2.5, "HB": 0.7, "HH": 1.8, "HE": 6.3, "MV": 1.6,
    "NI": 8.0, "NW": 17.9, "RP": 4.1, "SL": 1.0, "SN": 4.1, "ST": 2.2, "SH": 2.9, "TH": 2.1,
}
LAGS: dict[str, int] = {"lag168": 168, "lag336": 336, "lag504": 504, "lag364d": 364 * 24, "lag365d": 365 * 24}


def load_series() -> pd.Series:
    df = pd.read_csv(RUN / "opsd_de_load.csv", parse_dates=["utc_timestamp"], index_col="utc_timestamp")
    y = df.iloc[:, 0].rename("load")
    y.index = y.index.tz_convert(TZ)
    assert y.isna().sum() == 0 and y.index.is_monotonic_increasing
    return y


def holiday_frame(years: range) -> pd.DataFrame:
    nat = holidays.Germany(years=years)
    total = sum(STATE_POP.values())
    regional: dict[pd.Timestamp, float] = {}
    for code, pop in STATE_POP.items():
        for d in holidays.Germany(years=years, subdiv=code):
            if d not in nat:
                regional[pd.Timestamp(d)] = regional.get(pd.Timestamp(d), 0.0) + pop / total
    days = pd.date_range(f"{years.start}-01-01", f"{years.stop - 1}-12-31", freq="D")
    hf = pd.DataFrame(index=days)
    hf["hol_nat"] = [1.0 if d.date() in nat else 0.0 for d in days]
    hf["hol_reg"] = [regional.get(d, 0.0) for d in days]
    off = (hf["hol_nat"] == 1) | (hf.index.dayofweek >= 5)
    hf["day_before_off"] = off.shift(-1, fill_value=False).astype(float)
    hf["day_after_off"] = off.shift(1, fill_value=False).astype(float)
    # Bridge day: working day squeezed between two off days.
    hf["bridge"] = ((~off) & off.shift(1, fill_value=False) & off.shift(-1, fill_value=False)).astype(float)
    md = hf.index.month * 100 + hf.index.day
    in_xmas = (md >= 1224) | (md <= 106)
    hf["xmas_day"] = np.where(in_xmas, np.where(md >= 1224, md - 1224, hf.index.day + 7), -1).astype(float)
    return hf


def features(y: pd.Series) -> pd.DataFrame:
    idx = y.index
    hf = holiday_frame(range(idx.year.min(), idx.year.max() + 1))
    day = idx.tz_localize(None).normalize()
    X = pd.DataFrame(index=idx)
    X["hour"] = idx.hour
    X["dow"] = idx.dayofweek
    X["month"] = idx.month
    X["doy"] = idx.dayofyear
    X["year"] = idx.year
    for col in hf.columns:
        X[col] = hf[col].reindex(day).to_numpy()
    for name, h in LAGS.items():
        X[name] = y.shift(h).to_numpy()
    X["lag_week_mean"] = X[["lag168", "lag336", "lag504"]].mean(axis=1)
    X["lag168_ratio"] = X["lag168"] / X["lag_week_mean"]
    return X


def fit(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="l1",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        verbose=-1,
        random_state=1,
    )
    model.fit(X, y)
    return model


def window(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01 00:00", periods=HORIZON, freq="h", tz=TZ)


def score(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    err = actual - pred
    return {
        "mae_mw": float(err.abs().mean()),
        "rmse_mw": float(np.sqrt((err**2).mean())),
        "mape_pct": float((err.abs() / actual).mean() * 100),
    }


def main() -> None:
    y = load_series()
    X = features(y)
    usable = X.dropna().index

    results: dict[str, dict[str, dict[str, float]]] = {}
    ratios: list[np.ndarray] = []
    for yr in (2017, 2018, 2019):
        w = window(yr)
        train = usable[usable < w[0]]
        model = fit(X.loc[train], y.loc[train])
        pred = pd.Series(model.predict(X.loc[w]), index=w)
        ratios.append((y.loc[w] / pred).to_numpy())
        results[str(yr)] = {
            "lightgbm": score(y.loc[w], pred),
            "seasonal_naive_168h": score(y.loc[w], X.loc[w, "lag168"]),
            "last_year_364d": score(y.loc[w], X.loc[w, "lag364d"]),
        }
    lo_q, hi_q = np.quantile(np.concatenate(ratios), [0.1, 0.9])

    w = window(2020)
    train = usable[usable < w[0]]
    model = fit(X.loc[train], y.loc[train])
    pred = pd.Series(model.predict(X.loc[w]), index=w, name="forecast_mw")
    actual = y.loc[w]
    results["2020_target"] = {
        "lightgbm": score(actual, pred),
        "seasonal_naive_168h": score(actual, X.loc[w, "lag168"]),
        "last_year_364d": score(actual, X.loc[w, "lag364d"]),
    }
    results["interval"] = {"ratio_p10": float(lo_q), "ratio_p90": float(hi_q)}

    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    results["feature_importance_top"] = {k: float(v) for k, v in imp.head(8).items()}

    out = pd.DataFrame(
        {
            "utc_timestamp": w.tz_convert("UTC"),
            "forecast_mw": pred.round(0),
            "p10_mw": (pred * lo_q).round(0),
            "p90_mw": (pred * hi_q).round(0),
            "seasonal_naive_mw": X.loc[w, "lag168"].to_numpy(),
            "actual_mw": actual.to_numpy(),
        },
        index=w.rename("local_timestamp"),
    )
    out.to_csv(RUN / "forecast_jan2020.csv")
    (RUN / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

    daily = out.groupby(out.index.date)[["forecast_mw", "actual_mw"]].mean().round(0)
    daily["abs_pct_err"] = ((out["actual_mw"] - out["forecast_mw"]).abs() / out["actual_mw"]).groupby(out.index.date).mean().mul(100).round(1)
    print(daily.to_string())

    plot(out)


def plot(out: pd.DataFrame) -> None:
    surface, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    x = out.index
    ax.fill_between(x, out["p10_mw"], out["p90_mw"], color=orange, alpha=0.15, linewidth=0, label="Forecast 80% band")
    ax.plot(x, out["seasonal_naive_mw"], color=aqua, linewidth=1.6, label="Seasonal naive (previous week)")
    ax.plot(x, out["forecast_mw"], color=orange, linewidth=2, label="LightGBM forecast")
    ax.plot(x, out["actual_mw"], color=blue, linewidth=2, label="Actual load")
    names = {"actual_mw": "Actual", "forecast_mw": "LightGBM", "seasonal_naive_mw": "Naive"}
    ends = sorted((float(out[s].iloc[-1]), s) for s in names)
    # Push each end label at least 2.2 GW above the one below so they never overlap.
    placed: list[float] = []
    for value, series in ends:
        pos = value if not placed else max(value, placed[-1] + 2200)
        placed.append(pos)
        ax.annotate(names[series], (x[-1], pos), xytext=(6, 0), textcoords="offset points",
                    color=ink2, fontsize=9, va="center")
    ax.set_title("Germany hourly electricity load, 1 to 7 January 2020 (local time)", color=ink, fontsize=13, loc="left")
    ax.set_ylabel("Load (MW)", color=ink2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f} GW"))
    ax.xaxis.set_major_locator(mdates.DayLocator(tz=x.tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %-d Jan", tz=x.tz))
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c3")
    ax.tick_params(colors=ink2, length=0)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=ink2)
    ax.set_xlim(x[0], x[-1] + pd.Timedelta(hours=14))
    fig.tight_layout()
    fig.savefig(RUN / "forecast_jan2020.png", facecolor=surface)


if __name__ == "__main__":
    main()
