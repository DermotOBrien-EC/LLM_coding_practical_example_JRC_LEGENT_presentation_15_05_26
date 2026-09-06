"""Walk-forward backtest: train on everything before 1 Jan, predict Jan 1-7."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "work")
import loadfc as L


def load_series(path: str = "opsd_de_load.csv") -> pd.Series:
    df = pd.read_csv(path)
    s = pd.Series(df["DE_load_actual_entsoe_transparency"].to_numpy(dtype=float),
                  index=pd.to_datetime(df["utc_timestamp"], utc=True))
    return s.sort_index().rename("load_MW")


def load_weather(path: str = "work/temp_de.csv") -> pd.DataFrame:
    t = pd.read_csv(path, index_col=0)
    t.index = pd.to_datetime(t.index, utc=True)
    return L.weather_features(t["temp_C"].astype(float))


def target_index(year: int) -> pd.DatetimeIndex:
    """168 local hours: Jan 1 00:00 -> Jan 7 23:00 Europe/Berlin."""
    start = pd.Timestamp(f"{year}-01-01 00:00", tz=L.TZ)
    return pd.date_range(start, periods=168, freq="h").tz_convert("UTC")


def naive_168(s: pd.Series, tgt: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(s.reindex(tgt - pd.Timedelta(hours=168)).to_numpy(), index=tgt)


def analog_years(s: pd.Series, tgt: pd.DatetimeIndex, n_years: int = 3) -> pd.Series:
    """Same local calendar date+hour in the previous n years, level-rescaled."""
    loc = tgt.tz_convert(L.TZ)
    origin = tgt[0]
    frames = [s.reindex(pd.DatetimeIndex([t - pd.DateOffset(years=k) for t in loc])
                        .tz_convert("UTC")).to_numpy() for k in range(1, n_years + 1)]
    raw = np.nanmean(np.vstack(frames), axis=0)
    win = s.loc[origin - pd.Timedelta(days=56): origin - pd.Timedelta(hours=1)]
    clean = win[L.turn_of_year_offset(win.index.tz_convert(L.TZ)) == 99]
    ratios = []
    for k in range(1, n_years + 1):
        back = pd.DatetimeIndex([t - pd.DateOffset(years=k)
                                 for t in clean.index.tz_convert(L.TZ)]).tz_convert("UTC")
        prev = s.reindex(back)
        if prev.notna().sum() > 100:
            ratios.append(clean.mean() / prev.mean())
    return pd.Series(raw * (float(np.mean(ratios)) if ratios else 1.0), index=tgt)


def fit_predict(s: pd.Series, tgt: pd.DatetimeIndex, wx: pd.DataFrame | None,
                weather: bool, alpha: float = 1.0,
                toy_block: bool = True) -> dict[str, pd.Series]:
    """Ridge + GBM, each with a recent-window level correction applied."""
    origin = tgt[0]
    train = s.loc[: origin - pd.Timedelta(hours=1)]
    f_tr = L.build_features(train.index, wx)
    f_tg = L.build_features(tgt, wx)
    kw = dict(toy_block=toy_block, weather=weather)

    ridge, cols = L.fit_ridge(train, f_tr, alpha=alpha, **kw)
    k_r = L._level_correction(train, L.predict_ridge(ridge, cols, f_tr, **kw), origin)
    gbm = L.fit_gbm(train, f_tr, weather=weather)
    k_g = L._level_correction(train, L.predict_gbm(gbm, f_tr), origin)
    return {
        "ridge": L.predict_ridge(ridge, cols, f_tg, **kw) * k_r,
        "gbm": L.predict_gbm(gbm, f_tg) * k_g,
        "_k": pd.Series({"ridge": k_r, "gbm": k_g}),
    }


def run_origin(s: pd.Series, wx: pd.DataFrame, year: int,
               verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    tgt = target_index(year)
    preds: dict[str, pd.Series] = {
        "naive_lag168": naive_168(s, tgt),
        "analog_3y": analog_years(s, tgt),
    }
    cal = fit_predict(s, tgt, None, weather=False)
    wxp = fit_predict(s, tgt, wx, weather=True)
    preds["ridge_cal"] = cal["ridge"]
    preds["gbm_cal"] = cal["gbm"]
    preds["ens_cal"] = 0.5 * cal["ridge"] + 0.5 * cal["gbm"]
    preds["ridge_wx"] = wxp["ridge"]
    preds["gbm_wx"] = wxp["gbm"]
    preds["ens_wx"] = 0.5 * wxp["ridge"] + 0.5 * wxp["gbm"]

    actual = s.reindex(tgt)
    out = pd.DataFrame({k: L.metrics(actual, v) for k, v in preds.items()}).T
    if verbose:
        n = len(s.loc[: tgt[0] - pd.Timedelta(hours=1)])
        print(f"\n=== origin {year} | {n} training hours ===")
        print(out.sort_values("MAPE_%").round(3).to_string())
    frame = pd.DataFrame(preds)
    frame["actual"] = actual
    return frame, out


if __name__ == "__main__":
    s, wx = load_series(), load_weather()
    ms = []
    for y in (2016, 2017, 2018, 2019):
        _, m = run_origin(s, wx, y, verbose=True)
        m["year"] = y
        ms.append(m)
    cat = pd.concat(ms)
    print("\n=== mean over origins 2017-2019 (2016 has only 1 training year) ===")
    print(cat[cat["year"] >= 2017].groupby(level=0).mean(numeric_only=True)
          .drop(columns="year").sort_values("MAPE_%").round(3).to_string())
    cat.to_csv("out/backtest_metrics.csv")
