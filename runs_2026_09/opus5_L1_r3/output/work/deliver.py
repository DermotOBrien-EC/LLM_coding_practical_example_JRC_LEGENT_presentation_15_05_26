"""Produce the final forecast artifacts for 1-7 January 2020."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "work")
import loadfc as L
from backtest import load_series, load_weather, naive_168, analog_years

s, wx = load_series(), load_weather()
TZ = L.TZ

# Primary: winner of the declared rule "lowest mean MAPE over all 18 winter
# origins", evaluated entirely on 2017-2019 (out/selection_summary.csv). It also
# has the best worst-case there.
PRIMARY = dict(weather=True, window=30, alpha=0.3, lc=False, run_feat=False)
# Runner-up under the same rule: full history, no season window.
ALT = dict(weather=True, window=None, alpha=1.0, lc=True, run_feat=False)
# Post-hoc: adds the working-run feature, motivated by inspecting the 2020 miss.
POSTHOC = dict(weather=True, window=None, alpha=1.0, lc=True, run_feat=True)
POSTHOC_W = dict(weather=True, window=30, alpha=0.3, lc=False, run_feat=True)


def week_index(start_local: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start_local, tz=TZ), periods=168,
                         freq="h").tz_convert("UTC")




def predict(tgt, weather, window, alpha, lc, run_feat, wx_override=None):
    train = s.loc[: tgt[0] - pd.Timedelta(hours=1)]
    if window is not None:
        anchor = pd.Timestamp(tgt[84].tz_convert(TZ).date())
        train = train[L.season_distance(train.index, anchor) <= window]
    w = (wx_override if wx_override is not None else wx) if weather else None
    f_tr = L.build_features(train.index, w)
    f_tg = L.build_features(tgt, w)
    kw = dict(toy_block=True, weather=weather, work_run_feat=run_feat)
    m, cols = L.fit_ridge(train, f_tr, alpha=alpha, **kw)
    p = L.predict_ridge(m, cols, f_tg, **kw)
    if lc:
        p = p * L._level_correction(train, L.predict_ridge(m, cols, f_tr, **kw), tgt[0])
    return p


def horizon_day(tgt): return ((tgt - tgt[0]).total_seconds() // 86400).astype(int) + 1


NY = [f"{y}-01-01" for y in (2017, 2018, 2019)]
WINTER = NY + [f"{y}-{md}" for y in (2017, 2018, 2019)
               for md in ("01-15", "01-29", "02-12", "11-19", "12-03")]


def rel_error_quantiles(starts, cfg):
    recs = []
    for st in starts:
        t = week_index(st)
        p = predict(t, **cfg)
        a = s.reindex(t)
        recs.append(pd.DataFrame({"day": horizon_day(t),
                                  "rel": (a.to_numpy() - p.to_numpy()) / p.to_numpy()}))
    q = pd.concat(recs).groupby("day")["rel"].quantile([0.1, 0.9]).unstack()
    q.columns = ["q10", "q90"]
    return q


def weather_sensitivity(tgt, cfg, n=60, seed=7):
    rng = np.random.default_rng(seed)
    sd = 0.65 + 0.26 * horizon_day(tgt)
    act = s.reindex(tgt)
    out = []
    for _ in range(n):
        e = np.zeros(len(tgt))
        sh = rng.normal(0, 1, len(tgt))
        for i in range(1, len(tgt)):
            e[i] = 0.97 * e[i - 1] + sh[i]
        e = e / max(e.std(), 1e-9) * sd
        pert = wx["temp"].copy()
        pert.loc[tgt] = pert.loc[tgt].to_numpy() + e
        out.append(L.metrics(act, predict(tgt, wx_override=L.weather_features(pert), **cfg)))
    return pd.DataFrame(out)


def main() -> None:
    tgt = week_index("2020-01-01")
    actual = s.reindex(tgt)
    loc = tgt.tz_convert(TZ)
    hd = horizon_day(tgt)

    preds = {
        "primary (selected)": predict(tgt, **PRIMARY),
        "alt full-history": predict(tgt, **ALT),
        "posthoc_runfeat": predict(tgt, **POSTHOC),
        "posthoc_runfeat_w30": predict(tgt, **POSTHOC_W),
        "naive_lag168": naive_168(s, tgt),
        "analog_3y": analog_years(s, tgt),
    }
    q_ny = rel_error_quantiles(NY, PRIMARY)
    q_w = rel_error_quantiles(WINTER, PRIMARY)
    lo = preds["primary (selected)"].to_numpy() * (1 + q_ny["q10"].reindex(hd).to_numpy())
    hi = preds["primary (selected)"].to_numpy() * (1 + q_ny["q90"].reindex(hd).to_numpy())
    lo_w = preds["primary (selected)"].to_numpy() * (1 + q_w["q10"].reindex(hd).to_numpy())
    hi_w = preds["primary (selected)"].to_numpy() * (1 + q_w["q90"].reindex(hd).to_numpy())

    out = pd.DataFrame({
        "local_time_berlin": loc.strftime("%Y-%m-%d %H:%M"),
        "utc_timestamp": tgt.strftime("%Y-%m-%d %H:%M"),
        "horizon_day": hd,
        "forecast_MW": preds["primary (selected)"].to_numpy().round(1),
        "pi80_low_MW": lo.round(1),
        "pi80_high_MW": hi.round(1),
        "forecast_alt_full_history_MW": preds["alt full-history"].to_numpy().round(1),
        "forecast_posthoc_runfeat_MW": preds["posthoc_runfeat"].to_numpy().round(1),
        "actual_MW": actual.to_numpy(),
        "temp_C": wx.loc[tgt, "temp"].to_numpy().round(2),
    })
    out["error_MW"] = (out["forecast_MW"] - out["actual_MW"]).round(1)
    out["abs_pct_error"] = (100 * (out["error_MW"] / out["actual_MW"]).abs()).round(2)
    out.to_csv("out/forecast_2020_week1.csv", index=False)

    scores = pd.DataFrame({k: L.metrics(actual, v) for k, v in preds.items()}).T
    cov = float(((actual.to_numpy() >= lo) & (actual.to_numpy() <= hi)).mean() * 100)
    cov_w = float(((actual.to_numpy() >= lo_w) & (actual.to_numpy() <= hi_w)).mean() * 100)

    daily = out.assign(d=pd.DatetimeIndex(loc).date).groupby("d").agg(
        forecast=("forecast_MW", "mean"), alt=("forecast_alt_full_history_MW", "mean"),
        posthoc=("forecast_posthoc_runfeat_MW", "mean"),
        actual=("actual_MW", "mean"), MAPE=("abs_pct_error", "mean"), temp=("temp_C", "mean"))
    daily["bias"] = (daily["forecast"] - daily["actual"]).round(0)
    daily["dayname"] = [pd.Timestamp(d).strftime("%a") for d in daily.index]

    sens = weather_sensitivity(tgt, PRIMARY)

    print("=== 1-7 January 2020, hourly (Europe/Berlin), 168 h ===")
    print(f"training history : 2015-01-01 -> 2019-12-31 (43823 h), no data at or after the origin")
    print(f"forecast total   : {preds['primary (selected)'].sum()/1000:,.0f} GWh   "
          f"actual {actual.sum()/1000:,.0f} GWh")
    print(f"forecast peak    : {preds['primary (selected)'].max():,.0f} MW at "
          f"{loc[int(np.argmax(preds['primary (selected)'].to_numpy()))]:%a %d %b %H:%M}   "
          f"actual peak {actual.max():,.0f} MW at "
          f"{loc[int(np.argmax(actual.to_numpy()))]:%a %d %b %H:%M}")
    print("\n=== Accuracy on the held-out week ===")
    print(scores.round(3).to_string())
    print(f"\n80% PI coverage: {cov:.1f}% (New Year calibration, n=3) / "
          f"{cov_w:.1f}% (all winter weeks, n=18); nominal 80%")
    print("\n=== Daily (Europe/Berlin) ===")
    print(daily.round(1).to_string())
    print("\n=== Weather-forecast-error sensitivity, 60 draws (primary model) ===")
    print(sens[["MAPE_%", "MAE_MW", "bias_MW"]].describe()
          .loc[["mean", "std", "min", "max"]].round(3).to_string())

    json.dump({"primary_config": PRIMARY, "posthoc_config": POSTHOC, "alt_config": ALT, "posthoc_w30_config": POSTHOC_W,
               "metrics_2020": scores.to_dict(orient="index"),
               "pi80_coverage_pct": {"newyear_cal": cov, "winter_cal": cov_w},
               "weather_sensitivity_MAPE": sens["MAPE_%"].describe().to_dict(),
               "forecast_total_GWh": float(preds["primary (selected)"].sum() / 1000),
               "actual_total_GWh": float(actual.sum() / 1000)},
              open("out/summary.json", "w"), indent=2, default=float)
    pd.to_pickle({"preds": pd.DataFrame(preds), "lo": lo, "hi": hi,
                  "actual": actual, "loc": loc}, "work/plot_data.pkl")


if __name__ == "__main__":
    main()
