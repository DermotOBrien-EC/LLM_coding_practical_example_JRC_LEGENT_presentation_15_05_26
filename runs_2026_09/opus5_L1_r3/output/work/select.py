"""Model selection on 2017-2019 winter origins. The 2020 week is never read here.

Rule, declared before scoring: pick the configuration with the lowest mean MAPE
across all 18 winter origins.
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "work")
import loadfc as L
from backtest import load_series, load_weather

s, wx = load_series(), load_weather()

def week_index(st):
    return pd.date_range(pd.Timestamp(st, tz=L.TZ), periods=168, freq="h").tz_convert("UTC")

def predict(tgt, weather, window, alpha, lc, run_feat):
    train = s.loc[: tgt[0] - pd.Timedelta(hours=1)]
    if window is not None:
        anchor = pd.Timestamp(tgt[84].tz_convert(L.TZ).date())
        train = train[L.season_distance(train.index, anchor) <= window]
    w = wx if weather else None
    f_tr = L.build_features(train.index, w); f_tg = L.build_features(tgt, w)
    kw = dict(toy_block=True, weather=weather, work_run_feat=run_feat)
    m, cols = L.fit_ridge(train, f_tr, alpha=alpha, **kw)
    p = L.predict_ridge(m, cols, f_tg, **kw)
    if lc:
        p = p * L._level_correction(train, L.predict_ridge(m, cols, f_tr, **kw), tgt[0])
    return p

NY = [f"{y}-01-01" for y in (2017, 2018, 2019)]
OTHER = [f"{y}-{md}" for y in (2017, 2018, 2019)
         for md in ("01-15", "01-29", "02-12", "11-19", "12-03")]
CANDS = {
    "cal_full":        dict(weather=False, window=None, alpha=1.0,  lc=True,  run_feat=False),
    "wx_full":         dict(weather=True,  window=None, alpha=1.0,  lc=True,  run_feat=False),
    "wx_full_a3":      dict(weather=True,  window=None, alpha=3.0,  lc=True,  run_feat=False),
    "wx_full_nolc":    dict(weather=True,  window=None, alpha=1.0,  lc=False, run_feat=False),
    "wx_full_run":     dict(weather=True,  window=None, alpha=1.0,  lc=True,  run_feat=True),
    "wx_w30_a0.3":     dict(weather=True,  window=30,   alpha=0.3,  lc=False, run_feat=False),
    "wx_w45_a1":       dict(weather=True,  window=45,   alpha=1.0,  lc=False, run_feat=False),
    "wx_w60_a1":       dict(weather=True,  window=60,   alpha=1.0,  lc=False, run_feat=False),
    "wx_w90_a1":       dict(weather=True,  window=90,   alpha=1.0,  lc=False, run_feat=False),
}
rows = []
for grp, starts in (("NewYear", NY), ("OtherWinter", OTHER)):
    for st in starts:
        tgt = week_index(st); act = s.reindex(tgt)
        rec = {"group": grp, "start": st}
        for name, cfg in CANDS.items():
            m = L.metrics(act, predict(tgt, **cfg))
            assert m["n_hours"] == 168, (st, name, m["n_hours"])
            rec[name] = m["MAPE_%"]
        rows.append(rec); print("done", st, flush=True)
df = pd.DataFrame(rows); df.to_csv("out/selection_2017_2019.csv", index=False)
pd.set_option("display.width", 220)
num = df.drop(columns=["group", "start"])
summ = pd.DataFrame({"mean_all18": num.mean(), "worst": num.max(),
                     "mean_NewYear": df[df.group == "NewYear"][num.columns].mean(),
                     "mean_OtherWinter": df[df.group == "OtherWinter"][num.columns].mean()})
summ = summ.sort_values("mean_all18")
print("\n=== selection table (2017-2019 only; MAPE %) ===")
print(summ.round(3).to_string())
print(f"\nRULE: lowest mean over all 18 origins -> WINNER = {summ.index[0]}")
print("config:", CANDS[summ.index[0]])
summ.to_csv("out/selection_summary.csv")
