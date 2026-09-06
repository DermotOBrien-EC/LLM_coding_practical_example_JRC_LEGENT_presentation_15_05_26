from __future__ import annotations
import sys; sys.path.insert(0, ".")
from pathlib import Path
import numpy as np, pandas as pd
from forecast.data import load_series
from forecast.evaluate import metrics, january_windows
from forecast.models import (HolidayAdjustedNaive, LightGBMModel, LinearProfileModel,
                             SeasonalNaive, YearAgoNaive)

pd.set_option("display.width", 220)
s = load_series(Path("opsd_de_load.csv"))
bases = {"seasonal_naive_168h": SeasonalNaive, "year_ago_naive_364d": YearAgoNaive,
         "holiday_adjusted_naive": HolidayAdjustedNaive,
         "linear_profile": LinearProfileModel, "lightgbm": LightGBMModel}

preds: dict[pd.Timestamp, pd.DataFrame] = {}
actuals: dict[pd.Timestamp, pd.Series] = {}
for w in january_windows([2016, 2017, 2018, 2019]):
    hist = s.loc[: w.start - pd.Timedelta(hours=1)]
    cols = {name: f().fit(hist).predict(w.index) for name, f in bases.items()}
    preds[w.start] = pd.DataFrame(cols)
    actuals[w.start] = s.reindex(w.index)
    print(f"fitted fold {w.start.date()}", flush=True)

def geo(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    tot = sum(weights.values())
    acc = sum(w * np.log(df[k].to_numpy()) for k, w in weights.items())
    return pd.Series(np.exp(acc / tot), index=df.index)

combos = {
    "lightgbm": {"lightgbm": 1.0},
    "linear_profile": {"linear_profile": 1.0},
    "year_ago_naive_364d": {"year_ago_naive_364d": 1.0},
    "ens_lgb+lin_50_50": {"lightgbm": .5, "linear_profile": .5},
    "ens_lgb+lin_70_30": {"lightgbm": .7, "linear_profile": .3},
    "ens_lgb+lin+yoy_50_25_25": {"lightgbm": .5, "linear_profile": .25, "year_ago_naive_364d": .25},
    "ens_lgb+lin+yoy_60_20_20": {"lightgbm": .6, "linear_profile": .2, "year_ago_naive_364d": .2},
}
rows = []
for start, df in preds.items():
    for label, wts in combos.items():
        rows.append({"window": start.date(), "model": label, **metrics(actuals[start], geo(df, wts))})
res = pd.DataFrame(rows)
for title, sub in [("all 4 folds", res),
                   ("excl. short-history 2016 fold", res[res.window.astype(str) != "2016-01-01"])]:
    piv = sub.pivot(index="model", columns="window", values="MAPE")
    piv["mean"] = piv.mean(axis=1); piv["worst"] = piv.iloc[:, :-1].max(axis=1)
    print(f"\n=== MAPE % — {title} ===")
    print(piv.sort_values("mean").round(2).to_string())
res.to_csv("outputs/validation_ensembles.csv", index=False)
