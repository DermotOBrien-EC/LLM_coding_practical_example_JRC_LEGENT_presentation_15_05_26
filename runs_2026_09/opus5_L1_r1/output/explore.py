from __future__ import annotations
import pandas as pd, numpy as np

df = pd.read_csv("opsd_de_load.csv", parse_dates=["utc_timestamp"])
s = df.set_index("utc_timestamp")["DE_load_actual_entsoe_transparency"]
s.index = s.index.tz_convert("UTC")
print("rows:", len(s), "range:", s.index.min(), "->", s.index.max())
full = pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC")
print("expected hours:", len(full), "missing timestamps:", len(full.difference(s.index)))
print("NaNs:", int(s.isna().sum()))
nan_idx = s[s.isna()].index
if len(nan_idx):
    print("NaN blocks:")
    grp = (nan_idx.to_series().diff() != pd.Timedelta("1h")).cumsum()
    for _, g in nan_idx.to_series().groupby(grp):
        print("  ", g.index[0], "->", g.index[-1], len(g), "h")
print("\nyearly stats (MW):")
print(s.groupby(s.index.year).agg(["count", "mean", "min", "max"]).round(0))
print("\nJan 1-7 window per year (Europe/Berlin local):")
loc = s.tz_convert("Europe/Berlin")
for y in range(2015, 2021):
    w = loc[f"{y}-01-01":f"{y}-01-07"]
    print(f"  {y}: n={len(w)} mean={w.mean():.0f} min={w.min():.0f} max={w.max():.0f} nan={int(w.isna().sum())}")
