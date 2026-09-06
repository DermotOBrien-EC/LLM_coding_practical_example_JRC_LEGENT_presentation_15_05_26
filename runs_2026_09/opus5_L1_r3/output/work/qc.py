import pandas as pd, numpy as np
df = pd.read_csv("opsd_de_load.csv")
df["utc"] = pd.to_datetime(df["utc_timestamp"], utc=True)
s = df.set_index("utc")["DE_load_actual_entsoe_transparency"].sort_index()
print("rows", len(s), "range", s.index.min(), "->", s.index.max())
full = pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC")
print("expected hours", len(full), "missing index entries", len(full.difference(s.index)))
print("NaN values", int(s.isna().sum()))
print("duplicated ts", int(s.index.duplicated().sum()))
nan_idx = s[s.isna()].index
if len(nan_idx):
    g = pd.Series(nan_idx).dt.floor("D").value_counts().sort_index()
    print("NaN days:\n", g.to_string())
print(s.describe())
# per-year coverage
loc = s.tz_convert("Europe/Berlin")
print(loc.groupby(loc.index.year).agg(["count","mean","min","max"]))
