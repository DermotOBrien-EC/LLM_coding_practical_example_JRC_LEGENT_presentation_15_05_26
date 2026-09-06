from __future__ import annotations
import pandas as pd

df = pd.read_csv("opsd_de_load.csv", parse_dates=["utc_timestamp"])
s = df.set_index("utc_timestamp")["DE_load_actual_entsoe_transparency"]
s.index = pd.DatetimeIndex(s.index)
print("range:", s.index.min(), "->", s.index.max(), "n =", len(s))
print("unique index:", s.index.is_unique, "monotonic:", s.index.is_monotonic_increasing)

full = pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC")
print("expected hours:", len(full), "missing timestamps:", len(full.difference(s.index)))
miss = full.difference(s.index)
if len(miss):
    print(miss[:20])
print("NaN count:", int(s.isna().sum()))
nan_idx = s[s.isna()].index
if len(nan_idx):
    g = pd.Series(nan_idx).dt.to_period("M").value_counts().sort_index()
    print("NaNs by month:\n", g)
    print("first/last NaN:", nan_idx.min(), nan_idx.max())
print("\ndescribe:\n", s.describe())
print("\nzeros/negatives:", int((s <= 0).sum()))

# target week
tgt = s.loc["2020-01-01":"2020-01-07 23:00"]
print("\ntarget week hours:", len(tgt), "NaN:", int(tgt.isna().sum()))
print("train tail:", s.loc[:"2019-12-31 23:00"].index.max(), "n_train =", len(s.loc[:"2019-12-31 23:00"]))
print("\n2019-12-20..2020-01-10 NaNs:", int(s.loc["2019-12-20":"2020-01-10"].isna().sum()))
