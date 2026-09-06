from __future__ import annotations
import pandas as pd
import numpy as np

df = pd.read_csv("opsd_de_load.csv", parse_dates=["utc_timestamp"])
df = df.rename(columns={"DE_load_actual_entsoe_transparency": "load"})
df = df.set_index("utc_timestamp").sort_index()

print("range:", df.index.min(), "->", df.index.max())
print("rows:", len(df))
# full hourly index
full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
print("expected hourly rows:", len(full))
missing_ts = full.difference(df.index)
print("missing timestamps:", len(missing_ts))
print("NaN loads:", int(df["load"].isna().sum()))
# show any NaNs near the forecast window and overall
print("\nNaN by year:")
print(df["load"].isna().groupby(df.index.year).sum())

# Focus: train cutoff = 2020-01-01 00:00 UTC. Forecast Jan 1-7 2020 (168 hrs).
train = df.loc[:"2019-12-31 23:00"]
print("\ntrain rows:", len(train), "train NaNs:", int(train["load"].isna().sum()))
target = df.loc["2020-01-01 00:00":"2020-01-07 23:00"]
print("target rows:", len(target), "target NaNs:", int(target["load"].isna().sum()))
print("\ntarget head/tail:")
print(target.head(3))
print(target.tail(3))
print("\nlast train values:")
print(train.tail(3))
