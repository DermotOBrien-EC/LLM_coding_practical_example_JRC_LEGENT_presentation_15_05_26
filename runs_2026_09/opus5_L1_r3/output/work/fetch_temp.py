"""Population-weighted German 2m temperature from the Open-Meteo ERA5 archive."""
from __future__ import annotations
import json, sys, urllib.request
import numpy as np, pandas as pd

CITIES = [  # name, lat, lon, population weight (millions, city proper)
    ("Berlin", 52.52, 13.41, 3.68), ("Hamburg", 53.55, 9.99, 1.85),
    ("Muenchen", 48.14, 11.58, 1.47), ("Koeln", 50.94, 6.96, 1.09),
    ("Frankfurt", 50.11, 8.68, 0.76), ("Stuttgart", 48.78, 9.18, 0.63),
    ("Duesseldorf", 51.23, 6.78, 0.62), ("Leipzig", 51.34, 12.37, 0.59),
    ("Dortmund", 51.51, 7.47, 0.59), ("Essen", 51.46, 7.01, 0.58),
    ("Bremen", 53.08, 8.80, 0.57), ("Dresden", 51.05, 13.74, 0.56),
    ("Hannover", 52.37, 9.73, 0.54), ("Nuernberg", 49.45, 11.08, 0.52),
]
START, END = "2015-01-01", "2020-09-30"

url = ("https://archive-api.open-meteo.com/v1/archive"
       f"?latitude={','.join(str(c[1]) for c in CITIES)}"
       f"&longitude={','.join(str(c[2]) for c in CITIES)}"
       f"&start_date={START}&end_date={END}&hourly=temperature_2m&timezone=UTC")
print("requesting", len(CITIES), "locations", START, "->", END)
with urllib.request.urlopen(url, timeout=300) as r:
    payload = json.loads(r.read().decode())
if isinstance(payload, dict):
    payload = [payload]
assert len(payload) == len(CITIES), f"expected {len(CITIES)} blocks, got {len(payload)}"

frames = {}
for (name, lat, lon, _w), blk in zip(CITIES, payload):
    idx = pd.to_datetime(blk["hourly"]["time"], utc=True)
    frames[name] = pd.Series(blk["hourly"]["temperature_2m"], index=idx, dtype=float)
    assert abs(blk["latitude"] - lat) < 0.6 and abs(blk["longitude"] - lon) < 0.6, name
df = pd.DataFrame(frames)
print("shape", df.shape, "NaN", int(df.isna().sum().sum()))
df = df.interpolate(limit=6).ffill().bfill()
w = np.array([c[3] for c in CITIES]); w = w / w.sum()
out = pd.DataFrame({"temp_C": (df.to_numpy() * w).sum(axis=1)}, index=df.index)
out.index.name = "utc_timestamp"
out.to_csv("work/temp_de.csv")
print(out.describe().round(2).to_string())
print("\nJan 1-7 daily mean temp by year:")
loc = out["temp_C"]
for y in range(2015, 2021):
    v = loc[f"{y}-01-01":f"{y}-01-07"]
    print(f"  {y}: {v.mean():5.2f} C  (min {v.min():5.1f}, max {v.max():5.1f})")
