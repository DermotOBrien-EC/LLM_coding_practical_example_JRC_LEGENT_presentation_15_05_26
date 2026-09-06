import pandas as pd, numpy as np
df = pd.read_csv("opsd_de_load.csv")
s = pd.Series(df["DE_load_actual_entsoe_transparency"].values,
              index=pd.to_datetime(df["utc_timestamp"], utc=True)).sort_index()
loc = s.tz_convert("Europe/Berlin")
d = loc.resample("D").mean()

print("=== Daily mean load, Dec 20 -> Jan 10, by season ===")
for y in range(2015, 2020):
    win = d[f"{y}-12-20":f"{y+1}-01-10"]
    if len(win) == 0: continue
    nov = d[f"{y}-11-05":f"{y}-11-25"]
    base = nov[nov.index.dayofweek < 5].mean()   # normal November weekdays
    print(f"\nseason {y}/{y+1}  Nov-weekday baseline = {base:,.0f} MW")
    for ts, v in win.items():
        print(f"  {ts.date()} {ts.day_name()[:3]}  {v:8,.0f}  {100*v/base:5.1f}% of baseline")
