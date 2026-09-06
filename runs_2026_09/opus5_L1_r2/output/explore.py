from __future__ import annotations
import pandas as pd

s = pd.read_csv("opsd_de_load.csv", parse_dates=["utc_timestamp"], index_col="utc_timestamp")[
    "DE_load_actual_entsoe_transparency"]
loc = s.tz_convert("Europe/Berlin")

# Daily mean load around new year, each year
for y in range(2015, 2021):
    wk = loc.loc[f"{y}-01-01":f"{y}-01-07"].resample("D").mean()
    ref = loc.loc[f"{y-1}-11-01":f"{y-1}-11-30"].mean() if y > 2015 else float("nan")
    print(f"\n{y} Jan1-7 daily mean (ref Nov{y-1} mean={ref:,.0f}):")
    for d, v in wk.items():
        print(f"  {d.date()} {d.day_name()[:3]}  {v:8,.0f}" + (f"   ratio={v/ref:.3f}" if ref == ref else ""))

print("\n--- Christmas week (Dec25-31) vs first Jan week, daily means ---")
for y in range(2015, 2020):
    xmas = loc.loc[f"{y}-12-25":f"{y}-12-31"].mean()
    jan = loc.loc[f"{y+1}-01-01":f"{y+1}-01-07"].mean()
    nov = loc.loc[f"{y}-11-01":f"{y}-11-30"].mean()
    print(f"  Dec{y}: xmas={xmas:,.0f}  Jan{y+1}={jan:,.0f}  ratio Jan/xmas={jan/xmas:.3f}  Jan/Nov={jan/nov:.3f}")
