from __future__ import annotations

import pandas as pd

TZ = "Europe/Berlin"
df = pd.read_csv("opsd_de_load.csv", parse_dates=["utc_timestamp"])
s = df.set_index("utc_timestamp")["DE_load_actual_entsoe_transparency"]
s.index = s.index.tz_convert(TZ)

def wk(y: int) -> pd.Series:
    return s.loc[f"{y}-01-01":f"{y}-01-07"].resample("D").mean().round(0)

print("Daily mean load, 1-7 Jan, by year (with weekday):")
tab = pd.DataFrame({y: wk(y).to_numpy() for y in range(2016, 2021)}, index=[f"Jan {d}" for d in range(1, 8)])
print(tab.to_string())
for y in range(2016, 2021):
    print(y, " ".join(pd.Timestamp(f"{y}-01-0{d}").strftime("%a") for d in range(1, 8)))

print("\nMonthly mean load, Nov/Dec/Jan/Feb around each year end:")
m = s.resample("MS").mean().round(0)
for y in range(2016, 2021):
    sel = m.loc[f"{y-1}-11":f"{y}-02"]
    print(y, sel.to_numpy())

print("\nRatio of 1-7 Jan mean to the preceding 28 days (Dec 4-31) mean:")
for y in range(2016, 2021):
    pre = s.loc[f"{y-1}-12-04":f"{y-1}-12-31"].mean()
    jan = s.loc[f"{y}-01-01":f"{y}-01-07"].mean()
    print(y, f"pre28={pre:.0f} jan_wk={jan:.0f} ratio={jan/pre:.3f}")

print("\nRatio of 1-7 Jan mean to the SAME week previous year, vs Dec-28d ratio y/y:")
for y in range(2017, 2021):
    jan = s.loc[f"{y}-01-01":f"{y}-01-07"].mean() / s.loc[f"{y-1}-01-01":f"{y-1}-01-07"].mean()
    dec = s.loc[f"{y-1}-12-04":f"{y-1}-12-31"].mean() / s.loc[f"{y-2}-12-04":f"{y-2}-12-31"].mean()
    full = s.loc[f"{y-1}-01-08":f"{y-1}-12-31"].mean() / s.loc[f"{y-2}-01-08":f"{y-2}-12-31"].mean()
    print(y, f"jan_wk y/y={jan:.3f}  dec28 y/y={dec:.3f}  prior-year y/y={full:.3f}")

f = pd.read_csv("forecast_jan2020_week1.csv", index_col=0)
print("\nHourly bias (lightgbm - actual) by day, MW:")
f.index = pd.to_datetime(f.index, utc=True).tz_convert(TZ)
print((f["lightgbm"] - f["actual"]).resample("D").mean().round(0).to_string())
