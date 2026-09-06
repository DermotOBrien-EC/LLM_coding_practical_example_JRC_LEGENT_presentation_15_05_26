from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

CSV = Path(__file__).parent / "opsd_de_load.csv"
OUT_DIR = Path(__file__).parent

df = pd.read_csv(CSV, parse_dates=["utc_timestamp"])
df = df.rename(columns={"DE_load_actual_entsoe_transparency": "load"}).dropna()
df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True)
df = df.set_index("utc_timestamp").sort_index()

train = df.loc[df.index.year <= 2019].copy()
target_start = pd.Timestamp("2020-01-01 00:00", tz="UTC")
target_end = pd.Timestamp("2020-01-07 23:00", tz="UTC")
target_idx = pd.date_range(target_start, target_end, freq="H")

# Feature: DOY (day-of-year, 1..366) x hour-of-day x weekday.
# Strategy: for each target hour, average the load at the same
# (day-of-year, hour) from all training years, with additional weight
# on the same weekday to preserve weekly rhythm.

train["doy"] = train.index.dayofyear
train["hour"] = train.index.hour
train["dow"] = train.index.dayofweek
train["year"] = train.index.year

# Level trend: annualise per-year mean and detrend before averaging,
# then re-inflate to 2020 level using linear extrapolation.
yearly_mean = train.groupby("year")["load"].mean()
years = yearly_mean.index.values.astype(float)
slope, intercept = np.polyfit(years, yearly_mean.values, 1)
mean_2020 = slope * 2020.0 + intercept
train = train.join(yearly_mean.rename("year_mean"), on="year")
train["load_norm"] = train["load"] / train["year_mean"]

rows = []
for ts in target_idx:
    doy, hour, dow = ts.dayofyear, ts.hour, ts.dayofweek
    m = train[(train["doy"] == doy) & (train["hour"] == hour)]
    if m.empty:
        rows.append(np.nan)
        continue
    weights = np.where(m["dow"].values == dow, 2.0, 1.0)
    norm = np.average(m["load_norm"].values, weights=weights)
    rows.append(norm * mean_2020)

forecast = pd.Series(rows, index=target_idx, name="forecast_MW")

actual = df.loc[target_start:target_end, "load"].rename("actual_MW")
result = pd.concat([forecast, actual], axis=1)
result["error_MW"] = result["forecast_MW"] - result["actual_MW"]
result["abs_pct_err"] = (result["error_MW"].abs() / result["actual_MW"]) * 100

result.to_csv(OUT_DIR / "forecast_jan_week_2020.csv", index_label="utc_timestamp")

mape = result["abs_pct_err"].mean()
rmse = np.sqrt((result["error_MW"] ** 2).mean())
peak_actual = result["actual_MW"].max()
peak_forecast = result["forecast_MW"].max()
trough_actual = result["actual_MW"].min()
trough_forecast = result["forecast_MW"].min()

daily = result.resample("D").agg(
    forecast_MWh=("forecast_MW", "sum"),
    actual_MWh=("actual_MW", "sum"),
)
daily["err_pct"] = (daily["forecast_MWh"] - daily["actual_MWh"]) / daily["actual_MWh"] * 100

print(f"Training years:        2015 to 2019")
print(f"Level extrapolation:   mean 2020 estimated at {mean_2020:,.0f} MW")
print(f"                       (per-year linear slope {slope:+,.0f} MW/yr)")
print()
print(f"Forecast horizon:      2020-01-01 00:00Z .. 2020-01-07 23:00Z (168 h)")
print(f"MAPE:                  {mape:.2f}%")
print(f"RMSE:                  {rmse:,.0f} MW")
print(f"Peak load, actual:     {peak_actual:,.0f} MW at {result['actual_MW'].idxmax()}")
print(f"Peak load, forecast:   {peak_forecast:,.0f} MW at {result['forecast_MW'].idxmax()}")
print(f"Trough, actual:        {trough_actual:,.0f} MW at {result['actual_MW'].idxmin()}")
print(f"Trough, forecast:      {trough_forecast:,.0f} MW at {result['forecast_MW'].idxmin()}")
print()
print("Daily totals (MWh):")
print(daily.round(0).to_string())
