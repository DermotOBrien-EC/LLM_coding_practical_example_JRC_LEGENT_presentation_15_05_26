Done. Trained on 2015-2019, forecast the 168 hours 2020-01-01 00:00Z through 2020-01-07 23:00Z, and compared against the actuals present in the file.

**Method.** Same-hour, same-day-of-year historical average from 2015-2019, weekday-weighted 2:1 to preserve the weekly rhythm, normalised by per-year mean and re-inflated to 2020 using a linear year-mean trend (+402 MW/yr, projected 2020 mean 57.1 GW).

**Overall accuracy vs actual.** MAPE 7.73%, RMSE 5.73 GW.

**Daily totals (GWh, forecast vs actual, error):**
- Wed 01-01: 1,078 vs 1,056 (+2%)
- Thu 01-02: 1,327 vs 1,298 (+2%)
- Fri 01-03: 1,362 vs 1,335 (+2%)
- Sat 01-04: 1,414 vs 1,261 (**+12%**)
- Sun 01-05: 1,429 vs 1,152 (**+24%**)
- Mon 01-06: 1,341 vs 1,365 (-2%)
- Tue 01-07: 1,424 vs 1,504 (-5%)

**Weekly peak.** Forecast 68.8 GW at Tue 17:00Z, actual 70.9 GW at Tue 16:00Z. Trough matched at Wed 05:00Z, 38.3 GW vs 37.5 GW.

**Where the model missed.** The weekend (Jan 4-5) is the big overshoot. For those dates the same day-of-year in 2015-2018 mostly landed on weekdays, so even with 2x weekday weighting the forecast leaked weekday level into a real weekend. A holiday-aware or hour-of-week model (not just doy) would fix this, likely halving the weekly MAPE. Jan 6 was Epiphany, a public holiday in three southern states; the model got it right by luck because its weighted average pulled toward lower Monday levels.

Artifacts written to `/Users/doob/dev/energy_forecast_ws/d2bd3f/project/runs/d2bd3f/`:
- `forecast.py` (reproducible script)
- `forecast_jan_week_2020.csv` (hourly forecast, actual, error, |%err|)