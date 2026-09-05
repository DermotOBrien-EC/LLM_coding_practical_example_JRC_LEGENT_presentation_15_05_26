Plan:

1. Create `forecast_load.py`.
2. Read `opsd_de_load.csv` with pandas, parse UTC timestamps, sort, and clean missing load values.
3. Split data:
   - train: all rows before `2020-01-01 00:00:00+00:00`
   - test: 168 hours from `2020-01-01 00:00:00+00:00` through `2020-01-07 23:00:00+00:00`
4. Forecast with a transparent historical baseline:
   - use historical mean load by `(month, day_of_week, hour)`
   - fall back to `(day_of_week, hour)`, then `hour`, then global mean if needed
   - add a simple recent-level adjustment from late 2019 versus the long-run seasonal baseline
5. Compute MAE, RMSE, MAPE, and print them.
6. Plot actual versus forecast with matplotlib and save `german_load_forecast_jan2020.png`.
7. Run the script and report the accuracy and output files.

Approve this plan and I’ll implement it.