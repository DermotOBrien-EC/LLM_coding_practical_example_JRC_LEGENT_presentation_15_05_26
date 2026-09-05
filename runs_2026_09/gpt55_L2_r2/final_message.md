Plan:

1. Create `forecast_german_load.py`.
2. Read `opsd_de_load.csv`, using `utc_timestamp` as hourly UTC timestamps and `DE_load_actual_entsoe_transparency` as MW load.
3. Train only on rows before `2020-01-01 00:00:00+00:00`.
4. Forecast the 168 hours from `2020-01-01 00:00:00+00:00` through `2020-01-07 23:00:00+00:00`.
5. Use a simple, explainable baseline model:
   - same hour of week from the previous week,
   - adjusted by recent year-on-year residuals for the same hour of week.
6. Save:
   - `german_load_forecast_jan2020.csv`
   - `german_load_forecast_jan2020.png`
7. Report MAE, RMSE, MAPE, and peak error against the actual values.

One issue: this environment currently has `numpy` but not `pandas` or `matplotlib`, despite `AGENTS.md` asking for them. I’ll either install/use them in a local environment if available, or make the script fail with a clear dependency message and use a lightweight fallback only for my local verification. Approve this plan?