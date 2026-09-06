Built and ran the forecasting script.

Files created:

- `forecast_de_load.py`
- `forecast_de_load_jan2020.csv`
- `forecast_de_load_jan2020.png`

What it does:

- Reads `opsd_de_load.csv`
- Trains only on data before `2020-01-01 00:00 UTC`
- Forecasts the 168 hourly values from `2020-01-01 00:00 UTC` through `2020-01-07 23:00 UTC`
- Uses a ridge regression model with calendar features, seasonal terms, and lagged load features
- Saves the forecast CSV
- Plots actual versus forecast using matplotlib

Accuracy on the requested week:

```text
MAE:  2,293.4 MW
RMSE: 2,837.1 MW
MAPE: 4.69%
```

Verification run:

```text
python3 forecast_de_load.py
```

completed successfully and produced 168 forecast rows plus the PNG plot.

Assumptions made:

- The requested period means the 168 hours from `2020-01-01 00:00 UTC` inclusive to `2020-01-08 00:00 UTC` exclusive.
- “German hourly electricity load” refers to `DE_load_actual_entsoe_transparency`.
- Because this environment initially lacked pandas and matplotlib globally, I installed them into a local `.python_packages/` directory and made the script load that directory automatically if present.