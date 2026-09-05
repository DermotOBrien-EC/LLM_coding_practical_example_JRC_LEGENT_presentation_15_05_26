Plan:

1. Create `forecast_load.py` in the current directory.
   - Read `opsd_de_load.csv`.
   - Parse `utc_timestamp` as UTC datetimes.
   - Use all rows before `2020-01-01 00:00:00+00:00` as training data.
   - Use `2020-01-01 00:00` through `2020-01-07 23:00` UTC as the 168-hour test window.

2. Forecast with a transparent baseline model.
   - Fit a linear regression on calendar features: hour of day, day of week, month, weekend flag, holiday flag, plus lagged load features.
   - Use recursive one-hour-ahead prediction so the test week is truly forecast from history, not from future actuals.

3. Produce outputs.
   - Write hourly predictions and actuals to `forecast_jan_2020.csv`.
   - Plot actual versus forecast with matplotlib to `forecast_jan_2020.png`.
   - Print MAE, RMSE, MAPE, and mean actual load.

4. Verify.
   - Run the script end-to-end.
   - Confirm the forecast has exactly 168 rows.
   - Confirm there are no missing actuals or predictions.
   - Inspect the generated plot.

One environment issue: this Python environment currently has `numpy` but not `pandas` or `matplotlib`. I can either:

A. write the script plus a `requirements.txt`, then tell you the command to run, or  
B. create a local `.venv`, install `pandas` and `matplotlib`, run it, and report the actual accuracy.

I recommend B so I can give you the accuracy now. Proceed?