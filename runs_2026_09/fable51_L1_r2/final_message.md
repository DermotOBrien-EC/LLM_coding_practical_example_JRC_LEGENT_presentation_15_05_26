Forecast done. The week is a backtest, since the file already holds actuals through September 2020, so the model was trained only on data through 31 December 2019 and then scored against what actually happened.

**Method.** A direct 168-hour LightGBM model on calendar features (hour, weekday, day of year, month), a national holiday flag, a Christmas-to-Epiphany flag, and load lags of 1 to 4 weeks plus 52 weeks. All lags are at least a week old, so every forecast hour uses only data known on 31 December. Quantile models give a p10 to p90 band. Times are Europe/Berlin, so the window is Wed 1 Jan 00:00 to Tue 7 Jan 23:00 local.

**Accuracy on the target week** (mean absolute percentage error over 168 hours):

| Model | MAPE | MAE |
|---|---|---|
| LightGBM | 3.3% | 1.8 GW |
| Year-ago same weekday | 10.8% | 5.5 GW |
| Seasonal naive, same hour last week | 12.7% | 7.2 GW |

The held-out validation on the same week of 2019 gave 2.4% MAPE, so 2020 ran a bit worse. The model over-forecast by about 1.3 GW on average, concentrated on Friday 3 January (a bridge day) and Monday 6 January (Epiphany, a holiday in Bavaria, Baden-Württemberg and Saxony-Anhalt that the national holiday flag does not carry). Weekend and New Year's Day errors were 1 to 3%. The p10 to p90 band covered 62% of hours instead of the intended 80%, so the band is too narrow for this week.

**Outputs** in the run directory:
- `forecast_jan2020_week1.csv`: 168 rows with local and UTC timestamps, forecast, p10, p90, seasonal naive, and actual load in MW.
- `forecast_jan2020_week1.png`: actual versus forecast with the uncertainty band.
- `forecast.py`: the full pipeline, ruff-clean under the project config. Runs in about a minute with `../../.venv/bin/python forecast.py`.

The obvious next improvement is a population-weighted regional holiday feature, which should fix most of the Epiphany miss.