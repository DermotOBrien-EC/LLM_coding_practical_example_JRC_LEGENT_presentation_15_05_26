The forecast is done and scored against the actual load for the week. All outputs are in the run folder.

**Setup.** The data is the OPSD German load series, hourly from 2015 to September 2020, with no gaps. I trained on everything before 2020, forecast 1 to 7 January 2020 in Berlin local time, and compared against the actuals that exist for that week. The model is LightGBM on calendar features (hour, weekday, day of year, national holidays, a population-weighted regional holiday flag for Epiphany, bridge days, position inside the Christmas to Epiphany lull) plus lags of one, two and three weeks and of 364 and 365 days. Every lag is at least 168 hours back, so the whole week is predicted in one shot from data available at the origin, with no recursion. Two naive baselines give a sense of scale.

**Forecast quality, mean absolute percentage error:**

| Week | LightGBM | Seasonal naive (previous week) | Same week last year |
|---|---|---|---|
| Jan 2017 backtest | 5.0 | 16.0 | 6.9 |
| Jan 2018 backtest | 7.4 | 11.9 | 9.4 |
| Jan 2019 backtest | 2.5 | 11.6 | 6.5 |
| **Jan 2020 target** | **4.0** | 12.7 | 10.8 |

**Daily mean load for the target week, MW:**

| Day | Forecast | Actual | Error |
|---|---|---|---|
| Wed 1 Jan (holiday) | 44,561 | 43,988 | 2.2% |
| Thu 2 Jan | 55,839 | 53,865 | 3.5% |
| Fri 3 Jan | 59,727 | 55,655 | 7.2% |
| Sat 4 Jan | 54,383 | 52,640 | 3.5% |
| Sun 5 Jan | 49,862 | 47,989 | 3.9% |
| Mon 6 Jan (regional holiday) | 58,040 | 56,631 | 2.6% |
| Tue 7 Jan | 65,707 | 62,598 | 4.9% |

**What to know about the result.** The daily shape and the holiday dips are right, but the model runs high on every day, worst on the Friday between New Year and the weekend. Many businesses treat the whole week as a bridge, and the model only partly learns that from four prior Christmas seasons. The 80% band is empirical, taken from the three backtest weeks, so it is a rough guide rather than a calibrated interval.

**Files in the run folder:**
- `forecast_jan2020.csv` has the 168 hourly rows with local and UTC timestamps, forecast, 10th and 90th percentile, naive baseline and actual.
- `forecast_jan2020.png` is the chart above (actual, forecast with band, naive).
- `metrics.json` holds all backtest and target scores plus feature importances.
- `forecast_jan2020.py` reproduces everything in about 90 seconds.