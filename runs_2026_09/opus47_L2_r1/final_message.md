Done. Script is at `runs/2ad3e5/forecast.py`, plot at `runs/2ad3e5/forecast_2020_w01.png`.

**Approach.** LightGBM regressor trained on 43,656 hourly rows (2015-01-08 to 2019-12-31). Features: hour, day-of-week, month, day, day-of-year, ISO week, weekend flag, German public-holiday flag (via `holidays`), year (trend), and the same-hour-last-week load (`lag_168h`). Calendar features are derived from local `Europe/Berlin` time so weekday/holiday effects line up with human schedules. No recursion; every feature for the 168 target hours is knowable at forecast time.

**Accuracy on 2020-01-01 to 2020-01-07 (168 h).**
- Mean actual load: 53,400 MW
- MAE: 1,889 MW
- RMSE: 2,339 MW
- MAPE: **3.47 %**

**Reading the plot.** Daily and weekly shape track well. The model over-predicts by roughly 1-2 GW during Thu 2 to Fri 3 Jan and again the afternoons of Mon 6 and Tue 7 (the first non-holiday days of the year, when actual load is still slightly depressed by the extended German holiday period even though the calendar says a normal workday). New Year's Day itself is captured correctly thanks to the holiday flag.