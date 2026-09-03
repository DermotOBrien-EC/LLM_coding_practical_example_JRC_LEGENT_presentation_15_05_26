Done. The script is `forecast_load.py` in the project directory, and it produced the forecast, plot, and metrics.

**Result: 3.2% mean absolute percentage error over the 168 hours**, about four times better than either naive baseline. The same procedure run on the first week of January 2019 scored 3.1%, so the 2020 number is typical rather than lucky.

| Forecast (Jan 1 to 7, 2020) | MAE | RMSE | MAPE | Bias |
|---|---|---|---|---|
| Model | 1,737 MW | 2,268 MW | 3.20% | +1,332 MW |
| Same hour last week | 7,238 MW | 8,809 MW | 12.78% | -6,909 MW |
| Same hour 52 weeks ago | 5,483 MW | 7,253 MW | 10.76% | +5,483 MW |

**How it works.** A gradient-boosted tree model (scikit-learn's HistGradientBoostingRegressor) trained on 2016 through 2019. Features are Berlin-local hour, weekday, season, computed German public and regional holidays plus Dec 24 and 31, days to and from the nearest holiday, and load lags of 1 to 4 weeks and 52 weeks. Every lag is at least 168 hours, so nothing inside the forecast week is used; the cutoff is midnight UTC on Jan 1 and the whole week is predicted from data before it. Hour and day features are computed in local time because that is what drives German consumption.

**Where it was wrong.** The forecast ran high all week, worst on Friday Jan 3 (7.6% daily MAPE, up to 6,000 MW over). That is the classic bridge day between New Year and the weekend, when much of industry stays shut; the model has only four prior Januaries to learn that from. Jan 1 itself and the weekend were within about 2%. The plot marks Jan 1 and Jan 6 (Epiphany, regional) as shaded bands and shows the error per hour in a lower panel.

**Files written:**
- `forecast_2020_w1.png`: forecast vs actual with three days of pre-cutoff context, plus the hourly error panel.
- `forecast_2020_w1.csv`: hourly actual, forecast, both baselines, and error.
- `forecast_metrics.json`: all metrics for both the 2020 target and the 2019 check.

**Two things to know about running it.** The system Python has no pandas, and a venv created inside this directory would not launch (symlink resolution is denied there), so the working interpreter is at `~/.venvs/energy_forecast_b3612e/bin/python`. Also, scikit-learn's boosting was about 100 times slower with default OpenMP threading on this 15-core machine, so the script pins one thread before importing sklearn; the whole run takes about 15 seconds. To run it:

```
~/.venvs/energy_forecast_b3612e/bin/python forecast_load.py
```

**Assumption to flag.** I took "the 168 hours from 2020-01-01 to 2020-01-07" as UTC timestamps, matching the CSV. In Berlin time that window starts at 01:00 on Jan 1 and ends at 00:00 on Jan 8. If you want local calendar days instead, the cutoff moves one hour earlier and the numbers shift slightly.

No weather data was available, and temperature is the main driver the model lacks. Adding it would be the first step if you want to push below 3%.