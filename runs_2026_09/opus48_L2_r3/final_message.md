The plot confirms the forecast tracks the actual load well: it correctly anticipates the low New Year's Day (holiday) trough, the daily double-peak shape, and the weekend dip. The main systematic miss is a slight over-prediction of the midweek daytime peaks (Jan 2–3), where the model runs a bit hot.

## Done

I built `forecast_load.py` and ran it. Here's what it does and what came out.

**Data**: `opsd_de_load.csv` — 50,400 clean hourly load values (MW) for Germany, 2015-01-01 to 2020-09-30, no gaps or missing values. I trained on the 43,824 hours *before* 2020-01-01 and forecast the 168 hours of 1–7 Jan 2020.

**Model**: A gradient-boosted tree (`HistGradientBoostingRegressor`) on calendar features only — hour-of-day, day-of-week, month, weekend flag, a German public-holiday flag, and a smooth yearly seasonal cycle. All of these are known in advance, so this is an honest week-ahead forecast: the model never sees any actual load from inside the target week. I computed features in Berlin local time (load follows the local clock) and derived the national holiday dates properly, including the Easter-based ones, so 1 Jan is flagged as a holiday.

**Accuracy on the target week**:

| metric | model | naive day/hour profile |
|---|---|---|
| MAE | 1,947 MW | 4,821 MW |
| RMSE | 2,412 MW | 7,393 MW |
| MAPE | **3.57%** | 9.84% |

So roughly **96.4% accurate** on average across the 168 hours, with a typical hourly miss of ~1,950 MW against an average load of ~53,400 MW. It beats a seasonal-naive baseline (average load by day-of-week and hour) by nearly 3x on error, mainly because the holiday flag lets it capture the sharp New Year's Day dip that the naive profile misses.

The plot (`forecast_vs_actual.png`) shows the fit: the shape, weekend dip, and holiday trough all track well; the one visible bias is a mild over-prediction of the Thu/Fri daytime peaks.

Two honest caveats:
- The `is_holiday` flag treats nationwide holidays uniformly; Jan 6 (Epiphany) is only a holiday in three German states, so it's not flagged, but its effect on total load that week was small.
- A calendar model has no weather input. Adding temperature would cut the error further, but it wasn't in the dataset.

To rerun: `.venv/bin/python forecast_load.py`.

Note: your `pyproject` further up the tree pins Python `<3.13`, so I created the venv with Python 3.12 via `uv` rather than the system 3.14.