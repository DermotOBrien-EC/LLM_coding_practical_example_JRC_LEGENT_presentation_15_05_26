Done. Here's the forecast for the first week of January 2020 (168 hourly values, Jan 1–7).

## Approach
Since the data (OPSD/ENTSO-E, 2015-01-01 to 2020-09-30, clean and gap-free) contains the target week, I trained strictly on **2015–2019** and forecast the 168 hours of Jan 1–7 2020, then scored against the withheld actuals — a genuine out-of-sample backtest.

- **Model**: gradient-boosted trees (`HistGradientBoostingRegressor`) on features that are all known at forecast time: calendar terms in local (Europe/Berlin) time (hour, day-of-week, month, cyclical day-of-year), German public-holiday flags, a slow multi-year trend, and two fully-observed lags (load 1 week back = Dec 25–31 2019, and 364 days back = same weekday in Jan 2019). No recursive lags, so the whole week is a stable direct forecast.

## Results (out-of-sample, vs actuals)

| Method | MAPE | MAE | RMSE | Bias |
|---|---|---|---|---|
| **GBM model** | **2.80%** | 1,535 MW | 2,020 MW | +865 MW |
| Seasonal-naive (1 yr) | 10.76% | 5,483 MW | 7,253 MW | +5,483 |
| Naive (1 week) | 12.78% | 7,238 MW | 8,809 MW | −6,909 |

2.80% MAPE is strong for a week-ahead hourly forecast with no weather inputs, and ~4× better than the seasonal-naive baseline. The metric reproduces independently from the output CSV.

## Where it errs
The plot shows the daily double-peak shape tracked closely, with a consistent **positive bias on Jan 2–3 and Jan 6**. Cause: the model doesn't know the whole first week is still holiday-season (schools/industry on vacation, activity below a normal winter week), and **Jan 6 is Epiphany** — a public holiday in Bavaria, Baden-Württemberg and Saxony-Anhalt, so it dents national load, but it isn't in the federal-holiday flag. New Year's Day itself (flagged) and the Jan 4–5 weekend are near-perfect.

## Deliverables (in the working dir)
- `forecast_jan2020_week1.csv` — 168 rows: UTC + local timestamp, forecast_MW, actual_MW, abs_pct_err
- `forecast_jan2020_week1.png` — forecast vs actual with hourly error panel
- `forecast.py` — reproducible pipeline

One cosmetic note: the daily-totals table showed a tiny "2020-01-08" bucket — that's just the last UTC hour (Jan 7 23:00) mapping to 00:00 local Jan 8 under Berlin time; the forecast is exactly the 168 UTC hours of Jan 1–7.

If you want a lower bias, the clear next step is adding an explicit early-January/school-holiday feature and a regional-holiday (Epiphany) flag; adding temperature would be the bigger lever but needs an external weather series.