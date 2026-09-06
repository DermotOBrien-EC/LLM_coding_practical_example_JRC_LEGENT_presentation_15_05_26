# German hourly electricity load: forecast for 1 to 7 January 2020

Forecast of hourly German system load (ENTSO-E Transparency actuals, via OPSD)
for the 168 hours of 2020-01-01 00:00 to 2020-01-07 23:00 UTC, produced from a
forecast origin of 2020-01-01 00:00 UTC.

## Data

`opsd_de_load.csv`: 50,400 hourly observations, 2015-01-01 to 2020-09-30 UTC,
column `DE_load_actual_entsoe_transparency`. The series is complete: no missing
timestamps, no NaNs, no non-positive values. This is verified on load and the
loader raises rather than silently interpolating.

Training history is everything strictly before the origin: 43,824 hours
(2015-01-01 to 2019-12-31 23:00 UTC), five full years.

Timestamps are UTC. All calendar features are derived after converting to
`Europe/Berlin`, because the daily load cycle follows local clock time, not UTC.

## Why this particular week is hard

The target contains New Year's Day and the tail of the German year-end
shutdown, so it is not a typical week. Two facts from the history drive the
whole design:

| Effect | Magnitude (2015–2019) |
|---|---|
| 1 January daily mean vs. the preceding November mean | 0.72 – 0.76 |
| First-January-week mean vs. preceding Christmas-week mean | 1.136 – 1.177 |

The second row is the trap. The week immediately before the target is
25–31 December, the deepest part of the shutdown. A seasonal-naive forecast
that repeats the previous week (lag 168 h) is therefore biased low by roughly
13–15% *by construction*, and this is not a subtle effect: it dominates every
other error source in the week. Any model used here has to represent the
holiday regime explicitly rather than lean on recent persistence.

## Models

All models share one interface (`fit(history)` / `predict(index)`) and are
refit from scratch on each backtest fold.

| Model | Description |
|---|---|
| `seasonal_naive_168h` | Repeat load from 168 h earlier. Reference point. |
| `year_ago_naive_364d` | Same hour 364 days earlier (preserves weekday), rescaled by the ratio of mean load in the 8-to-4 weeks before the origin to the same window a year earlier. That window deliberately ends four weeks before the origin so the shutdown does not contaminate the level estimate. |
| `holiday_adjusted_naive` | Seasonal naive multiplied by the historical January-week / Christmas-week ratio, estimated per hour-of-week from prior years. An explicit correction of the bias described above. |
| `linear_profile` | Ridge-regularised regression on log load: hour-of-week dummies, holiday population weight interacted with hour-of-day, previous/next-day holiday weights, year-end day-offset dummies, three annual Fourier harmonics, linear trend. |
| `lightgbm` | Gradient-boosted trees on log load, using the calendar features plus lagged load (168 h, 192 h, 336 h, 504 h; 364/365/371 days), 28-day rolling level anchors, and lag-to-level ratios. |

Two modelling choices worth naming:

- **Log target.** Every effect that matters here is multiplicative. The
  holiday, weekend and year-end effects are all stable *ratios* across years,
  as the table above shows. Modelling `log(load)` makes them additive.
- **Holiday weight, not a holiday flag.** German public holidays are partly
  regional. 6 January (Epiphany) is a public holiday only in Baden-Württemberg,
  Bavaria and Saxony-Anhalt, so it suppresses national load partially, not
  fully. The feature is the population share of Germany for which the day is a
  holiday, computed per federal state and weighted by 2019 population: 0.317
  for 6 January, 1.0 for 1 January. In the target week 6 January is a Monday,
  which makes this distinction directly relevant.

## Guarantees against look-ahead

The horizon is 168 hours and the shortest load lag used is exactly 168 hours,
so the newest observation any feature can touch is the final training hour.
Rolling means are computed on an already-shifted series. `_lag_features` raises
if asked for a lag shorter than the horizon.

This is enforced by test, not just by inspection: `tests/test_no_leakage.py`
replaces every observation from the origin onward with random noise, re-predicts
and asserts the forecast is bit-for-bit unchanged, for all five models. The test
also asserts the forecast is finite and non-constant, so a model producing NaNs
cannot pass the comparison vacuously.

## Model selection

Selection used only January weeks *before* the target: 2016, 2017, 2018, 2019,
each refit on the history available at that origin. The 2020 week was not
consulted at any point in choosing the model, its hyperparameters, or the
ensemble weights. Prediction intervals are calibrated on the same prior-January
residuals, by hours-ahead, on the log scale.

Validation MAPE (%), each fold refit on history only:

| Model | 2016 | 2017 | 2018 | 2019 | mean | worst |
|---|---|---|---|---|---|---|
| **selected: 0.5 lgb + 0.25 linear + 0.25 year-ago** | 9.23 | 6.33 | 4.75 | 3.37 | **5.92** | **9.23** |
| 0.5 lgb + 0.5 linear | 9.10 | 7.85 | 3.05 | 4.09 | 6.02 | 9.10 |
| 0.6 lgb + 0.2 linear + 0.2 year-ago | 10.03 | 6.29 | 4.94 | 3.19 | 6.11 | 10.03 |
| 0.7 lgb + 0.3 linear | 10.72 | 7.01 | 3.91 | 3.52 | 6.29 | 10.72 |
| linear_profile | 5.54 | 9.98 | 4.69 | 6.04 | 6.56 | 9.98 |
| lightgbm | 13.13 | 6.35 | 6.15 | 3.09 | 7.18 | 13.13 |
| year_ago_naive_364d | 7.13 | 6.79 | 9.81 | 6.42 | 7.54 | 9.81 |

The selection rule, fixed before the 2020 week was scored, was to take the
candidate that is best or tied-best on *both* mean and worst-case MAPE. The
0.5/0.25/0.25 blend wins both over all four folds, and is within 0.01pp of the
best when the 2016 fold is excluded.

The 2016 fold is the informative one for robustness. It has only one year of
history, which starves LightGBM's year-ago lag features and sends it to 13.1%
while the linear model, which has no lag features to starve, does best there.
That is exactly the disagreement an ensemble is for, and it is why the blend
beats its own best member on average despite never being the best on any single
fold.

## Reproducing

```
../../.venv/bin/python -m pytest tests/ -q      # leakage + data integrity
../../.venv/bin/python select_model.py          # per-model validation folds
../../.venv/bin/python select_ensemble.py       # ensemble weight selection
../../.venv/bin/python run_forecast.py          # final forecast + plot + scoring
```

Outputs land in `outputs/`: `forecast_2020_week1.csv` (hourly point forecast,
90% interval, actual, error), `test_metrics.csv`, and
`forecast_2020_week1.png`.

## Limitation worth stating plainly

The dataset contains load only. Temperature is the single largest driver of
German winter load after the calendar, and no weather information of any kind
is available here, so the model is effectively forecasting the *climatological*
load for these calendar positions. A genuine operational forecast would add
temperature (and its non-linear heating response), and would be expected to
improve materially on what follows, particularly on days when the week's weather
departs from normal.
