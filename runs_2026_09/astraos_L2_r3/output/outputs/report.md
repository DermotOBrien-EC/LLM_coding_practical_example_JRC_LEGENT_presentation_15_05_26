# German hourly electricity load forecast

Selected model: **calendar_only**, a histogram gradient-boosted tree regressor.

Forecast: 2020-01-01T00:00:00+00:00 through 2020-01-07T23:00:00+00:00, inclusive (168 hours).
Historical observations: 2015-01-01T00:00:00+00:00 through 2019-12-31T23:00:00+00:00 (43,824 rows).
The first 672 hours provide lag warmup, leaving 43,152 fitted rows.

## Holdout accuracy

| Method | MAE (MW) | RMSE (MW) | MAPE (%) | WAPE (%) | Bias (MW) |
|---|---:|---:|---:|---:|---:|
| calendar_only | 2,810.3 | 3,447.7 | 5.08 | 5.26 | +2,720.2 |
| previous_week | 7,238.0 | 8,808.6 | 12.78 | 13.55 | -6,908.5 |
| previous_52_weeks | 5,483.2 | 7,253.4 | 10.76 | 10.27 | +5,483.2 |
| previous_365_days | 4,899.9 | 5,910.6 | 9.18 | 9.18 | +2,533.2 |

MAE is the average absolute hourly error. RMSE penalizes larger misses more strongly.
MAPE is the average absolute percentage error, not an 'accuracy percentage'.
WAPE divides total absolute error by total actual load. Positive bias means overprediction.

## Method and safeguards

- All 168 predictions are issued together; no observed January load feeds later predictions.
- Time is UTC for the forecast and evaluation. Calendar features use Europe/Berlin local time.
- Features include daily/weekly/annual cycles, date, nationwide holidays and adjacent days, and year-end calendar flags.
- Epiphany is a separate date flag, not incorrectly labeled a nationwide public holiday.
- Lagged candidates use load from 1, 2, 3, 4 and 52 weeks, and 365 days earlier. Every lag is at least 168 hours.
- Early annual lags can be missing; the tree model handles them natively. No backward filling is used.
- Three fixed candidate models are compared on six rolling-origin seven-day validation windows before 2020.
- Validation includes the first weeks of 2018 and 2019 and a Christmas week, as well as ordinary seasons.
- Selection minimizes equally weighted mean validation RMSE among the trained candidates; baselines are reported separately.
- Each fold refits on strictly earlier data. Early stopping is disabled, so there is no random validation split.
- Target actuals and later-2020 observations are not used to choose or fit the model.
- Input timestamp gaps, duplicates and invalid loads cause an error rather than silent imputation.

## Limitations

This is a retrospective holdout evaluation of one holiday-heavy week, not a guarantee of future accuracy.
The preceding-week baseline copies Christmas week, so also compare the annual baselines.
The 52-week baseline aligns weekdays; the 365-day baseline aligns calendar dates for this non-leap interval.
There are no weather, industrial-activity or state-level school-holiday inputs, and no calibrated prediction intervals.
Only Epiphany is explicitly flagged as a regional holiday; other regional effects are left to date features.
The supplied OPSD file may contain retrospective revisions; its historical publication vintage is not verified.

## Outputs

- `forecast.csv`: all 168 actuals, forecasts, baselines and hourly errors.
- `forecast.png` and `forecast.svg`: actual versus forecast.
- `accuracy.csv`, `daily_scores.csv`, `validation_scores.csv`: numerical results.
- `metrics.json`: results, selection details, timestamps, input digest and dependency versions.
