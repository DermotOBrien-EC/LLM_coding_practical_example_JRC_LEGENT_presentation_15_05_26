# German hourly electricity load forecast

## Forecast

- Window: **1 January 2020 00:00 to 7 January 2020 23:00 UTC**, 168 hourly intervals.
- File: `forecast.csv`; columns: `utc_timestamp`, `forecast_load_mw`.
- Values are hourly mean electrical power, interpreted as MW under the OPSD/ENTSO-E column convention. They are not instantaneous peaks or hourly MWh values.
- UTC matches the supplied timestamps. In Germany this is 1 January 01:00 to 8 January 00:00 CET. Calendar features use Europe/Berlin local time.
- Selected model: **equal_blend**.
- Forecast mean load: **55,793 MW**. Sum of hourly mean loads times one hour: **9.373 TWh**.
- Lowest hourly forecast: **37,293 MW** at 2020-01-01T05:00:00+00:00.
- Highest hourly forecast: **72,660 MW** at 2020-01-07T16:00:00+00:00.

### Daily summary (UTC)

| date_utc | mean_load_mw | minimum_load_mw | maximum_load_mw |
| --- | --- | --- | --- |
| 2020-01-01 | 45618 | 37293 | 52542 |
| 2020-01-02 | 57176 | 42064 | 66911 |
| 2020-01-03 | 60286 | 47592 | 68590 |
| 2020-01-04 | 53685 | 46291 | 61169 |
| 2020-01-05 | 50003 | 41166 | 58532 |
| 2020-01-06 | 59688 | 43692 | 68178 |
| 2020-01-07 | 64093 | 49048 | 72660 |

## Data and information cutoff

The supplied `opsd_de_load.csv` is the sole data source. The series is
`DE_load_actual_entsoe_transparency`. All observations on or after 2020-01-01 00:00 UTC
are removed before load-value parsing, feature generation, fitting, or validation.
The target week's actual loads were not used to select, fit, or score the forecast.
This is a retrospective forecast with an enforced historical information cutoff,
not a reconstruction using the now-known 2020 outcomes.

Training coverage: 2015-01-01T00:00:00+00:00 through
2019-12-31T23:00:00+00:00, **43,824 observations**. The input
history has no missing hours, duplicate timestamps, nonfinite values, or
nonpositive loads. No load imputation was required. Historical load range:
31,307 to 77,549 MW. The input was not independently
reconciled against the original provider or an as-of-2019 data vintage, so
historical revisions in this supplied series cannot be ruled out.

## Method fixed before validation

Six candidates were compared:

1. Previous-week load, lag 168 hours.
2. Median of the previous four same-weekday/hour loads, lags 168/336/504/672.
3. Previous year's weekday-aligned load, lag 8736 hours (52 weeks), scaled
   by the ratio of the latest 28-day mean to the 28-day mean 52 weeks earlier.
   This baseline is weekday-aligned, not holiday-aligned.
4. Calendar-only LightGBM regression.
5. LightGBM with the same calendar plus historical load lags.
6. A fixed 50:50 average of candidates 4 and 5.

Calendar features include local hour, weekday, month, day of month/year,
annual Fourier terms, year, German nationwide public holidays, New Year's Day,
Christmas, Christmas Eve, New Year's Eve, and a regional Epiphany indicator.
Signed, clipped distances to New Year and Christmas allow a holiday-period
ramp without classifying every weekday as an ordinary workday. Epiphany is
an indicator for 6 January, not an assumed nationwide closure or state-weighted
load estimate. No school-holiday calendar or weather forecast is used.
Historical weekday Epiphany examples are scarce and receive reduced weight
under the recency weighting, making the 6 January estimate particularly
uncertain.

The lag model uses [168, 336, 504, 672, 8736, 8760, 8784]-hour lags, the corresponding local calendar
states, and weekly median/mean features. The annual lags span 364/365/366 days,
so the model can distinguish weekday and calendar-date analogs. It omits the
first 672 training hours but keeps subsequent rows with absent annual lags,
using LightGBM's missing-feature handling. It does not invent missing loads.
Every target-week load lag is observed before the forecast origin; no recursive
predictions or realized loads from inside the forecast week enter the features.

Both regressors use 650 trees, learning rate 0.035, 31 leaves, minimum
24 observations per leaf, L2 regularization 20, column sampling 0.9,
seed 2020, two threads, and deterministic column-wise training.
Training weights decay with a 730-day half-life. There is no early stopping,
no scored-week tuning of tree count, and no broad hyperparameter search.
The year feature in a tree model does not extrapolate a linear trend beyond
2019. The selected candidate is refitted on all eligible pre-2020 history.

## Backtests

Selection minimizes mean hourly absolute error (MAE) across the complete
first weeks of January in 2017, 2018, and 2019. Each is a fixed-origin,
168-hour forecast trained solely on earlier observations. All three weeks
have equal length, so mean fold MAE equals pooled hourly MAE. The following
RMSE column is the mean of the three fold RMSE values, not pooled RMSE.

| model | mae_mw | rmse_mw | mape_pct |
| --- | --- | --- | --- |
| equal_blend | 2471.83 | 2795.3 | 4.64 |
| lag_calendar_lgbm | 2559.26 | 2963.6 | 4.84 |
| calendar_lgbm | 2594.05 | 2913.07 | 4.8 |
| scaled_annual_naive | 4364.73 | 6139.95 | 8.57 |
| four_week_median | 5053.72 | 7745.51 | 10.06 |
| weekly_naive | 7823.68 | 9138.64 | 13.2 |

The selected model's MAE is 43.4% lower than the best of the three
seasonal baselines on these selection folds. These baselines are imperfect:
the previous week contains Christmas, while a 52-week analog can place
New Year's Day on a different holiday status.

### Learned candidates by January week (MAE, MW)

| fold_start_utc | calendar_lgbm | equal_blend | lag_calendar_lgbm |
| --- | --- | --- | --- |
| 2017-01-01T00:00:00+00:00 | 4694.23 | 3592.34 | 2872.96 |
| 2018-01-01T00:00:00+00:00 | 1646.22 | 2431.77 | 3370.08 |
| 2019-01-01T00:00:00+00:00 | 1441.69 | 1391.37 | 1434.75 |

The lowest-error component changes across years. The fixed blend is selected
for its three-year mean performance, not because it dominates every fold.

### Selected model by validation week

| fold_start_utc | fold_type | mae_mw | rmse_mw | mape_pct | bias_mw |
| --- | --- | --- | --- | --- | --- |
| 2017-01-01T00:00:00+00:00 | new_year | 3592.34 | 4023.67 | 6.59 | -2049.43 |
| 2018-01-01T00:00:00+00:00 | new_year | 2431.77 | 2726.12 | 4.75 | 2094.94 |
| 2019-01-01T00:00:00+00:00 | new_year | 1391.37 | 1636.12 | 2.58 | -889.8 |
| 2019-10-09T00:00:00+00:00 | robustness | 965.31 | 1179.31 | 1.74 | -587.77 |
| 2019-11-06T00:00:00+00:00 | robustness | 1328.92 | 1564.2 | 2.24 | -1276.23 |
| 2019-12-04T00:00:00+00:00 | robustness | 1015.57 | 1254.54 | 1.7 | -114.7 |
| 2019-12-18T00:00:00+00:00 | robustness | 2381.9 | 3087.96 | 4.41 | 2176.89 |
| 2019-12-25T00:00:00+00:00 | robustness | 1523.68 | 1835.04 | 3.35 | 826.06 |

The five late-2019 weeks are separate robustness checks and do not enter
model selection. January's three folds represent only three New Year
transitions, starting on Sunday, Monday, and Tuesday, whereas 2020 starts
on Wednesday. These are selection-set errors, not an unbiased estimate of
the selected model's future error. The observations within each week are
strongly correlated, so 504 hourly residuals are not 504 independent holiday
examples. No calibrated prediction intervals are claimed or supplied.
Unusual weather, industrial shutdowns, and calendar-specific return-to-work
patterns can materially shift the realized load.

## Reproduction and artifacts

From this directory, using the supplied project environment:

```sh
../../.venv/bin/python forecast_load.py
../../.venv/bin/python -m pytest -q test_forecast_load.py
```

- `forecast.csv`: all 168 point forecasts, MW, two decimal places.
- `daily_summary.csv`: daily mean/minimum/maximum point forecasts, MW.
- `backtest_metrics.csv`: all models on all eight validation weeks.
- `backtest_predictions.csv`: historical validation labels and forecasts for independent metric recomputation; contains no 2020 target labels.
- `run_metadata.json`: cutoff, model settings, versions, and source fingerprint.
- `forecast_load.py` and `test_forecast_load.py`: executable method and tests.

Input SHA-256: `2f20c7560c962c7b5804b1854243c1b27ecd6358e24951238e12ae4336fa529c`.
