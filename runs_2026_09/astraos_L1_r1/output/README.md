# German hourly electricity-load forecast: 1–7 January 2020

## Deliverable

`forecast.csv` contains all 168 hourly point forecasts and nominal 90%
empirical prediction bands. Load is interpreted as MW, consistent with the
OPSD ENTSO-E series named in the supplied input. The values forecast that
specific series, not a separately estimated measure of gross German demand.

The forecast covers **2020-01-01 00:00 UTC through 2020-01-07 23:00 UTC**,
inclusive. UTC follows the input timestamp convention. This is **01:00 CET
on 1 January through 00:00 CET on 8 January**, not the seven complete German
civil days. Both UTC and Europe/Berlin timestamps are included in the CSV.
No interpretation as ISO calendar week 1 is intended.

| UTC date | Mean load (GW) | Peak hourly load (GW) |
|---|---:|---:|
| 1 January | 44.75 | 51.54 |
| 2 January | 56.08 | 64.37 |
| 3 January | 57.35 | 64.99 |
| 4 January | 51.12 | 58.07 |
| 5 January | 48.12 | 55.48 |
| 6 January | 57.68 | 64.94 |
| 7 January | 60.26 | 67.91 |

- Weekly mean load: **53.62 GW**.
- Weekly energy: **9.009 TWh**, calculated by summing 168 hourly MW values
  over their one-hour intervals.
- Peak: **67.91 GW at 10:00 UTC on 7 January** (11:00 CET).
- Minimum: **37.45 GW at 05:00 UTC on 1 January** (06:00 CET).

The low New Year's Day forecast, weekend decline and subsequent weekday
recovery come from historical calendar patterns, not from target-week actuals.
CSV numbers are rounded to two decimal places for reproducibility of delivery;
that precision is not a claim of predictive accuracy.

## Information boundary

The source is `opsd_de_load.csv`, column
`DE_load_actual_entsoe_transparency`. Although that file extends into
September 2020, every observation dated **1 January 2020 or later is excluded
from model fitting, selection and uncertainty estimation**. Target-week actuals
were not used to tune or evaluate this forecast.

Training uses 43,824 observations from 2015-01-01 00:00 UTC through
2019-12-31 23:00 UTC. The training series has unique, continuous hourly
observations with finite, positive load values. This is a retrospective
forecast assuming those historical observations were available at the origin;
publication delays and subsequent revisions cannot be reconstructed from the
supplied file.

## Method and selection

Five candidates were specified before running the backtests:

1. Weekly naive: repeat the previous 168 hours.
2. Calendar ridge: regularized additive hour/day, month/hour and holiday-season
   effects, with a recent non-holiday level adjustment.
3. Calendar boosting: LightGBM using German local-calendar features only.
4. Lagged boosting: the same calendar inputs plus lags of 168, 336, 504, 672,
   8,736 and 8,760 hours.
5. Holiday ensemble: the equal-weight mean of the ridge and two boosting models.

Features include local hour and weekday, month, annual seasonal harmonics,
trend, national holidays, separate New Year's Day and Christmas indicators,
a regional Epiphany indicator, and date-specific Christmas shutdown / January
recovery features. Epiphany is not classified as a national German holiday.
The holiday calendar comes from the installed `holidays` package.

Training observations receive exponentially decaying weights with a two-year
half-life. Both boosting models use 500 trees, learning rate 0.04, 31 leaves,
minimum 40 observations per leaf, L2 regularization 5 and fixed seed 2020.
Ridge uses penalty 2. No hyperparameter search or target-week inspection was
performed. Lagged candidates use no lag shorter than the entire 168-hour
horizon, so all forecast-time load features are known at the forecast origin. These are
fixed-hour offsets, not leap-year-adjusted same-calendar-date lags. Composite
calendar codes enter the trees as numeric features, while the ridge model
uses one-hot encoding.

Each validation origin is 1 January of 2017, 2018 or 2019. A model is fitted
using only the data strictly before that origin and forecasts the next 168
hours. Selection uses the lowest mean fold MAE. Each fold has the same number
of hours. The selected model is **calendar boosting**; it was refitted using
all pre-2020 training observations.

| Candidate | Mean fold MAE (GW) | Mean fold MAPE |
|---|---:|---:|
| Calendar boosting | 2.156 | 3.99% |
| Holiday ensemble | 2.208 | 4.14% |
| Lagged boosting | 2.429 | 4.64% |
| Calendar ridge | 2.690 | 4.98% |
| Weekly naive | 7.824 | 13.20% |

Calendar boosting's MAE is 72.4% lower than the weekly baseline on these
selection folds. The weekly baseline is particularly weak here because it
copies Christmas-week demand into January. This is not a claim of superiority
over all reasonable forecasting methods.

Selected-model MAE by year is 3.274 GW (2017), 1.810 GW (2018), and 1.383 GW
(2019). The difference between the top two candidates is small relative to
variation across years. Reported metrics are model-selection results, not an
independent test of the chosen model. In `model_comparison.csv`, every metric,
including RMSE, is the arithmetic mean of the three per-year metrics; mean
fold RMSE is not the same as RMSE pooled over all 504 hours. Bias is forecast
minus actual.

## Uncertainty and limitations

The `lower_90_mw` and `upper_90_mw` columns add the 5th and 95th percentiles
of the selected model's 504 historical validation residuals to each point
forecast. Residual means actual minus forecast. The offsets are approximately
**−3.761 GW and +4.753 GW**.

These are **nominal pointwise 90% empirical bands**, not independently
calibrated coverage guarantees or simultaneous bounds on the entire week.
The same three weeks are used for model selection and residual estimation,
and hourly residuals are correlated. Applying the pooled bands back to those
selection folds gives coverage of **75.6% in 2017, 94.0% in 2018 and 99.4% in
2019**. Those figures are descriptive, not independent coverage tests. The
folds also have different amounts of training history. Bands have a constant
width and do not model variation in uncertainty by hour or holiday. Do not sum
their endpoints to obtain a claimed 90% weekly-energy interval.

No temperature forecasts, weather observations, industrial production
forecasts, or external demand scenarios were supplied or used. Exceptional
weather, unusual shutdown patterns and changes in the source series can
therefore produce errors outside these bands.

## Files and reproduction

- `forecast.csv`: 168 UTC hours, local timestamps, point forecasts and bands.
- `daily_forecast.csv`: UTC-day mean, minimum, maximum and energy.
- `model_comparison.csv`: mean validation metrics by candidate.
- `backtest_metrics.csv`: metrics for every candidate and validation year.
- `backtest_predictions.csv`: all validation forecasts, actuals and residuals.
- `forecast_metadata.json`: cutoff, model, versions, data hashes and totals.
- `forecast_model.py`: complete reproducible forecasting pipeline.
- `test_forecast.py`: executable input-boundary and calendar tests.
- `test_forecast_artifacts.py`: output checks, metric recomputation and a full
  rerun with every post-origin load replaced by a nonnumeric sentinel.
- `review.md`: independent peer review and the response to each finding.

From this directory, using the supplied project environment:

```sh
../../.venv/bin/python -m pytest test_forecast.py test_forecast_artifacts.py -q
../../.venv/bin/python forecast_model.py --output-dir reproduced
```

The original source CSV is unchanged. Output contains no realized 2020 loads.
Reproduction is verified in the supplied environment; byte identity across
platforms or library versions is not guaranteed. Scikit-learn is present as
a transitive dependency in that environment; a standalone environment must
install it explicitly along with NumPy, pandas, LightGBM and holidays.

Eight runtime tests pass, including the full future-value-poisoning test.
Static typing is not fully verified: `mypy --strict` reports missing pandas
and scikit-learn typing information in the supplied environment. The
source-level holiday import and fixture annotation findings were corrected
without changing the forecast. The parent environment and dependency
manifest were not modified to install additional type stubs.
