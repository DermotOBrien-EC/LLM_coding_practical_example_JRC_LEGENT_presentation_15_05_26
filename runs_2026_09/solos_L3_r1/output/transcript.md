# German hourly load forecasting bake-off

## Data

The study uses German national hourly electricity load from Open Power System Data, derived from the ENTSO-E Transparency Platform. The file covers 2015-01-01 00:00 UTC through 2020-09-30 23:00 UTC and contains exactly 50,400 observations. Timestamps were verified as unique, increasing and complete at hourly frequency; load was finite with no missing values, so no imputation or row removal was needed. The only inputs were load, UTC calendar fields, and German federal-holiday dates knowable in advance. Holiday indicators use the UTC calendar date so they align exactly with the frozen evaluation days.

## Why these six models

The six models form a deliberate complexity gradient. Seasonal naive tests whether copying the same hour from the previous week is already enough. SARIMA represents classical stochastic dynamics with daily seasonality. Prophet adds smooth daily, weekly and yearly structure plus German holidays. LightGBM combines calendar variables with lag and rolling summaries. N-BEATS tests a target-only deep basis-expansion network. Darts 0.41.0 does not expose `PatchTSTModel`, so the sixth slot uses the prompt-approved `TSMixerModel` substitute, an all-MLP sequence mixer with the same 168-hour input and output chunks. This substitution is recorded rather than silently changing the model class.

## Validation strategy

The initial training period contains 41,616 hours from 2015-01-01 through 2019-09-30. The separate validation period contains 2,208 hours from 2019-10-01 through 2019-12-31, and the untouched test contains 168 hours from 2020-01-01 through 2020-01-07. SARIMA order, Prophet priors and LightGBM tree settings were selected by validation MAPE. The deep models used validation MAPE for early stopping and epoch selection. Every selected configuration was then refit from scratch on all 43,824 pre-test hours. LightGBM validation and test forecasts were recursive from the forecast origin, so no observed validation or test load leaked through 24-hour lags or rolling windows.

## Results table

| Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2-7 MAPE (%) |
|---|---:|---:|---:|---:|---:|
| LightGBM | 5.31 | 3,756 | 2,891 | 5.54 | 5.28 |
| Prophet | 7.58 | 4,854 | 3,775 | 13.31 | 6.62 |
| SARIMA | 9.78 | 6,624 | 5,417 | 11.24 | 9.53 |
| N-BEATS | 10.02 | 7,342 | 5,607 | 6.61 | 10.59 |
| TSMixer substitute | 10.07 | 7,677 | 5,851 | 4.36 | 11.03 |
| Seasonal naive | 12.78 | 8,809 | 7,238 | 6.52 | 13.82 |

The winning model's pinball losses were 552.5 MW at q=0.1, 1,445.6 MW at q=0.5, and 707.6 MW at q=0.9. Nominal 80% and 95% prediction intervals achieved 84.5% and 100.0% empirical coverage.

## Discussion

LightGBM won with 5.31% MAPE, compared with 7.58% for Prophet. Its nominal 80% and 95% intervals covered 84.5% and 100.0% of observations. The full ranking was LightGBM, Prophet, SARIMA, N-BEATS, TSMixer substitute, Seasonal naive. The ranking is not a simple complexity ladder: this one-week test rewards models that reproduce the New Year transition, not models that are merely more flexible. The winner overpredicted most strongly on Mon Jan 06, where its median residual was -6,927 MW.

Seasonal naive recorded 6.52% MAPE on Jan 1 and 13.82% on Jan 2-7. Its weekly copy rule is strong when adjacent weeks resemble one another, but it cannot distinguish different holiday and working-day roles across weeks. SARIMA recorded 11.24% MAPE on Jan 1 and 9.53% on Jan 2-7. Its daily seasonal state smooths repeated cycles, but there is no explicit holiday signal and long validation forecasts tend to return toward learned average dynamics. Prophet recorded 13.31% MAPE on Jan 1 and 6.62% on Jan 2-7. It includes an explicit German holiday calendar, but its Jan 1 error shows that a named-holiday flag alone did not reproduce the observed profile. LightGBM recorded 5.54% MAPE on Jan 1 and 5.28% on Jan 2-7. Its calendar, annual, weekly and rolling features give it several relevant anchors, although recursive use of its own predictions can compound an early error. N-BEATS recorded 6.61% MAPE on Jan 1 and 10.59% on Jan 2-7. Its one-week input can learn recurring load shapes, but the target-only network has no direct way to know that Jan 1 is a holiday. TSMixer substitute recorded 4.36% MAPE on Jan 1 and 11.03% on Jan 2-7. The TSMixer substitute also sees only one target-history week, so unusual calendar events must be inferred indirectly from load shape alone.

These results are deliberately narrow. They cover one winter week, so they cannot establish annual average superiority or robustness during heat waves, industrial disruptions, or later pandemic conditions. No weather or market data were allowed, which isolates temporal structure but places a known ceiling on national load accuracy. Validation used one contiguous quarter, and the deep models were capped at short training schedules for a reproducible bake-off. The interval coverage estimates also use only 168 observations, so modest departures from the nominal rates should not be over-interpreted.

## Recommendation

If exactly one of these models had to enter a JRC short-term load pipeline, this bake-off supports LightGBM, because it produced the lowest error under the frozen comparison. Before deployment, I would run rolling-origin tests across at least a full year, calibrate intervals on multiple seasons, stress-test every German holiday separately, and monitor forecast drift by hour. I would then test weather covariates as a separate production extension because temperature is a major physical driver of national load; weather was intentionally excluded here to preserve the study's univariate question and should not be added without retaining this benchmark as an ablation.

## Reproducibility note

All stochastic code used seed 20200930. Total wall-clock runtime for the successful model fits and output assembly was 957.9 seconds. Hardware and operating-system record: `Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64`. The environment used the pre-installed project virtual environment and installed no packages. The initial N-BEATS attempt made the MPS backend exit with code 139, so both final deep-model runs used CPU. Reproduce from this directory with `../../.venv/bin/python code/forecast.py --force`.
