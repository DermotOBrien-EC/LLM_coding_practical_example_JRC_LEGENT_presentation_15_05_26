# German hourly load forecasting bake-off

## Data

The study uses the German national load series published by Open Power System Data and derived from the ENTSO-E Transparency Platform. The file contains 50,400 consecutive hourly UTC observations from 2015-01-01 00:00 through 2020-09-30 23:00, measured in megawatts. The timestamp grid was checked for duplicates and one-hour spacing, and the load column was checked for missing values. No gaps or missing values were found, so no imputation or row removal was needed.

## Why these six models

The six models form a deliberate complexity gradient. Seasonal naive copies the same hour from one week earlier and supplies the reference skill level. SARIMA removes last week's level, then uses a linear probability model to extend recurring daily changes. Prophet adds separate curves for long-run movement, daily, weekly, yearly, and German-holiday effects. LightGBM uses decision trees to combine calendar fields, federal holidays, past observations, and strictly past rolling summaries. N-BEATS uses stacked neural-network blocks to map the previous 168 hours into the next 168. The requested PatchTST slot uses Darts TSMixerModel because Darts 0.41.0 does not expose PatchTSTModel in this environment. TSMixer mixes information across the hours in the input window and keeps the same 168-hour input and output lengths, but it is not a literal PatchTST implementation.

## Validation strategy

The initial training window contains 41,616 hours from 2015-01-01 through 2019-09-30. The validation window contains 2,208 hours from 2019-10-01 through 2019-12-31, and the untouched test window contains 168 hours from 2020-01-01 through 2020-01-07. SARIMA order, Prophet prior scale, LightGBM tree settings, and the deep models' stopping epoch were selected only from validation MAPE. For each deep model, independent fits at 6, 12, 18, 24, and at most 30 epochs were compared using the median forecast from one fixed origin; the search stopped after two successive candidates failed to improve MAPE. Every selected configuration was then fit again on all 43,824 pre-test hours. Test observations were passed only to the common scoring functions after every forecast had been produced.

## Results table

| Rank | Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2-7 MAPE (%) | Runtime (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | LightGBM | 5.281 | 3687.0 | 2866.6 | 4.834 | 5.355 | 113.3 |
| 2 | Prophet | 7.607 | 4877.4 | 3795.0 | 13.254 | 6.666 | 30.7 |
| 3 | PatchTST / TSMixer | 9.605 | 7428.0 | 5570.3 | 6.021 | 10.202 | 240.2 |
| 4 | N-BEATS | 9.652 | 6899.6 | 5337.5 | 6.246 | 10.220 | 270.7 |
| 5 | SARIMA | 12.514 | 8718.4 | 7109.1 | 5.757 | 13.640 | 53.8 |
| 6 | Seasonal naive | 12.781 | 8808.6 | 7238.0 | 6.520 | 13.824 | 0.0 |

The winning LightGBM intervals achieved 83.3% coverage at the nominal 80% level and 100.0% coverage at the nominal 95% level. Pinball losses were 576.5 MW at q=0.1, 1420.9 MW at q=0.5, and 614.4 MW at q=0.9.

## Discussion

LightGBM won on the preregistered primary metric with a test MAPE of 5.281%, ahead of Prophet at 7.607%. Its largest daily error was 11.6% on Jan 06, so the weekly average does not describe uniformly good performance. The complete rank order was LightGBM, Prophet, PatchTST / TSMixer, N-BEATS, SARIMA, Seasonal naive. The seasonal-naive baseline ranked 6th. Its one-week copy failed when Christmas-week levels were carried into the first working days: MAPE rose from 6.520% on Jan 1 to 13.824% thereafter. SARIMA ranked 5th. Weekly differencing improved only slightly on naive, and its Jan 2-7 MAPE remained 13.640% because the linear model had no explicit holiday or working-day regime. Prophet ranked 2nd, but its explicit German-holiday term did not solve the holiday itself: Jan 1 MAPE was 13.254%, about twice its 6.666% error on Jan 2-7. A smooth additive holiday effect was not flexible enough for the observed hourly shape. LightGBM ranked 1st. Calendar, holiday, lag, and rolling features supported its nonlinear forecast, but its daily MAPE reached 11.6% on Jan 06 as recursive errors accumulated. The winning LightGBM uncertainty estimates were also imperfect: the nominal 80% and 95% intervals covered 83.3% and 100.0% of observations. N-BEATS and the TSMixer substitute ranked 4th and 3rd. Their worst daily MAPEs were 19.0% and 20.4% on Jan 07 and Jan 07. Both received only past load, so they had to infer future holiday regimes indirectly. The ordering is only partly predicted by model complexity: flexible models help when their inputs identify the coming regime. This one winter holiday week is not an annual skill estimate. Weather was deliberately excluded, deep training remains stochastic despite fixed seeds, and no intervals were recalibrated over multiple forecast origins.

## Recommendation

If exactly one of these implementations had to enter a JRC short-term load forecasting pipeline, I would choose LightGBM. It had the lowest held-out MAPE in the common test and its error profile is directly visible in the daily and residual figures. Before production use, I would run rolling-origin backtests across at least one full year, recalibrate its prediction intervals on those origins, test forecast latency and failure recovery, and add weather only in a separate study so its incremental value is measured rather than assumed. The seasonal-naive forecast should remain deployed beside it as a live fallback and drift alarm, not as an unreported comparator.

## Reproducibility note

All stochastic code used seed 42. The complete run took 713.0 seconds (11.9 minutes) on `Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64`. Reproduce from this directory with `../../.venv/bin/python code/forecast.py`; no package installation or external data download is required. Calendar and holiday indicators use the UTC date carried by the input, matching the evaluation labels rather than converting to German civil time. LightGBM point forecasts append their own prediction before computing later lag and rolling features, while each forecast quantile advances on its own recursive path. No observed value from a forecast window enters a later feature row. N-BEATS and TSMixer training windows used a stride of 4 hours to keep the complete six-model run within the thirty-minute reproducibility target.
