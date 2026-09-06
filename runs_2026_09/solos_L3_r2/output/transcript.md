# German hourly load forecasting bake-off

## Data

The study uses German national hourly load from Open Power System Data, derived from the ENTSO-E Transparency Platform. The file covers 2015-01-01 00:00 UTC through 2020-09-30 23:00 UTC and contains exactly 50,400 hourly observations in megawatts. The timestamp index is unique and continuous, and the load column has no missing values, so no rows were dropped and no imputation was needed.

## Why these six models

The six models form a deliberate complexity gradient. Seasonal naive tests whether a model can beat a one-week copy. SARIMA adds classical autoregressive and daily seasonal structure. Prophet adds smooth daily, weekly, and yearly components plus German public holidays. LightGBM adds nonlinear interactions among calendar fields, lags, and prior-only rolling summaries. N-BEATS learns a direct week-to-week mapping with a deep multilayer perceptron. Darts 0.41.0 does not expose `PatchTSTModel`, so the sixth slot uses the prompt-approved `TSMixerModel` substitute, another global neural forecaster with the same 168-hour input and output chunks. No weather, price, fuel, or other external series was used.

## Validation strategy

The initial training window contains 41,616 hours from 2015-01-01 through 2019-09-30. The validation window contains 2,208 hours from 2019-10-01 through 2019-12-31, and the untouched test contains 168 hours from 2020-01-01 through 2020-01-07. SARIMA candidates used explicit 24-hour differencing and Burg estimates for AR orders 1, 24, and 168, equivalent to `(p,0,0)(0,1,0,24)` models; Prophet regularization and seasonality mode, and LightGBM tree settings were also selected by validation MAPE. N-BEATS and TSMixer used validation-MAPE monitoring under capped training, with a maximum of 8 and 15 epochs respectively; both selected the maximum permitted epoch count. Each selected configuration was then rebuilt and refit on all 43,824 pre-test hours. LightGBM validation and test forecasts were recursive: after the forecast origin, lag and rolling features used the model's own earlier predictions rather than held-out actual values. This keeps the 168-hour test genuinely untouched.

## Results

| Rank | Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2 to 7 MAPE (%) | Runtime (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | LightGBM | 5.105 | 3676.7 | 2768.3 | 4.498 | 5.207 | 68.1 |
| 2 | Prophet | 7.614 | 4878.2 | 3798.0 | 13.275 | 6.670 | 47.1 |
| 3 | TSMixer | 10.246 | 7260.3 | 5100.8 | 32.888 | 6.473 | 6.2 |
| 4 | SARIMA | 12.219 | 8584.1 | 6974.3 | 5.223 | 13.385 | 3.6 |
| 5 | N-BEATS | 12.547 | 8466.0 | 6278.0 | 33.682 | 9.024 | 9.4 |
| 6 | Seasonal naive | 12.781 | 8808.6 | 7238.0 | 6.520 | 13.824 | 0.0 |

The winning probabilistic forecast had actual coverage of 85.1% for the nominal 80% interval and 100.0% for the nominal 95% interval. Its pinball losses were 553.2 MW at q=0.1, 1379.7 MW at q=0.5, and 655.2 MW at q=0.9.

## Discussion

LightGBM won the held-out week with a MAPE of 5.11%, ahead of Prophet at 7.61%. Its MAPE was 60.1% lower than the seasonal-naive anchor. LightGBM handled New Year's Day best (4.50% MAPE) and LightGBM was also best over Jan 2 to Jan 7 (5.21%). Prophet's explicit German holiday calendar did not guarantee good holiday performance: its Jan 1 MAPE was 13.28%. N-BEATS and TSMixer failed most sharply on Jan 1, with MAPE of 33.68% and 32.89% respectively.

The seasonal-naive model copied the preceding week's shape exactly, so it could not adjust for differences between Christmas week and the first week of January. SARIMA represented daily autocorrelation and differencing but had no explicit holiday switch. Both happened to track Jan 1 more closely than the later days, then degraded to 13.82% and 13.38% MAPE from Jan 2 to Jan 7. Prophet had daily, weekly, yearly, and holiday components, but its smooth additive or multiplicative structure did not capture the New Year's Day level shift well in this test. LightGBM could combine the holiday flag with annual, weekly, and daily lags and rolling summaries, although recursive forecasting allowed its own errors to feed later lag and rolling features. N-BEATS and TSMixer learned week-sized mappings directly from the univariate series. Their results reflect a single random seed and deliberately short, validation-monitored, capped training, as well as model class.

LightGBM's nominal 80% interval covered 85.1% of observations, while its nominal 95% interval covered 100.0%. Covering every observation suggests that the 95% interval was conservative or overwide for this single week, not that its calibration is proven. The test window contains only one week and one federal holiday, so it cannot establish year-round robustness, rare-event performance, or interval calibration across seasons.

## Recommendation

If exactly one of these models had to enter a JRC short-term load pipeline, I would take LightGBM, the model that won the untouched test week. Before production, I would repeat the comparison with rolling-origin tests across all seasons and several years, calibrate its prediction intervals on multiple validation windows, monitor holiday and daylight-saving transitions separately, and then test whether adding operationally available weather forecasts improves accuracy without weakening data governance.

## Reproducibility note

All stochastic models and sampling calls used random seed 2020. N-BEATS used one training window every 336 hours and TSMixer used one every 168 hours to keep the default 168-hour architectures reproducible within the time budget while still spanning the full training history. The complete run took 134.3 seconds on Apple M5 Pro with 24 GiB RAM. System record: `Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64`. Run from this directory with `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ../../.venv/bin/python code/forecast.py`. The study uses only `opsd_de_load.csv` plus deterministic timestamp-derived German federal holiday indicators from the installed `holidays` package.
