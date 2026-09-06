# German hourly load: a six-model week-ahead bake-off

## 1. Data

Open Power System Data, derived from ENTSO-E Transparency, supplies **50,400** German national hourly load observations (MW), from 2015-01-01 00:00 to 2020-09-30 23:00 UTC. Checks confirmed exact endpoints, consecutive unique hours, positive finite loads and no missing values; no imputation was needed. Later 2020 data appear only in the overview. No external load drivers were used.

## 2. Why these six models

Seasonal naive repeats the preceding week. SARIMA adds daily dynamics and timestamp-derived weekly Fourier terms. Prophet adds daily, weekly and yearly patterns plus German federal holidays. LightGBM learns nonlinear calendar/lag relationships. N-BEATS learns a direct week-to-week mapping; a transformer adds attention-based sequence processing. **Darts 0.41.0 has no PatchTSTModel**, so TransformerModel substitutes in that slot (JSON key `patchtst`). TSMixer was not used because it is an MLP, not a transformer.

## 3. Validation strategy

Train: **41,616** hours through September 2019; validation: **2,208** hours in October-December; test: **168** hours on January 1-7, 2020. Validation scores thirteen consecutive week-ahead forecasts plus a final 24-hour block. Previous validation observations supply later-origin context, never fitting or gradients. SARIMA filters with all parameters fixed; Prophet remains a fixed calendar regression. Every selected configuration is refitted on **43,824** pre-test hours, maximizing available history without exposing test outcomes.

All test forecasts use a single January 1 origin. LightGBM recursively substitutes predictions for unknown loads; rolling features end at t-1. Neural contexts/outputs are 168 hours, with training-only scaling and 24-hour training stride. The live neural cap was 2 epochs; the best epoch was chosen on validation and refitted from scratch. This cap is too short for the configured five-epoch patience to trigger; the neural results are budget-limited, not converged comparisons. Candidate grids, selected settings, validation scores and interval construction are in [the protocol supplement](artifacts/methods.md).

## 4. Results

Sorted by held-out MAPE. **Jan 2-7 includes the weekend**, not just working days; federal holidays and daily metrics use UTC calendar dates.

| Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2-7 MAPE (%) |
|---|---:|---:|---:|---:|---:|
| LightGBM | 3.91 | 2,360 | 1,983 | 5.65 | 3.62 |
| Prophet + DE holidays | 7.61 | 4,878 | 3,798 | 13.28 | 6.67 |
| Seasonal naive | 12.78 | 8,809 | 7,238 | 6.52 | 13.82 |
| Transformer (substitute) | 12.99 | 8,513 | 6,357 | 31.80 | 9.85 |
| N-BEATS | 13.45 | 8,744 | 6,921 | 25.60 | 11.42 |
| SARIMA + weekly Fourier | 19.25 | 11,521 | 10,473 | 10.23 | 20.75 |

Winner interval coverage: **75.6% at 80% nominal**, **95.8% at 95% nominal**. Pinball losses at q=0.1/0.5/0.9: **465.1 / 991.5 / 384.9 MW** (lower is better).

Five quantiles support 80% and 95% bands for every fitted model. LightGBM's recursive marginal intervals do not propagate lag uncertainty; no intervals receive test-based calibration. “Winner” describes this week, not a production model selected by test performance.

## 5. Discussion

LightGBM won with 3.91% MAPE, a 69.4% reduction relative to seasonal naive. Calendar indicators and recent/annual load lags let it combine holiday information with nonlinear reuse of past patterns. This is a plausible explanation, not a causal attribution: no feature ablation was run. This is an unusual benchmark: naive copies December 25-31, so the test mixes New Year's Day with a return from the Christmas period. The ranking may differ in ordinary winter weeks.

LightGBM overpredicted on average and was worst on Jan 1 (5.6%); Prophet + DE holidays underpredicted on average and was worst on Jan 1 (13.3%); Seasonal naive underpredicted on average and was worst on Jan 7 (24.7%); Transformer (substitute) overpredicted on average and was worst on Jan 1 (31.8%); N-BEATS underpredicted on average and was worst on Jan 1 (25.6%); SARIMA + weekly Fourier underpredicted on average and was worst on Jan 4 (27.3%). The holiday breakdown and daily heatmap show whether the overall score hides a calendar-day failure. High Jan 1 error was a hypothesis, not an imposed outcome. Naive and SARIMA stay too low as demand resumes; the neural forecasts fail to suppress the New Year's Day peak. Prophet's holiday indicator helps distinguish calendars but does not eliminate its Jan 1 mismatch.

Theory suggests that holiday calendars can help at calendar breaks and that neural models need sufficient training. It does not imply a universal complexity ranking. Here the short neural budget and narrowed layers are part of the measured configuration, not evidence about the best attainable performance of either model class. The validation origins start on a different weekday from test, but every model shares the same forecast protocol.

There is no weather information, multi-seed analysis or repeated test-week experiment. Hourly errors are dependent: 168 hours are not 168 independent replications. Interval coverage on one holiday week cannot establish calibration, and correlated lag features make split-gain importance descriptive rather than causal. Neither test rank nor a plausible mechanism justifies deployment alone.

## 6. Recommendation

For JRC production qualification I would choose **LightGBM**, the validation leader (3.30% MAPE), rather than select a production class by this test week. First evaluate untouched weeks across seasons/holidays, publication delays and daylight-saving boundaries, and calibrate horizon-specific intervals. Repeat neural seeds where relevant. Weather augmentation would be a separate experiment, outside this univariate study.

## 7. Reproducibility note

From this directory, without installing packages:

```sh
../../.venv/bin/python code/forecast.py
../../.venv/bin/python code/verify_outputs.py
```

Seed **2026**; CPU execution, four compute threads. Successful-run wall time: **523.7 s (8.7 min)**, including validation, refits, sampling and initial rendering. Hardware:

`Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64`

Versions and input hash: `artifacts/run_manifest.json`. Saved point/quantile forecasts allow independent rescoring. `--render-only` regenerates figures/report without training. Eight consumer tests passed; independent model review did not complete (see the supplement). Figure 03 uses baseline-normalized bars with raw %/MW labels, avoiding mixed units; all figures use a fixed tab10 subset. Optional figures follow the requested top-two rule.
