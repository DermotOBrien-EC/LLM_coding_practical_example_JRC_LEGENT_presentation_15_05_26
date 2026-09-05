The study is complete. Final state:

```
code/     common.py  naive.py  sarima.py  prophet.py
          lightgbm_features.py  nbeats.py  patchtst.py  forecast.py
figures/  01..08 .png  (all 300 dpi)
metrics.json  metrics.csv  transcript.md  artifacts/ (result cache)
```

## Results

| Model | Test MAPE | RMSE (MW) | MAE (MW) | 1 Jan (holiday) | 2–7 Jan |
|---|---:|---:|---:|---:|---:|
| **LightGBM** | **5.46%** | 3787 | 3016 | 2.19% | 6.00% |
| Prophet | 7.61% | 4878 | 3798 | 13.28% | 6.67% |
| TSMixer | 10.78% | 7293 | 5685 | 4.66% | 11.80% |
| N-BEATS | 11.06% | 7727 | 5824 | 2.75% | 12.44% |
| SARIMA | 11.44% | 6871 | 6027 | 16.31% | 10.63% |
| Seasonal naive | 12.78% | 8809 | 7238 | 6.52% | 13.82% |

Winner coverage: 80% interval 53.6% actual, 95% interval 91.7%. Pinball 735 / 1476 / 497 MW at q=0.1/0.5/0.9.

## Things worth flagging

**PatchTST substitution.** darts 0.41.0 exposes no `PatchTSTModel`, so I used `TSMixerModel` (the prompt's named alternative). It keeps the `patchtst` slot name in `metrics.json` for schema conformance, with `model_class_used` recording the real class. Documented in `transcript.md` §2.

**One methodological decision that changes the headline number.** The prompt states all LightGBM lags are available in the test window. That is true for `lag_168h` and `lag_8760h`, but *not* for `lag_24h` or the rolling windows past the first day — those would be read from held-out actuals. I ran LightGBM recursively instead (its own predictions feed the short lags), which is leakage-free but scores worse than the leaky alternative would. Flagged in `transcript.md` §3.

**SARIMA training-window length was treated as a hyperparameter** (26 vs 52 weeks) because full-train MLE is minutes per candidate, making a real grid search infeasible. The validation search earned its keep: weekly Fourier regressors cut validation MAPE from ~9.5% to 6.4%.

**Deep models got a 22-epoch budget**, not 30, because the run environment kills long jobs; early stopping fired at 19 epochs for the winning N-BEATS candidate, so the cap likely did not bind there, but both TSMixer candidates ran to the cap and may be undertrained.

## Two findings the figures make plain

The rank ordering inverts the complexity ladder, and the reason is diagnosable rather than mysterious: N-BEATS and TSMixer were handed 25–31 December as their context window and extrapolated a holiday-shaped week into a working week. N-BEATS scores 2.75% on Jan 1 and 12.44% on Jan 2–7. Prophet, the only model with an explicit holiday term, is the only one *worse* on the holiday than on ordinary days.

The winner's error is almost entirely bias, not noise: mean residual −2809 MW (−5.26% of mean load). The 364-day lag carries 73.8% of its split gain, and that reference week averaged 58,883 MW, 10.3% above the test week. Its 80% intervals cover 53.6% because the quantile models were trained one-step-ahead then applied to recursively-built features, so the bands never widen with lead time. Both are named as the first fixes in the recommendation.

The single-origin, single-week design means differences of a few tenths of a point should not be read as real.