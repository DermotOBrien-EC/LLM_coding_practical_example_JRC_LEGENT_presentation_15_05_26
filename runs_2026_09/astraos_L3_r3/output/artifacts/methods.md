# Protocol, validation and engineering supplement

## Selected configurations

| Model | Selected setting | Validation MAPE (%) | Runtime (s) |
|---|---|---:|---:|
| LightGBM | 31 leaves, 400 trees | 3.30 | 20.5 |
| Seasonal naive | 168-hour repeat | 5.48 | 0.0 |
| Prophet + DE holidays | changepoint prior 0.05 | 5.64 | 20.9 |
| SARIMA + weekly Fourier | [2, 0, 0] x [1, 1, 0, 24] | 5.66 | 460.0 |
| N-BEATS | 2 epochs; 30 stacks, width 64 | 10.25 | 3.7 |
| Transformer (substitute) | 2 epochs; 2+2 attention layers, width 32 | 10.98 | 10.7 |

The validation leader is LightGBM. The baseline has no tunable parameters. SARIMA compares (1,0,0) and (2,0,0), each with (1,1,0,24), plus three weekly Fourier harmonics derived from UTC timestamps. Daily differencing alone does not create weekly structure. The residual variance is explicitly estimated in training and frozen, like every other parameter, during validation filtering. Prophet compares changepoint priors 0.01 and 0.05 with additive daily/weekly/yearly components and German federal holidays. LightGBM compares 15 leaves/250 trees and 31 leaves/400 trees (learning rate 0.05).

N-BEATS retains the default generic architecture: 30 stacks, one block per stack and four layers per block; layer width is reduced from 256 to 64 for the CPU budget. TransformerModel uses width 32, four heads, two encoder and two decoder layers, feed-forward width 64 and dropout 0.1. Neural training uses Adam at 0.001, batch size 32 and daily-strided examples covering all weekdays. Quantile loss trains the models; full rolling validation MAPE selects epochs. Patience is five epochs with minimum improvement 0.01 percentage points; the lowest recorded MAPE determines the refit epoch count. The live neural cap was 2 epochs; the best epoch was chosen on validation and refitted from scratch. This cap is too short for the configured five-epoch patience to trigger; the neural results are budget-limited, not converged comparisons. The originally planned cap was 30, but the file changed to 2 before neural fitting; the current value was preserved. No test score motivated this change.

## Features and forecast isolation

LightGBM uses UTC hour, weekday, month, weekend and public-holiday indicators; lags 24/168/8760 hours; and means/population standard deviations for the preceding 24/168 hours, excluding the target hour. It loses 8,760 supervised rows to annual-lag warm-up, but retains them as history. Unknown loads are replaced only by the predicted median, never actuals. The prompt's annual-lag timestamp is incorrect: 2020-01-01 00:00 UTC minus 8,760 hours is 2019-01-01 00:00 UTC. Jan 6 is not a nationwide German holiday. No external covariates were obtained; weather is discussed solely as future work.

Validation parameters and neural scalers are trained on data through September 30 only. The final 24-hour remainder is scored with 24 hours of weight; all other blocks are 168 hours. Neural forecasts for that last origin are generated for 168 hours but only the 24 validation hours are scored. Validation observations enter later prediction contexts, not fitting. Final refits use all pre-test history, except feature warm-up and daily-strided neural sampling as explicitly described. Test scoring happens after all six final forecasts have been saved.

## Probabilistic forecasts

SARIMA uses analytic Gaussian state-space errors conditional on estimated parameters. Prophet uses 1,000 predictive samples from its MAP fit, covering predictive noise and simulated trend changes rather than full posterior parameter uncertainty. Both neural models directly fit marginal quantiles 0.025, 0.1, 0.5, 0.9, 0.975. LightGBM fits those same five quantiles for every validation candidate and for the final refit, so the point-forecast rule is identical at selection and evaluation. Learned quantiles are monotonically rearranged before scoring. Its intervals condition on a recursive median path and do not propagate lag uncertainty. Neural/LightGBM point forecasts are the rearranged median; SARIMA/Prophet use their native conditional mean. The naive baseline has no native probabilistic model.

## Reproduction and checks

```sh
../../.venv/bin/python -m pytest code/test_study.py --import-mode=importlib --rootdir=. -o cache_dir=.cache/pytest -q
../../.venv/bin/python code/verify_outputs.py
../../.venv/bin/python code/forecast.py --render-only
```

`artifacts/*_run.json` retain every candidate or epoch score and exact settings. Per-model runtimes exclude import overhead; total runtime includes imports within the orchestrator plus initial figure/report generation, but not the final small timing-file refresh. Repeated runs overwrite the generated outputs. Hardware, BLAS and software versions can change timings and small floating-point details. The code uses a local namespace so the required `code/prophet.py` never shadows the external Prophet package. The forecasting entrypoint directs caches and temporary files into this directory; no packages are installed. An initial pytest invocation and early Ruff commands inherited the parent project's configuration and created `../../.pytest_cache` and `../../.ruff_cache` outside the requested directory. This was unintended; those caches were left untouched. Subsequent pytest commands explicitly set the local root/cache, and subsequent Ruff commands use `--cache-dir .cache/ruff`.

Tests were written first and initially failed because the implementation was absent. They cover split counts, every validation hour scored once, metric formulas, strictly shifted rolling features, recursive unobserved-lag substitution, no-gradient deep validation, frozen SARIMA variance and identical LightGBM quantile sets in validation and final fitting. The deep callback is also compared with public Darts.predict after training. The independent output checker recomputes six metric rows, winner coverage/pinball losses, quantile ordering, CSV agreement, PNG dimensions/DPI and input SHA-256 from saved files without importing the implementation's scoring functions.

Two runs were stopped before test scoring: the first lacked SARIMA filtering states with low_memory=True (`attempt1.log`); the second exposed residual-scale re-estimation under concentrated-scale validation filtering (`attempt2.log`). The corrected run retains states and holds an explicit sigma2 parameter fixed. A subsequent run completed, but a post-run structural audit found that sorting five fitted quantiles can change LightGBM's median while its validation candidates had fitted only q=0.5. A failing consumer test pinned the mismatch (`quantile_protocol_red.log`); the repair fits all five quantiles during validation too, then reruns all six models. This correction follows protocol identity, not test-error optimization. The superseded result is retained under `artifacts/pre_quantile_protocol_fix/`. No data were changed. The requested independent Claude review produced no findings before it was stopped, so cross-vendor review remains incomplete; `artifacts/review.md` records this without claiming approval.

## Figure encoding

Time-series figures are 11 x 6 inches; square comparisons are 6 x 6 inches, all at 300 dpi. A reordered tab10 subset passed the palette validator's adjacent-pair checks. Direct names and table values provide contrast relief, and forecast panels are faceted rather than relying on six color-only overlaid lines. Heatmap colors encode MAPE magnitude, not model identity. Metric bars show ratios to each metric's own seasonal-naive value, with raw %/MW labels and separate hatches for the three measures. Feature importance is share of total split gain, not causal importance.
