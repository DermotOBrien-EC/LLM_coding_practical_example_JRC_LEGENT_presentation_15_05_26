# German hourly load: six-model forecasting bake-off

## 1. Data

The supplied Open Power System Data / ENTSO-E Transparency series contains
50,400 hourly load observations (MW), from 2015-01-01 00:00 to
2020-09-30 23:00 UTC. Exact grid, uniqueness and finite positive values were
verified: no imputation or resampling was needed. Inputs are load and known
calendar features only. Later-2020 data appear only in the overview.

## 2. Why these six models

Weekly naive is the reference. SARIMA adds linear daily dependence and three
weekly Fourier pairs derived from weekday/hour. Prophet adds daily, weekly,
yearly and federal-holiday effects. LightGBM captures nonlinear calendar,
lag and rolling-feature relationships; default-stack N-BEATS learns a
week-to-week mapping. The sixth model tests attention-based forecasting:
Darts 0.41.0 lacks `PatchTSTModel`, so its compact `TransformerModel` was
substituted before scoring. The available TSMixer is not a transformer.

## 3. Validation strategy

Train has 41,616 hours through 2019-09-30; October–December validation has
2,208. Train-fitted candidates forecast 13 seven-day blocks plus 24 hours,
weighted equally per hour. Pre-origin observations may update context or
filter state, never coefficients. Prophet's calendar curve needs no state
update. Every selected model is then fitted afresh on all 43,824 pre-test
hours, maximizing usable history. The 168-hour test forecast is issued at
2020-01-01 00:00 UTC: no within-test observations enter any model's features
or selection. Test ranks are descriptive, not a second selection round.

| Model | Selected validation MAPE (%) | Selection |
|:--|--:|:--|
| SARIMA | 6.32 | best converged candidate |
| Prophet + holidays | 4.90 | best of two fixed candidates |
| LightGBM | 3.32 | best of two fixed candidates |
| N-BEATS | 5.44 | epoch 24 |
| Transformer substitute | 6.89 | epoch 18 |

## 4. Results

| Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2–7 MAPE (%) |
|:--|--:|--:|--:|--:|--:|
| LightGBM | 5.11 | 3,639 | 2,774 | 5.41 | 5.07 |
| Prophet + holidays | 8.18 | 4,900 | 4,125 | 14.61 | 7.10 |
| N-BEATS | 10.23 | 6,711 | 5,406 | 8.86 | 10.46 |
| Transformer substitute | 10.99 | 7,328 | 5,636 | 19.42 | 9.58 |
| Seasonal naive | 12.78 | 8,809 | 7,238 | 6.52 | 13.82 |
| SARIMA | 14.66 | 9,466 | 8,165 | 6.95 | 15.95 |

Jan 2–7 is the **remainder of the week**, not six working days: it includes
Saturday and Sunday. Jan 6 is a regional, not nationwide, German holiday;
no subdivision-specific holiday inputs were added. All date partitions and
calendar features use UTC, not Europe/Berlin local time.

For the winner, nominal 80% and 95% intervals cover 79.2% and 97.6% of the 168 observations. Pinball losses at q=0.1/0.5/0.9 are 470.6/1,213.3/501.3 MW (lower is better).
Coverage values in JSON/CSV are fractions (0–1); nominal levels are 0.80 and
0.95. Other models' coverage, widths and pinball scores are available in
`artifacts/probabilistic_metrics.csv`. Intervals are pointwise, not a joint
95% guarantee for the whole week.

## 5. Discussion

LightGBM wins this week with 5.11% MAPE,
a 60.0% reduction relative to seasonal naive. Its combination of calendar rules, federal holidays and load history can describe changes that a repeated week cannot. That is a plausible explanation, not a measured causal attribution: no feature ablation was run.

LightGBM was weakest on Jan 6 (11.4% MAPE). Prophet + holidays was weakest on Jan 1 (14.6% MAPE). N-BEATS was weakest on Jan 5 (16.7% MAPE). Transformer substitute was weakest on Jan 1 (19.4% MAPE). Seasonal naive was weakest on Jan 7 (24.7% MAPE). SARIMA was weakest on Jan 4 (22.8% MAPE).
Across the six models, 3 have higher MAPE on Jan 1 than on the
remaining days. Whether Jan 1 is harder is a measured per-model result,
not an assumption.
The seasonal-naive reference copies Dec 25–31, including Christmas; its
errors reflect both the forecast week's calendar and the unusual reference
week. SARIMA has no holiday indicator, while neither neural model receives
calendar covariates, so their treatment of exceptional days relies on load
history alone. Prophet's additive effects may miss changing load-profile
shape; recursive LightGBM can accumulate errors as forecast values replace
observed lags. The per-day panel localizes these failures but does not prove
their causes.

The ordering is not a universal complexity ranking. These are small,
prespecified searches with different inductive assumptions and equal data
cutoffs, not equally exhaustive optimization budgets. Daily-strided neural
training uses all years but fewer overlapping examples to meet the runtime
budget; a single seed leaves training variability unmeasured. UTC calendar
features also shift relative to German civil time at daylight-saving changes.
The test is just one winter holiday week, with strongly related hourly errors;
168 points are not 168 independent replications. Interval coverage can differ
substantially from nominal levels under this calendar shift. No confidence
claim about year-round superiority or weather sensitivity follows from this
experiment.

## 6. Recommendation

Use LightGBM as the provisional candidate for a JRC short-term load
pipeline, not as an immediately deployable winner. Before committing to one
production model, freeze its specification and evaluate on many untouched
rolling-origin weeks across seasons, holiday transitions and daylight-saving
changes, repeat neural fits across seeds, and check horizon-specific interval
calibration and operational latency. Keep seasonal naive as a monitored
fallback. These follow-up comparisons should remain univariate unless the
production objective is explicitly broadened; no additional input is needed
to reproduce this study.

## 7. Reproducibility note

Run from this directory, using the existing environment (no installation):

```sh
../../.venv/bin/python code/test_study.py
../../.venv/bin/python -u code/forecast.py
```

Seed: **206** for NumPy sampling, LightGBM and Darts/PyTorch; CPU execution,
four threads for fitting, deterministic PyTorch algorithms. The complete
fit/forecast pass took **580.9 s**; total computation including
report rendering took **582.2 s**. This is
measured execution, not authoring or peer-review time. Per-model runtimes
include selection, scratch refit and prediction, but exclude module imports.
`--report-only` rebuilds figures and tables from frozen forecast CSVs without
refitting. Full-precision metrics, all five probabilistic quantiles per hour,
validation traces, versions and the input SHA-256 are archived in `artifacts/`.

LightGBM uses the prescribed lag/rolling/calendar features, with the annual
lag set to **8,736 hours (364 days)** and all moments shifted one hour.
Its lag warm-up removes 8,736 rows from the supervised matrix only.
Intervals use 1,000 recursive simulated paths and five quantile regressors;
**32.1%** of path-hours required quantile sorting.
The point path and simulated marginal median differ by up to
**2,277 MW**.
Independent innovations can produce unrealistic within-path roughness.
SARIMA/neural intervals are Gaussian; Prophet uses 1,000 native samples.
Final SARIMA convergence: **True**; non-converged candidates are
ineligible. Full assumptions and settings are in the per-model JSON files.

Neural models use 168/168-hour chunks, stride 24, maximum 30 epochs and
validation-MAPE patience four. Refit epochs: 24
(N-BEATS), 18 (transformer).
Figure 03 uses naive = 100 with original-unit labels, not a mixed-unit axis.
All plots are 300 dpi, using a fixed reordered `tab10` palette and text labels.

Hardware (`uname -a`):

```text
Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64
```

Exact package versions and artifact digests: `artifacts/provenance.json`.
Runtime compatibility corrections and peer-review decisions:
`artifacts/review_record.md`.
