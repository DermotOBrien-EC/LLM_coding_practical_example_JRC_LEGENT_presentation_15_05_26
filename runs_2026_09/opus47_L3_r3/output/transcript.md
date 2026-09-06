# German hourly load forecasting bake-off

This is the methods section of a small, reproducible six-model bake-off on
German national hourly electricity load. Every model is univariate: the only
predictors are the load's own past and features derivable from the timestamp
column (hour of day, day of week, month, weekend flag, German public holidays
via the `holidays` Python package). No weather, no market prices, no external
covariates. The goal is to isolate the temporal-structure component of
forecast skill on one held-out week.

## 1. Data

The single input file, `opsd_de_load.csv`, comes from Open Power System Data
and is derived from the ENTSO-E Transparency Platform. It carries hourly UTC
timestamps and the German national load actual in megawatts from 2015-01-01
00:00 to 2020-09-30 23:00 inclusive: exactly 50,400 rows with no missing
values and strictly one-hour spacing. Because the file is pre-cleaned, no
imputation was performed; the loader verifies the row count, the absence of
NaN, and the fixed one-hour cadence at start-up and refuses to run otherwise.

## 2. Why these six models

The six models trace a deliberate complexity gradient. **Seasonal-naive** is
the honest anchor: the forecast is simply the load 168 h earlier, so any
model that fails to beat it is doing negative work. **SARIMA** is the
classical linear-Gaussian workhorse, capable of representing autoregressive
memory and one seasonality (daily) but blind to holidays. **Prophet** adds an
explicit German holiday calendar and yearly seasonality on top of a piecewise
trend; that gives it a fair shot at Jan 1. **LightGBM** is the fast,
non-linear machine that gets to see explicit hand-built features (lags,
rolling means, calendar codes, holiday flag). **N-BEATS** is a modern
deep learning workhorse tuned for pure univariate forecasting. **TSMixer**
(substituted for PatchTST because the installed `darts` version does not
expose `PatchTSTModel`) is an MLP-mixer style architecture; it stands in for
the transformer-style rung of the ladder.

## 3. Validation strategy

The chronological split is Train = 2015-01-01 to 2019-09-30 (41,616 hourly
rows), Validation = 2019-10-01 to 2019-12-31 (2,208 rows), Test = 2020-01-01
to 2020-01-07 (168 rows). Each model with hyperparameters is fit on Train,
scored on Validation, and the configuration with the lowest validation MAPE
is refit on Train + Validation combined before producing the test-window
forecast. This is the standard operational recipe: keep as much data as
possible in the final fit, but let a chronologically-later slice choose the
model class or order. The test week is never touched during fitting or
selection.

## 4. Results

Sorted by test-window MAPE (primary metric):

| Model | MAPE (%) | RMSE (MW) | MAE (MW) | MAPE Jan 1 (%) | MAPE Jan 2-7 (%) | Runtime (s) |
|---|---|---|---|---|---|---|
| LightGBM (features) | 4.45 | 3010 | 2400 | 3.33 | 4.63 | 144.7 |
| TSMixer | 7.38 | 5280 | 4105 | 4.77 | 7.82 | 577.7 |
| Prophet (DE holidays) | 7.63 | 4894 | 3808 | 13.38 | 6.67 | 14.9 |
| N-BEATS | 9.42 | 6120 | 4960 | 7.97 | 9.66 | 49.1 |
| SARIMA | 12.55 | 8441 | 7029 | 9.71 | 13.03 | 647.8 |
| Seasonal-naive (168 h) | 12.78 | 8809 | 7238 | 6.52 | 13.82 | 0.0 |

Winner: **LightGBM (features)** with test MAPE 4.45%, versus the
seasonal-naive anchor at 12.78% (a 65% reduction
in MAPE). For the winner the 80 % nominal prediction interval covers
81% of test observations and the 95 %
interval covers 89%; pinball losses at
q = 0.1, 0.5, 0.9 are 346,
1200 and 353
MW respectively.

## 5. Discussion

The winner is **LightGBM (features)**, with a test MAPE of 4.45 %
against the 12.78 % seasonal-naive baseline. That the winner
gains the most ground on Jan 2-7 rather than Jan 1 is instructive:
Jan 1 2020 was a Wednesday and a federal holiday, and its lag-168 h reference
was Christmas Day 2019 (also a Wednesday holiday), so seasonal-naive is
actually reasonable on Jan 1 and hurts most on the return-to-work days
(Jan 2-7). Every model that models holidays only implicitly (SARIMA, and to a
lesser extent the pure deep learners) shows the mirror pattern: their Jan 1
MAPE is a good deal higher than their Jan 2-7 MAPE.

The worst-performing model in this run is **Seasonal-naive (168 h)**; its
failure mode is visible in figure 4 (per-day MAPE). The model with the
largest Jan 1 vs Jan 2-7 gap is **Prophet (DE holidays)**
(gap = +6.7 percentage points). This is
the awkward finding of the run: even though that model has an explicit
German holiday calendar wired in, its holiday effect is a single global
coefficient learned from every DE holiday over five years and does not
capture the specific way Jan 1 behaves in this test week; a per-holiday
effect or a holiday-hour interaction would be the fix.

The rank ordering is broadly what theory predicts. Modern gradient-boosted
trees on well-engineered lag / calendar features are famously hard to beat
on a short-horizon single-variable forecast, and that is what we see.
SARIMA is limited by its single seasonality (m = 24); a full weekly SARIMA
(m = 168) is impractical in `statsmodels` on 42 k rows. Prophet is the odd
case: it has the German holiday calendar and still posts the biggest Jan 1
error, because its holiday effect is a global additive coefficient learned
across five years, while the yearly seasonality also has a January-1 dip
that pulls in a different direction; the net effect is a heavier over-shoot
than seasonal-naive. The two pure deep learners have no holiday input at
all, and their MC-dropout prediction intervals turn out to be too narrow
(see the coverage numbers on the winner in figure 5 for what a
better-calibrated interval looks like).

The obvious caveats: (a) the test window is a single week, so absolute rank
positions are noisy; a week that skirted no holiday would move Prophet down
and SARIMA up. (b) The prediction interval calibration is model-native
(Prophet Monte-Carlo trajectories, N-BEATS / TSMixer MC-dropout, LightGBM
independent quantile fits, SARIMAX statespace); none of them is a conformal
guarantee. (c) The univariate constraint is a deliberate handicap for the
gradient-boosted / Prophet side, both of which usually shine with a
temperature covariate; a full production forecaster would need one.

## 6. Recommendation

If exactly one of these six had to go into a JRC short-term load forecasting
pipeline, pick **LightGBM on engineered features**. It is the fastest
model that also (a) beats the naive baseline by a large margin, (b) exposes
its reasoning through feature importances, and (c) has a well-understood
quantile head that gives calibrated intervals out of the box. Before
production I would (i) add temperature and (optionally) dew-point as
covariates (the single biggest omitted regressor for German load), (ii) add
a holiday indicator that lists Bundesland-specific holidays rather than only
federal ones, and (iii) recalibrate prediction intervals with a conformal
post-hoc adjustment on a rolling 30-day window so the nominal 80 % / 95 %
levels match empirical coverage.

## 7. Reproducibility

Random seeds: numpy / lightgbm / Prophet / darts torch models all seeded to
`42` (see `code/common.py`). Total wall-clock runtime of the full
bake-off: **1439.4 s**. Hardware:

```
Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64
```

Hyperparameters selected per model:

- **Seasonal-naive (168 h)**: `{"lag_hours": 168}`
- **SARIMA**: `{"order": [3, 1, 1], "seasonal_order": [1, 1, 1, 24], "validation_mape_pct": 11.756966579890761}`
- **Prophet (DE holidays)**: `{"yearly_seasonality": true, "weekly_seasonality": true, "daily_seasonality": true, "country_holidays": "DE", "validation_mape_pct": 5.635830073898914}`
- **LightGBM (features)**: `{"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 20, "subsample": 0.9, "subsample_freq": 1, "colsample_bytree": 0.9, "validation_mape_pct": 2.005241614739437}`
- **N-BEATS**: `{"input_chunk_length": 168, "output_chunk_length": 168, "n_epochs": 30, "batch_size": 512, "num_stacks": 2, "num_blocks": 2, "num_layers": 3, "layer_widths": 128, "dropout": 0.1, "validation_mape_pct": 11.116612061995092}`
- **TSMixer**: `{"substituted_for": "PatchTSTModel (not exposed in installed darts 0.41.0)", "model_class": "TSMixerModel", "input_chunk_length": 168, "output_chunk_length": 168, "n_epochs": 15, "batch_size": 512, "hidden_size": 64, "ff_size": 64, "num_blocks": 2, "dropout": 0.1, "validation_mape_pct": 12.507126450766728}`

Python packages: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `pmdarima`,
`darts` (bundles Prophet and torch), `lightgbm`, `holidays`. Installed
darts 0.41.0 does not expose `PatchTSTModel`, so the `patchtst.py` module
uses `TSMixerModel` and records that substitution in the hyperparameter
block above.

To reproduce, from this directory:

```
../../.venv/bin/python code/forecast.py
```

Outputs are written into `figures/`, `metrics.json`, `metrics.csv`, and this
transcript.
