# Forecasting bake-off: German hourly electricity load, one week ahead

Six models, one held-out week (1 to 7 January 2020, 168 hours), same data,
same metrics. Everything below is produced by `code/forecast.py`; the
per-model forecasts it scores are kept in `forecasts/` so a reader can
re-check any number without re-training.

## 1. Data

The series is the German national electricity load in megawatts from Open
Power System Data, which derives it from the ENTSO-E Transparency Platform
(`opsd_de_load.csv`, column `DE_load_actual_entsoe_transparency`). It runs
hourly in UTC from 2015-01-01 00:00 to 2020-09-30 23:00: 50,400 rows, mean
55.5 GW, range 31.3 to 77.5 GW. The loader in `code/common.py` re-checks the
three stated guarantees on every run and raises if any fails: exactly
50,400 rows, no missing values, and a single one-hour step between every
pair of consecutive timestamps. All three held, so no imputation and no
dropping of hours was needed, and no anomalies were found. The only
preprocessing is a change of representation: timestamps are made
timezone-naive UTC for the libraries, and calendar features (hour,
weekday, holiday) are read in Berlin local time because demand follows the
local clock. Only the load and features derived from its own timestamps are
used; no weather, price or other external data enters any model.

## 2. Why these six models

The six form a complexity gradient, and each brings one idea. The
seasonal-naive baseline (load one week earlier) is the floor every model
must beat and is also the implicit "last week" heuristic of a control room.
SARIMA is the classical statistical model: a small set of coefficients
describing how the recent past propagates, with a 24-hour season. Prophet is
a curve fit (trend plus daily, weekly and yearly cycles plus a fixed effect
per German public holiday); it has no memory of the last few hours, which
makes it robust to a strange final week but blind to what the last days are
doing. LightGBM is a gradient-boosted tree ensemble on twelve engineered
numbers per hour (calendar facts, the load 24 h, 168 h and 364 days earlier,
and moving averages and spreads of the recent past); it is the workhorse of
applied load forecasting because it combines calendar knowledge with recent
memory. N-BEATS and TSMixer are deep networks that read the last 168 hours
and write the next 168 in one shot, learning the daily and weekly shapes
from 41,000 sliding training windows without any calendar input. TSMixer
occupies the "PatchTST" slot because the installed darts (0.41.0) does not
ship `PatchTSTModel`; the task names `TSMixerModel` as the alternative, and
it is used as such. It is not a transformer (it mixes across time and across
features with small fully connected layers), which is recorded here as a
substitution. The darts `TransformerModel` was not used because it is a
vanilla encoder-decoder that is slow and weak at 168-step direct forecasting.

## 3. Validation strategy

Train is 2015-01-01 00:00 to 2019-09-30 23:00 (41,616 hours), Validation is
2019-10-01 to 2019-12-31 (2,208 hours), Test is 2020-01-01 to 2020-01-07
(168 hours). The 6,408 hours after the test week are never used. Every
model with settings to choose is scored on the same validation protocol:
its parameters are estimated on Train only, then from each of seven origins
two weeks apart (1 Oct, 15 Oct, 29 Oct, 12 Nov, 26 Nov, 10 Dec, 24 Dec 2019)
it forecasts the next 168 hours using only the history before that origin,
and the seven MAPEs are averaged. The last origin covers Christmas week, the
closest thing the validation window has to the New Year test week. SARIMA
applies its Train-estimated coefficients to each origin's history without
re-estimation; LightGBM runs its recursive forecast from each origin;
Prophet, whose forecast does not depend on the origin, is scored on the
same seven weeks from one fit; the two deep networks pick their epoch count
by early stopping on the per-epoch validation MAPE (computed on random
draws from the quantile head, so slightly pessimistic, and used only to
decide when to stop), and their learning rate by the same seven-origin
median MAPE as everyone else. The winning configuration of every model is
then refitted from scratch on Train + Validation (43,824 hours, ending
31 December 2019) and issues the single 168-hour test forecast. The refit
is standard practice: the final model should use all data before the
forecast origin, and the three months of autumn 2019 are the most relevant
history the test week has. For the deep networks the refit runs for exactly
the epoch count found on validation, since no validation data is left to
stop on.

## 4. Results

Test week, sorted by MAPE. "Validation MAPE" is the seven-origin score used
for model selection (it is on autumn 2019, so not directly comparable to the
test column). Runtimes are wall-clock seconds for selection, refit and
forecast together.

| Model | Test MAPE (%) | RMSE (MW) | MAE (MW) | MAPE 1 Jan, holiday (%) | MAPE 2 to 7 Jan (%) | Validation MAPE (%) | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (features) | 5.03 | 3712 | 2801 | 2.08 | 5.52 | 3.27 | 114 |
| Prophet (DE holidays) | 7.61 | 4878 | 3798 | 13.28 | 6.67 | 6.35 | 30 |
| TSMixer (PatchTST slot) | 7.62 | 6162 | 4384 | 3.89 | 8.24 | 5.31 | 559 |
| N-BEATS | 10.25 | 6833 | 5475 | 4.09 | 11.28 | 5.71 | 623 |
| SARIMA | 12.34 | 8744 | 7049 | 4.87 | 13.59 | 4.63 | 25 |
| Seasonal naive (t-168 h) | 12.78 | 8809 | 7238 | 6.52 | 13.82 | 5.82 | 0 |

Per-day MAPE (%), UTC days:

| Model | Wed 1 Jan (holiday) | Thu 2 Jan | Fri 3 Jan | Sat 4 Jan | Sun 5 Jan | Mon 6 Jan | Tue 7 Jan |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM (features) | 2.08 | 1.85 | 4.26 | 2.89 | 5.03 | 12.22 | 6.84 |
| Prophet (DE holidays) | 13.28 | 6.80 | 3.57 | 11.77 | 9.03 | 2.88 | 5.98 |
| TSMixer (PatchTST slot) | 3.89 | 13.52 | 5.30 | 2.78 | 3.38 | 6.36 | 18.12 |
| N-BEATS | 4.09 | 14.29 | 4.19 | 10.43 | 22.79 | 5.10 | 10.86 |
| SARIMA | 4.87 | 18.62 | 12.02 | 9.17 | 5.39 | 11.67 | 24.65 |
| Seasonal naive (t-168 h) | 6.52 | 19.17 | 12.35 | 9.40 | 5.54 | 11.77 | 24.71 |

Winner (lowest test MAPE): LightGBM.

| Winner statistic | Nominal | Actual |
|---|---:|---:|
| 80 % prediction-interval coverage | 80 % | 57.1 % |
| 95 % prediction-interval coverage | 95 % | 94.0 % |
| Pinball loss, quantile 0.1 (MW) | | 779 |
| Pinball loss, quantile 0.5 (MW) | | 1337 |
| Pinball loss, quantile 0.9 (MW) | | 441 |

For context, the same coverage check on the other probabilistic models
(80 % / 95 % bands): TSMixer 79.8 % / 95.2 %, Prophet 64.3 % / 83.3 %,
N-BEATS 42.3 % / 73.2 %, SARIMA 46.4 % / 69.0 %.

Selected settings (full grids and every candidate's score are in
`forecasts/<model>.json`):

- SARIMA: weekly-differenced family, ARMA(2,0,2)(1,0,1)_24 on the load minus
  its value 168 h earlier, fitted on the last 12 weeks, no regressor. The
  grid held 30 candidates in two families: "daily" SARIMA(p,d,q)(P,1,Q)_24 on
  the raw load with optional Saturday and Sunday-or-holiday dummies, and the
  weekly-differenced family with an optional week-on-week holiday-change
  regressor. The best daily-family candidate scored 9.63 % on validation
  against 4.63 % for the winner. The holiday regressor changed nothing at
  6 and 12 weeks because those windows (July to September 2019) contain no
  German public holiday, so its coefficient is unidentifiable; a 26-week
  window was added for that reason and scored 4.97 % with the regressor
  versus 5.02 % without, still behind the 12-week winner. One 26-week
  candidate, ARMA(2,0,2) with the regressor, failed with an LU decomposition
  error and was excluded.
- Prophet: additive seasonality, changepoint prior 0.05, seasonality prior
  10, German holidays, daily + weekly + yearly cycles (4 candidates, all
  within 0.05 percentage points of each other on validation). Its fitted
  effect for 1 January is about -10.5 GW. Holidays are dated in UTC, one
  hour off Berlin time; the effect on the scores is negligible.
- LightGBM: 127 leaves, 400 trees, learning rate 0.05, minimum 50 rows per
  leaf, 80 % row and column subsampling (4 candidates). The one-year lag is
  taken at 364 days (8,736 h) rather than 8,760 h so that it lands on the
  same weekday. Rolling means and spreads use hours strictly before the
  target hour, because the target hour's own load is what is being
  predicted. The forecast is recursive: each predicted hour is appended to
  the history and feeds the 24-hour lag and rolling features of later
  hours, so no test observation is ever read. Intervals come from five
  extra quantile-objective models (0.025, 0.1, 0.5, 0.9, 0.975) evaluated on
  the same recursive feature rows; the point forecast is the squared-error
  model, which is why the point line can sit at the edge of its own band.
- N-BEATS: darts default generic architecture (30 stacks, width 256),
  learning rate 3e-4, 22 epochs; TSMixer: hidden size 64, 2 blocks, dropout
  0.1, learning rate 1e-3, 11 epochs. Both: 168 h in, 168 h out, batch 256,
  quantile-regression head with the five quantiles above (sorted per hour
  to remove the occasional crossing), series divided by 50,000 MW before
  training, CPU only. For N-BEATS the two learning rates disagreed between
  the early-stopping metric and the seven-origin median MAPE (1e-3: 5.73 %
  vs 6.66 %; 3e-4: 6.75 % vs 5.71 %); the seven-origin score decided, as for
  every other model.

## 5. Discussion

LightGBM won clearly, 5.0 % MAPE against 7.6 % for the next two models,
and it also won validation (3.3 %). It alone holds both kinds of
knowledge the week demands: the calendar (2.1 % on the 1 January holiday,
the best of all six) and recent memory (the 168-hour and 364-day lags
carry 89 % of its split gain, figure 07). Its failures are calendar gaps: from 2 January on it
over-predicted every day, by 2.5 GW on average (figure 06), worst on
Monday 6 January (12.2 %), Epiphany, a holiday in three southern and
eastern states but not a federal one, so the holiday flag is zero.

Seasonal naive and SARIMA failed for one reason: their base week was 25 to
31 December, all holidays and bridge days. They under-predict working days
by about 10 GW, with 19 to 25 % errors on 2 and 7 January. The
weekly-differenced SARIMA is "naive plus a daily correction", so it beats
the naive on validation (4.6 % vs 5.8 %) but inherits the collapse.

The deep networks' only input is the same distorted Christmas week, so
they reproduce a week with too little contrast: N-BEATS
forecast a weekday-like Sunday 5 January (22.8 %), TSMixer a holiday-like
Thursday 2 January (13.5 %) and a too-low Tuesday 7 January (18.1 %).
Prophet is the mirror image, calendar without recent memory. Its holiday
term has the right size, but its daily profile on 1 January is wrong
(13.3 %), and its yearly Fourier terms carry a dip
at the turn of the year that drags down the 4 and 5 January weekend.

The rank order matches expectation: feature-based boosting first,
calendar-aware curve fit and deep nets in the middle, memory-only models
last on a holiday week. The surprises are how badly a well-validated SARIMA
does on an atypical base week, and how little the deep networks gained
from 41,000 training windows without calendar covariates. On uncertainty, LightGBM's 95 % band is honest (94.0 %) but its 80 % band is
overconfident (57.1 %), because the quantile models learned ordinary-week
spreads and cannot see the post-holiday bias. TSMixer's bands were best calibrated
(79.8 % and 95.2 %).

Caveats: one unusual 168-hour week cannot rank models with confidence;
validation and test rankings disagree for the base-week-dependent models;
MAPE inflates errors on a low-load week; the deep models had a minimal
search (two learning rates).

## 6. Recommendation

For a JRC production short-term load forecasting pipeline, LightGBM on
engineered features is the one to take: it won on both validation and test,
it trains in under two minutes on a laptop, it exposes why it errs (feature
importance, per-feature residuals), and its failures were calendar gaps
rather than model gaps. Before deployment I would do four things, in order.
First, enrich the calendar with a priori knowledge the current flags miss:
regional holidays weighted by each state's share of load, bridge days, the
Christmas-to-Epiphany period, and school holidays. Second, replace the
recursive scheme with one model per horizon step (or a direct multi-output
model) so that predictions are never fed back as lags, and recalibrate the
intervals with conformal adjustment on rolling-origin residuals, since the
80 % band covered only 57 %. Third, evaluate on a rolling basis over at
least a full year of weekly origins, reporting per-season and per-holiday
skill, before trusting any single number. Fourth, only then relax the
univariate constraint and add day-ahead weather forecasts as features,
which is where the remaining error on ordinary weeks mostly lives.

## 7. Reproducibility note

- Random seed 42 everywhere (`random`, NumPy, torch, LightGBM, darts
  `random_state`, Prophet). Deep networks train on CPU to keep results
  deterministic; MPS was benchmarked and offered no speed-up on this data.
- Total wall-clock runtime of `../../.venv/bin/python code/forecast.py`
  (all six models, selection, refit, forecast, figures): 1354 s (22.6 min).
  Per-model runtimes are in the results table and `metrics.json`. Re-running
  the orchestrator reproduced every test MAPE to the printed precision.
- Hardware: Apple M5 Pro, 15 CPU cores, 24 GB memory. `uname -a`:
  `Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0:
  Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050
  arm64`
- Software (project venv at `../../.venv`, Python 3.12): pandas 2.3.3,
  numpy 2.4.4, matplotlib 3.10.9, statsmodels 0.14.6, pmdarima 2.1.1,
  darts 0.41.0, prophet 1.3.0, lightgbm 4.6.0, holidays 0.96, torch 2.10.0,
  pytorch-lightning 2.5.2. No packages were installed.
- Layout: `code/` (one module per model, `common.py` shared, `forecast.py`
  orchestrator; `forecast.py --from-cache` rebuilds metrics and figures from
  `forecasts/` without re-training, `--models a,b` re-runs a subset),
  `figures/` (eight PNGs at 300 dpi; 07 and 08 are produced because LightGBM
  and Prophet finished first and second), `metrics.json`, `metrics.csv`,
  `forecasts/` (per-model test forecasts with quantiles, selected settings,
  full validation tables, and `summary_tables.md`).
- Note on `code/prophet.py`: the task's file name shadows the `prophet`
  package, so that module imports the real package with its own directory
  temporarily removed from the import path, and the orchestrator loads it
  under the name `bakeoff_prophet`. The other five modules are imported
  normally.
