# German hourly load: a six-model forecasting bake-off

A held-out week of German national electricity demand (1 to 7 January 2020, 168
hours) forecast by six models of increasing complexity, all fitted on the same
history, all judged on the same numbers.

**Headline: LightGBM wins with a test MAPE of 4.99 percent**, against 12.78
percent for the seasonal-naive baseline (61 percent lower error).

---

## 1. Data

The series is hourly German national electricity load in megawatts, taken from
Open Power System Data, who in turn derive it from the ENTSO-E Transparency
Platform. It runs from 2015-01-01 00:00 UTC to 2020-09-30 23:00 UTC: 50,400
hourly observations with no missing values and no missing hours. The loader in
`code/common.py` checks all three of those claims on every run and raises if
any of them fails, so nothing was imputed, interpolated or dropped because
nothing needed to be. The timestamps are UTC throughout. Calendar features are
derived from the Berlin local time that the same instants correspond to,
because German demand follows the local clock (including summer time), not UTC.

## 2. Why these six models

The six are a deliberate ladder of complexity, and each rung is supposed to add
one specific thing.

The **seasonal-naive** forecast says the load at any hour equals the load at
the same hour one week earlier. It has no parameters and knows nothing. It is
the anchor: any model that cannot beat it is not paying for itself.

**SARIMA** adds a statistical description of short memory: how this hour
relates to the last few hours and to the same hour yesterday. It brings
principled, analytically derived prediction intervals.

**Prophet** adds explicit structure that a human names in advance: a bending
trend, a daily shape, a weekly shape, a yearly shape, and a separate effect for
every German public holiday. It is the only model in the set that is *told*
what 1 January is.

**LightGBM on engineered features** adds the ability to learn interactions. It
sees the same calendar information Prophet does, plus lags of the load itself
(24 hours, one week, and roughly one year back) and rolling averages, and
learns from data how they combine.

**N-BEATS** and the **TSMixer** entry (see the substitution note below) add
capacity and remove prior knowledge. Both read 168 raw hours and emit the next
168 in one shot, with no calendar input at all. Anything they know about
weekends or holidays they had to infer from the numbers.

**Substitution note.** The brief asked for `darts.models.PatchTSTModel` or
`darts.models.TSMixerModel`. darts 0.41.0 in this environment exposes
`TSMixerModel` but not `PatchTSTModel`, so TSMixer occupies that slot. It is an
all-MLP architecture rather than a transformer, and it is reported under the
name `patchtst` in `metrics.json` only because the schema fixes that label.

## 3. Validation strategy

Three windows, cut once and never moved:

| Window | Range (UTC) | Hours |
|---|---|---:|
| Train | 2015-01-01 00:00 to 2019-09-30 23:00 | 41,616 |
| Validation | 2019-10-01 00:00 to 2019-12-31 23:00 | 2,208 |
| Test | 2020-01-01 00:00 to 2020-01-07 23:00 | 168 |

Every model with hyperparameters was fitted on Train alone and scored on
Validation. The scoring is not one-step-ahead error: the validation window
holds thirteen whole weeks, and each candidate was asked to forecast each of
those weeks 168 hours ahead from a cold start, exactly the job it would be
given on the test window. Averaging over thirteen weeks rather than one keeps
the choice from being decided by a single unusual week. (SARIMA is scored on
four evenly spaced weeks rather than thirteen, purely to keep its sweep of
twelve candidate specifications inside a few minutes.)

Once a configuration was chosen it was refitted from scratch on Train plus
Validation combined, that is on everything from 2015-01-01 to 2019-12-31, and
only then pointed at the test week. Refitting is not optional: the last quarter
of 2019 is the most informative data available about January 2020, and throwing
it away to preserve a tidy split would make the forecast worse for no
methodological gain. The test window itself was never read by any fitting or
selection code; the naive baseline's own guard raises if its lags would reach
into it.

Configurations chosen:

| Model | Chosen configuration | Selected on |
|---|---|---|
| Seasonal naive | `seasonal_lag_hours`=168 | no hyperparameters to select |
| SARIMA | `order`=[2, 1, 2], `seasonal_order`=[1, 1, 1, 24], `training_window_hours`=2184 | mean MAPE over 4 held-out validation weeks |
| Prophet | `changepoint_prior_scale`=0.05, `seasonality_mode`=additive, `country_holidays`=DE, `uncertainty_samples`=500 | mean MAPE over 13 held-out validation weeks |
| LightGBM | `n_estimators`=800, `learning_rate`=0.05, `num_leaves`=63, `forecast_strategy`=recursive, 168 one-step-ahead calls | mean MAPE over 13 held-out validation weeks |
| N-BEATS | `learning_rate`=0.001, `batch_size`=1024, `input_chunk_length`=168, `output_chunk_length`=168, `epochs_used_for_final_fit`=29, `max_epochs`=30, `early_stopping_patience`=5, `likelihood`=QuantileRegression(0.025, 0.1, 0.5, 0.9, 0.975), `scaler`=min-max on the training sample | mean MAPE over 13 held-out validation weeks |
| TSMixer (PatchTST slot) | `learning_rate`=0.001, `batch_size`=1024, `input_chunk_length`=168, `output_chunk_length`=168, `epochs_used_for_final_fit`=25, `max_epochs`=30, `early_stopping_patience`=5, `likelihood`=QuantileRegression(0.025, 0.1, 0.5, 0.9, 0.975), `scaler`=min-max on the training sample, `darts_class`=TSMixerModel | mean MAPE over 13 held-out validation weeks |

## 4. Results

All figures are test-window numbers on the 168 held-out hours. "1 Jan" is the
24 UTC hours of New Year's Day, a German federal public holiday falling on a
Wednesday; "2-7 Jan" is the remaining 144 hours, which are four ordinary
working days and a weekend.

| Rank | Model | MAPE (%) | RMSE (MW) | MAE (MW) | MAPE 1 Jan (%) | MAPE 2-7 Jan (%) | Runtime (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | LightGBM | 4.99 | 3,594 | 2,793 | 2.39 | 5.43 | 246 |
| 2 | Prophet | 7.61 | 4,878 | 3,798 | 13.28 | 6.67 | 52 |
| 3 | TSMixer (PatchTST slot) | 10.73 | 8,023 | 6,194 | 4.90 | 11.70 | 973 |
| 4 | N-BEATS | 11.13 | 7,368 | 5,732 | 7.58 | 11.72 | 450 |
| 5 | Seasonal naive | 12.78 | 8,809 | 7,238 | 6.52 | 13.82 | 0 |
| 6 | SARIMA | 13.24 | 8,951 | 7,452 | 8.71 | 14.00 | 111 |

Per-day MAPE for every model is in `figures/04_per_day_mape.png` and in the
per-day columns of `metrics.csv`.

For the winning model, LightGBM:

| Quantity | Nominal | Actual |
|---|---:|---:|
| 80% prediction-interval coverage | 80.0% | 56.5% |
| 95% prediction-interval coverage | 95.0% | 92.9% |

| Pinball loss (MW) | q = 0.1 | q = 0.5 | q = 0.9 |
|---|---:|---:|---:|
| LightGBM | 731 | 1,449 | 437 |

Both bands are drawn over the observed week in
`figures/05_winner_with_intervals.png`, and section 5 reads the miscalibration.

## 5. Discussion

**LightGBM wins** at 4.99 percent MAPE against 12.78 percent for the
seasonal-naive anchor. LightGBM, Prophet, TSMixer (PatchTST slot) and N-BEATS
beat the anchor; SARIMA did not.

The ordering is roughly what theory predicts, but the reason is narrower than
"more complex is better": what separates these models is whether the calendar
can change the *shape* of a day, not just its level. SARIMA, last, fails
structurally rather than through bad tuning. Its seasonal period is fixed at 24
hours because a 168-hour state-space term is impractical to fit, so it
reproduces the daily shape and cannot say that Saturday differs from Tuesday.
Over a week-long horizon that is most of the signal, and its forecast settles
into a repeated average day (figure 02).

The holiday gives the most useful result: **carrying a holiday feature is not
the same as being able to use it**. Prophet knows 1 January is a holiday and
still scores 13.28 percent on it, its worst day of the seven, against 6.67
percent on the rest of the week. Figure 08 shows why: its holiday term is one
constant, about -10,500 MW, applied to all 24 hours alike, whereas New Year is
a differently shaped day, not a uniformly lower Wednesday. LightGBM, given the
same flag, scores 2.39 percent, because a tree can split on the flag and then
on hour of day. The baseline's respectable 6.52 percent there is an accident of
copying 25 to 31 December, paid for on the ordinary days (13.82 percent).

The neural models see no calendar, only the 168 hours before the forecast,
which here are the Christmas shutdown. N-BEATS keeps the level but loses the
weekly phase, over-predicting Sunday 5 January by about 14,000 MW (29.6
percent) after validating at 4.78 percent over thirteen ordinary autumn weeks.
The winner's own worst day is 6 January (11.1 percent, over by about 6,500 MW):
Epiphany, a holiday in three federal states but not nationally, so
`holidays.Germany()` misses it. That calls for a regional holiday calendar, not
a better model.

Its intervals are too narrow: the nominal 80 percent band held 57 percent of
the actuals, the 95 percent band 93 percent. The quantile models bracket a
one-hour-ahead error, yet are asked to bracket a week-ahead recursive forecast.

Caveats: one window is a single draw, and an unusually hard one that flatters
holiday-aware models; temperature is absent by design, so these are not
production error levels; and N-BEATS stopped at the brief's epoch cap, so that
score is a floor rather than a ceiling.

## 6. Recommendation

If exactly one of these had to go into a JRC short-term load forecasting
pipeline, it would be **LightGBM**, and accuracy is only part of the reason. It
retrains in minutes on a laptop CPU; its features are readable by a domain
expert, who can therefore challenge them; its failure modes are diagnosable
from the importance and residual plots rather than opaque (this run's largest
error was traced to a specific missing regional holiday in an afternoon); and
adding a driver later is a new column, not a new architecture. Prophet is the
natural challenger to keep in any regular re-evaluation, and it is the better
choice if interpretable components matter more than accuracy.

Five things would have to happen before it is trusted in production. First,
replace this single test week with a rolling-origin backtest of at least
fifty-two weekly origins across two or more years: one week cannot separate a
better model from a luckier one, and this particular week is not
representative. Second, add temperature. It is the largest omitted driver of
German winter demand; the univariate constraint here was deliberate, to isolate
temporal structure, and it is not a constraint a production system should keep.
Third, replace the federal holiday flag with a load-weighted regional calendar
covering the sixteen Laender, plus school holidays and bridging days, which
would have caught the 6 January miss. Fourth, fix the interval calibration:
either train the quantile models on the recursive multi-step errors the system
will actually make, or wrap the point forecast in conformal prediction
calibrated separately for each lead time, so a day-seven interval is honestly
wider than a day-one interval. Fifth, decide the recursion explicitly: either
train one model per horizon step, which removes the compounding at the cost of
168 models, or keep the recursion and publish how error grows with lead time so
that nobody reads a day-seven number with day-one confidence.

## 7. Reproducibility

- **Seed.** A single seed, `SEED = 42` in `code/common.py`, is passed to
  LightGBM, Prophet, N-BEATS and TSMixer, and `torch.manual_seed` is called
  before each neural fit. The naive baseline and SARIMA are deterministic. Two
  independent full runs on this machine produced byte-identical metrics for all
  six models, so nothing here depends on run order or on the fit cache.
  Bit-identical results on different hardware are not guaranteed: threaded
  floating-point reduction in the neural fits can move the last decimals.
- **How to reproduce.** `../../.venv/bin/python code/forecast.py`, run from
  this directory. It rewrites `metrics.json`, `metrics.csv`, every figure and
  this file. Each model is fitted in its own subprocess, so a failure in one
  does not take the study down with it.
- **Total wall-clock runtime.** 30.6 minutes for the full run, including all
  hyperparameter sweeps, all refits and all figures. Per-model runtimes are in
  the results table above.
- **Software.** Python 3.12.13, pandas 2.3.3, numpy 2.4.4, darts 0.41.0
  (Prophet, N-BEATS, TSMixer), plus statsmodels, lightgbm and holidays. Neural
  training ran on CPU, not the machine's GPU, so that the run reproduces on
  machines without one.
- **Files.** Model code is one module per model under `code/`, with shared
  loading, metrics and plotting style in `code/common.py` and the orchestration
  in `code/forecast.py`.
- **Hardware.** `uname -a` reports:

  ```
  Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64
  ```
