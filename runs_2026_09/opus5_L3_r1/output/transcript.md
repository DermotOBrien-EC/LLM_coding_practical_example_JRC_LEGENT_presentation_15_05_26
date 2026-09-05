# Forecasting German hourly electricity load: a six-model bake-off

A comparison of six forecasting methods on one held-out week of German
national load, 1 to 7 January 2020. All six see exactly the same data, are
selected on the same validation window, and are scored on the same 168
hours.

## 1. Data

The single input is `opsd_de_load.csv`, an hourly series of German national
electricity load in megawatts, published by Open Power System Data and
derived from the ENTSO-E Transparency Platform. It runs from 2015-01-01
00:00 UTC to 2020-09-30 23:00 UTC. We checked the file rather than trusting
it: it has exactly 50,400 rows, which is precisely the number of hours in
that span, the timestamps form an unbroken hourly grid with no duplicates,
and there are no missing values. Load ranges from 31,307 MW to 77,549 MW
with a mean of 55,492 MW. Because there is nothing missing, no imputation
was needed and none was done. The loader in `code/common.py` re-checks all
of this on every run and raises an error rather than silently repairing
anything, so a future change to the file cannot pass unnoticed.

Two conventions are worth stating because they affect the numbers. First,
calendar features (hour of day, weekday, month, weekend flag, German public
holiday flag) are derived from Berlin local time, not UTC. People switch
appliances on according to the clock on the wall, and Germany moves that
clock twice a year, so deriving "hour of day" from UTC would smear the daily
shape by an hour between summer and winter. Second, the per-day error
breakdown slices the test window by UTC date, which divides it into seven
blocks of exactly 24 hours. In January the two conventions differ by one
hour, which is immaterial at this resolution.

## 2. Why these six models

The six form a deliberate ladder of complexity, and the point of the
exercise is to find out where on that ladder the useful skill actually
appears.

The **seasonal naive** baseline forecasts each hour as whatever was observed
168 hours earlier. It has no parameters and encodes one idea: this week
looks like last week. Everything else has to earn its complexity by beating
it. **SARIMA** is the classical statistical answer, modelling the load as a
process whose next value depends on recent values and recent errors, and it
comes with analytic prediction intervals. **Prophet** is a pure calendar
model: it adds up a slow trend, a daily shape, a weekly shape, a yearly
shape and a public-holiday effect, and never looks at recent load at all.
**LightGBM** is the workhorse of applied forecasting: it turns every hour
into a row of a table with calendar columns and lagged-load columns and
learns a regression, which lets a human inject domain knowledge (this is a
holiday, this is a Sunday) directly as a feature. **N-BEATS** and the sixth
entry are the deep-learning end of the ladder: they read a raw window of 168
past values and emit the next 168, with no calendar and no human-chosen
lags, on the premise that a large network can discover that structure by
itself.

**Substitution, recorded as required.** The study asks for PatchTST. The
installed darts version (0.41.0) does not expose a `PatchTSTModel`; the
classes available in that family are `TSMixerModel`, `TFTModel` and
`TransformerModel`. We used `darts.models.TSMixerModel`, which the study
text names as the permitted alternative. TSMixer shares PatchTST's central
idea, mixing information across time positions with cheap feed-forward
layers instead of recurrence. Throughout this report and in the figures the
model is labelled "TSMixer"; in `metrics.json` it occupies the `patchtst`
slot required by the schema, and the field `model_class_used` in its
hyperparameters records the real class, so the slot name cannot mislead.

## 3. Validation strategy

The data is cut into three consecutive blocks. **Train** is 2015-01-01 to
2019-09-30 (41,616 hours). **Validation** is 2019-10-01 to 2019-12-31 (2,208
hours). **Test** is 2020-01-01 to 2020-01-07 (168 hours) and was not read by
any fitting or selection step.

Every model with hyperparameters was fitted on Train alone and then scored
on Validation. Scoring is deliberately shaped like the real task: the
validation quarter is forecast in thirteen back-to-back blocks of 168 hours,
each started from observed data and then run out to the end of the block
with no further observations. Scoring one long 2,208-hour forecast instead
would have measured a harder and different job, and would have selected
hyperparameters for it. (For Prophet the two are identical, because its
forecast is a function of the timestamp alone, so the block was scored once.)

Once a configuration was chosen, the model was **refit from scratch on Train
plus Validation combined**, 2015-01-01 to 2019-12-31, and only then asked
for the 168 test hours. This step is required rather than optional: a
production model would obviously use the December data, and the December
data is the part of the record closest to the week being forecast.

Two details deserve a reader's attention because they change the numbers
materially.

**LightGBM is run recursively, not fed the answers.** Its feature list
includes a 24-hour lag and rolling means over the previous 24 and 168 hours.
For 5 January, the 24-hour lag is 4 January, which is inside the held-out
week. Reading those values from the actuals would be leakage and would
flatter the model heavily. Instead the model predicts hour one, appends its
own prediction to the history, builds hour two's features from that extended
history, and so on for all 168 steps. Errors compound, which is honest, and
it is what a pipeline running once a week actually faces.

**SARIMA's training window length is itself a hyperparameter.** Fitting a
state-space model by maximum likelihood on all 41,616 training hours takes
minutes per candidate, which makes a real search impossible. We therefore
searched over the length of the recent window used for fitting (26 or 52
weeks) alongside the model orders and a switch for weekly Fourier
regressors, and let the validation MAPE choose. This is defensible for
ARIMA-type models, which are local by construction: after seasonal
differencing, 2015 says very little about January 2020. The chosen
configuration used 52 weeks.

## 4. Results

Sorted by test MAPE, best first. "Validation MAPE" is the selection score
described above, shown so the reader can see where a model's test result was
and was not anticipated.

| Model | Test MAPE (%) | RMSE (MW) | MAE (MW) | MAPE 1 Jan, holiday (%) | MAPE 2-7 Jan (%) | Validation MAPE (%) | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 5.46 | 3787 | 3016 | 2.19 | 6.00 | 3.00 | 102 |
| Prophet | 7.61 | 4878 | 3798 | 13.28 | 6.67 | 5.64 | 59 |
| TSMixer | 10.78 | 7293 | 5685 | 4.66 | 11.80 | 6.16 | 2511 |
| N-BEATS | 11.06 | 7727 | 5824 | 2.75 | 12.44 | 5.59 | 2165 |
| SARIMA | 11.44 | 6871 | 6027 | 16.31 | 10.63 | 6.41 | 634 |
| Seasonal naive | 12.78 | 8809 | 7238 | 6.52 | 13.82 | 5.48 | 0 |

Probabilistic scores for the winner, LightGBM:

| Quantity | Nominal | Actual |
|---|---:|---:|
| 80% prediction interval coverage | 80.0% | 53.6% |
| 95% prediction interval coverage | 95.0% | 91.7% |
| Pinball loss at q=0.1 | | 735 MW |
| Pinball loss at q=0.5 | | 1476 MW |
| Pinball loss at q=0.9 | | 497 MW |

## 5. Discussion

LightGBM won at 5.46% MAPE, less than half the naive baseline's 12.78%, and
was also the cheapest serious contender to fit: 102 seconds against 36
minutes for N-BEATS.

The rank ordering is not what a complexity ladder predicts. The two deep
models finished third and fourth, barely ahead of the naive baseline,
despite twenty times the compute. The holiday split shows why. LightGBM is
told that 1 January is a public holiday and scores 2.19% on it. Nothing
tells N-BEATS or TSMixer that; they see 168 raw numbers, and their context
week, 25 to 31 December, is itself a holiday period. They extrapolated a
holiday-shaped week into a working week: N-BEATS scores an excellent 2.75%
on 1 January and a poor 12.44% on 2 to 7 January. The naive baseline fails
the same way for the same reason, since its reference week averages
46,491 MW against 53,400 MW in the test week.

Prophet is the instructive failure. It is the only model given an explicit
holiday term, and the only one that does worse on the holiday (13.28%) than
on ordinary days (6.67%). Its holiday effect is one additive shift per
holiday name, fitted from five previous New Year's Days: it can move the
daily profile down but not flatten its shape, and on 1 January the shape is
what changes. SARIMA fails differently: with a seasonal period of 24 hours
and a 168-hour horizon it has almost no memory left by the end of the week
and decays to the structure it was given.

The winner's weaknesses are instructive too. Its mean residual is -2,809 MW,
an over-forecast of 5.26% of mean load, so almost all its error is a level
offset rather than noise. Figure 7 shows the cause: the 364-day lag carries
73.8% of the model's split gain, and the week 364 days earlier averaged
58,883 MW, 10.3% above the test week. The model anchored on a
busier year. Its 80% interval covered only 53.6% of hours, because the
quantile models were trained on one-step-ahead residuals and then applied to
recursively built features, so the bands never widen with lead time.

Two caveats. This is one origin and one week, chosen because it contains a
holiday, so gaps of a few tenths of a point are not meaningful. And no model
sees temperature, the largest single driver of German load, because the
study is univariate by design.

## 6. Recommendation

For a JRC production short-term load forecasting pipeline, we would put
LightGBM on engineered features into service. It won on accuracy, it is by
far the cheapest to retrain, its errors are inspectable feature by feature,
and it is the only entry that accepts domain knowledge such as a holiday
calendar directly. Before deployment we would do four things, in this order.
First, fix the prediction intervals, which are the most serious defect
found: train one model per horizon step, or add horizon as a feature to the
quantile models, so the bands widen with lead time as they should. Second,
address the level bias by making the year-ago lag relative rather than
absolute, for instance as a ratio to a recent rolling mean, so the model is
not anchored to the absolute level of a busier year. Third, replace this
single-window evaluation with a rolling-origin backtest over one to two
years of weekly origins, reporting the distribution of weekly MAPE rather
than one number, and paying separate attention to public holidays and to
the days around them. Fourth, once the univariate constraint of this study
is lifted, add temperature and a solar and wind generation forecast, which
in our judgement would matter more than any further change of model class.

## 7. Reproducibility

- **Random seed:** 42, applied to `random`, `numpy` and `torch` and passed
  to LightGBM (`random_state`) and to the darts models (`random_state`).
  SARIMA and Prophet are deterministic given their inputs; Prophet's
  interval sampling and the deep models' interval sampling use the seeded
  generators.
- **Total wall-clock runtime:** 5,473 seconds (about 91 minutes), summed
  over the six model runs plus scoring and figure generation. The individual
  model runtimes are in the results table and in `metrics.json`. Deep-model
  runtimes include their two-candidate hyperparameter search and the final
  retrain; two of those stages were run concurrently, so their recorded
  seconds overlap in wall-clock terms and overstate the elapsed time.
- **Hardware and OS:**
  `Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64`
- **Software:** Python 3.12.13; pandas 2.3.3, numpy 2.4.4, matplotlib 3.10.9,
  statsmodels 0.14.6, pmdarima 2.1.1, darts 0.41.0, lightgbm 4.6.0,
  holidays 0.96, torch 2.10.0, prophet 1.3.0.
- **Compute note:** the deep models were trained on the CPU. An Apple-silicon
  GPU run was measured at 27.9 seconds per epoch against 23.3 on the CPU, so
  there was nothing to gain. Their epoch budget was set to 22 with early
  stopping (patience 4) on the validation loss, which fits the study's "at
  most about 30 epochs" and the wall-clock limits of the run environment. The
  winning N-BEATS candidate stopped at 19 epochs; both TSMixer candidates ran
  the full 22.
- **How to reproduce:**

  ```
  ../../.venv/bin/python code/forecast.py --refit all
  ```

  This refits all six models, rewrites `metrics.json`, `metrics.csv` and all
  figures. Without `--refit` the orchestrator reuses the cached per-model
  results in `artifacts/` and only redraws the outputs, which takes seconds.
  The two deep models can also be run one stage at a time
  (`python code/nbeats.py sweep 0`, `sweep 1`, `final`) if a single long job
  is inconvenient.
- **Intermediate files:** `artifacts/` holds one pickled result per model
  plus the deep models' per-candidate search records. It is a cache, not a
  deliverable, and can be deleted at the cost of a full refit.
