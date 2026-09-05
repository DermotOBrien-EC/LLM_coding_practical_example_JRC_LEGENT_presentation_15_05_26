# German hourly load forecasting bake-off: six models, one held-out week

## 1. Data

The series is German national electricity load in megawatts at hourly
resolution, taken from the Open Power System Data time-series package,
which in turn derives it from the ENTSO-E Transparency Platform
(`opsd_de_load.csv`, column `DE_load_actual_entsoe_transparency`). It runs
from 2015-01-01 00:00 to 2020-09-30 23:00 UTC: 50,400 rows, one per hour.
The loader in `code/common.py` checks all of this before anything else
runs: exactly 50,400 rows, strictly increasing timestamps, an unbroken
hourly grid, and no missing values. All three checks pass, so no
imputation or gap filling was needed and none was done. Timestamps are
UTC; calendar features (hour, weekday, holiday) are computed in
Europe/Berlin time because load follows the local clock, while the
train, validation and test boundaries are the UTC times the study
specifies.

## 2. Why these six models

The six models form a gradient from "no model at all" to modern deep
learning, and each brings one distinct idea. The seasonal-naive baseline
repeats the previous week and sets the bar; anything that cannot beat it
has learned nothing useful. SARIMA is the classical statistical
benchmark: a small set of interpretable coefficients that describe how
the next hour depends on recent hours and on the same hour yesterday.
Prophet is a curve-fitting model (trend plus daily, weekly and yearly
curves) with explicit German holiday effects, so it is the one model
that is told in so many words that 1 January is special. LightGBM is a
gradient-boosted tree model fed engineered features (calendar flags,
lags of 24 h, one week and one year, and moving averages); it represents
the pragmatic industry approach and can also learn the holiday flag.
N-BEATS and TSMixer are neural networks that read only the last week of
load and emit the next week; they test whether a flexible model can
infer the calendar from the shape of the series alone, with no calendar
information at all.

## 3. Validation strategy

Three fixed, chronologically ordered windows are used. Train covers
2015-01-01 to 2019-09-30 23:00 (41,616 hours); Validation covers
2019-10-01 to 2019-12-31 23:00 (2,208 hours); Test is the single week
2020-01-01 to 2020-01-07 23:00 (168 hours). Every model with settings to
choose is fitted on Train only, and each candidate setting is scored on
Validation the same way the test week is scored: thirteen non-overlapping
168-hour forecasts, each made from the hour before the week starts
(pooled MAPE over 2,184 hours; the leftover 24 hours of the validation
window are not scored). The setting with the lowest validation MAPE is
then refitted on Train plus Validation, so the final model has seen all
data up to the hour before the test week. The refit matters because the
test week directly follows the validation window: a model that had to
skip the last quarter of 2019 would forecast January from a September
vantage point. The test week was not used for fitting, selection, or
choosing model classes; it is scored exactly once.

## 4. Results

Test week 2020-01-01 00:00 to 2020-01-07 23:00 UTC, 168 hourly
observations, one forecast per model made from the hour before the week
starts. Sorted by MAPE. Validation MAPE is the pooled score of the
thirteen one-week forecasts used for selection (Train fit only). Runtime
is wall-clock for the whole model step: candidate search, refit and
forecast.

| Rank | Model | MAPE (%) | RMSE (MW) | MAE (MW) | MAPE 1 Jan, holiday (%) | MAPE 2 to 7 Jan (%) | Validation MAPE (%) | Runtime (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | LightGBM (features) | 5.53 | 3,814 | 3,046 | 2.26 | 6.08 | 3.06 | 68 |
| 2 | TSMixer (PatchTST slot) | 7.22 | 5,007 | 3,887 | 5.87 | 7.45 | 5.12 | 623 |
| 3 | Prophet (DE holidays) | 7.84 | 4,761 | 3,955 | 15.59 | 6.54 | 4.47 | 35 |
| 4 | N-BEATS | 9.76 | 6,343 | 5,135 | 6.44 | 10.31 | 4.67 | 205 |
| 5 | Seasonal naive (t-168 h) | 12.78 | 8,809 | 7,238 | 6.52 | 13.82 | 5.51 | 0 |
| 6 | SARIMA (3,0,2)(1,1,1,24) + calendar shifts | 15.55 | 9,513 | 8,373 | 6.81 | 17.01 | 5.86 | 545 |

Per-day MAPE (%) on the test week (UTC calendar days):

| Model | Wed 1 Jan | Thu 2 Jan | Fri 3 Jan | Sat 4 Jan | Sun 5 Jan | Mon 6 Jan | Tue 7 Jan |
|---|---|---|---|---|---|---|---|
| LightGBM | 2.26 | 1.59 | 5.03 | 5.30 | 6.35 | 11.72 | 6.47 |
| TSMixer | 5.87 | 6.50 | 3.72 | 5.36 | 13.83 | 2.86 | 12.43 |
| Prophet | 15.59 | 7.49 | 3.87 | 8.85 | 8.39 | 5.04 | 5.62 |
| N-BEATS | 6.44 | 9.59 | 9.03 | 7.18 | 23.25 | 7.03 | 5.79 |
| Seasonal naive | 6.52 | 19.17 | 12.35 | 9.40 | 5.54 | 11.77 | 24.71 |
| SARIMA | 6.81 | 6.15 | 12.48 | 23.49 | 23.41 | 15.25 | 21.26 |

Winning model, LightGBM, probabilistic scores on the test week:

| Quantity | Nominal | Actual |
|---|---|---|
| 80 % interval coverage (q0.1 to q0.9) | 0.80 | 0.488 |
| 95 % interval coverage (q0.025 to q0.975) | 0.95 | 0.833 |
| Pinball loss, q0.1 (MW) | | 1,021 |
| Pinball loss, q0.5 (MW) | | 1,456 |
| Pinball loss, q0.9 (MW) | | 427 |

The winner's intervals are too narrow: 49 % of hours fell inside the
80 % band and 83 % inside the 95 % band. The quantile models were trained
on features built from observed values (a day-ahead setting) and applied
up to seven days out with predicted values in the 24-hour lag and rolling
features, so the recursion error is outside their training distribution.
The pinball losses show the same bias as the residuals: the upper
quantile is rarely exceeded because the point forecast already runs high.
As a supplementary figure outside the bake-off, the final LightGBM run
one day ahead, with observed values in its lag features, scores 3.71 %
on the same week; that is closer to how such a model is used in practice.

Figures: `figures/01_overview.png` (full series with the test week),
`02_forecast_comparison.png` (one panel per model), `03_metric_comparison.png`
(MAPE, RMSE and MAE as three stacked panels rather than one grouped chart,
because the three metrics have different units), `04_per_day_mape.png`,
`05_winner_with_intervals.png`, `06_residuals.png` and
`07_feature_importance.png` (LightGBM is in the top two; Prophet is not,
so `08_decomposition.png` is not produced). Machine-readable versions:
`metrics.json`, `metrics.csv`, and `forecasts.csv` with every model's
point and quantile forecasts.

## 5. Discussion

**Who won and why.** LightGBM won with a test MAPE of 5.53 %, ahead of
TSMixer, Prophet and N-BEATS and far ahead of naive and SARIMA. It is
the only model that sees both the calendar, holiday flag included, and
the recent past through lags; its top feature is the same weekday a year
earlier (73 % of split gain). On 1 January it scored 2.3 %: the flag
says holiday, and the one-week lag points at Christmas Day, another
Wednesday holiday.

**The week is hard for everyone.** Every model was two to three times
worse than on its validation weeks. The week opens on a holiday, 2 and
3 January are bridge days, and Monday 6 January (Epiphany) is a holiday
in three states holding close to 30 % of German demand, yet not federal
and so invisible to the flag. Load climbs out of the Christmas trough
all week; the series alone cannot say how fast.

**Where each model failed.** Naive repeats Christmas week and under-
predicts every working day by 8 to 10 GW (2 January copies Boxing Day,
19 %). SARIMA did worse: its origin sits in the trough, the model
carries that low level forward, and the weekday shifts subtract a full
weekend effect from it, giving 25 to 30 GW troughs on Saturday and
Sunday (23 % each). Prophet, holiday term and all, scored 15.6 % on 1
January: the term is a constant shift on a weekday shape, so it still
draws a morning ramp (up to 13 GW too high). The calendar-blind neural
models lost track of the weekday: N-BEATS gave Sunday working-day levels
(23 %); TSMixer did the same (14 %) and under-predicted the Tuesday
recovery (12 %). LightGBM's weakness is the mirror image: it runs high
all week and worst on 6 January (5 to 10 GW), because that Monday a year
earlier was a normal working day.

**Ordering versus theory.** Broadly as expected: calendar plus recent
past beats either alone, and both beat repeating last week. SARIMA below
naive is an artefact of a 52-week window ending in the Christmas trough,
and the middle three are not separable on one week (N-BEATS won
validation at 4.67 % but lost the test). Only "LightGBM best, naive and
SARIMA worst" is robust.

**Caveats.** One atypical 168-hour week; SARIMA on a 52-week window; a
small neural search (two configurations, 30 epochs); TSMixer standing in
for PatchTST.

## 6. Recommendation

For a JRC short-term load forecasting pipeline I would pick LightGBM
with engineered features. It won here, it is cheap to fit (68 s
including the search and five quantile models), it can be retrained
daily, and its failures are legible through feature importances and
residual plots. Before putting it into production I would do four
things. First, replace the federal holiday flag with a load-weighted
regional holiday calendar and add bridge-day and "between the years"
flags, which is where the remaining large errors sit. Second, replace
the recursive strategy with direct multi-horizon models (a lead-time
feature, or one model per lead time), and train the quantile models per
horizon so that the intervals reflect one-week-ahead uncertainty and
become calibrated. Third, run a rolling-origin backtest over at least
one full year of weekly origins, because one January week cannot rank
the middle of the field. Fourth, add the weather forecast inputs this
study deliberately excluded; temperature explains most of what remains
once the calendar is right.

## 7. Reproducibility note

Run from this directory with `../../.venv/bin/python code/forecast.py`;
it fits all six models, writes `metrics.json`, `metrics.csv`,
`forecasts.csv`, the figures and `run.log`, and pickles each model's
forecast under `cache/` so that `--figures-only` redraws without
refitting and `--only <name>` reruns one model. Each model also runs on
its own (`python code/<model>.py`). The shared helper for the two
neural models lives in `code/torch_common.py`; `code/common.py` moves
the script directory to the end of `sys.path` because the mandated
file name `prophet.py` would otherwise shadow the installed `prophet`
package.

Seeds: 42 everywhere (LightGBM `random_state`, darts `random_state` for
N-BEATS, TSMixer and Prophet sampling, which also seeds torch). SARIMA
and the Prophet point fit are deterministic optimisations. The neural
models train on the Apple GPU (`accelerator="mps"`), which is why the
run fits in the time budget; the CPU path gives the same ranking but
slightly different numbers (in a CPU-only run the neural scores were
9.76 % against 10.94 % for N-BEATS and 7.22 % against 7.53 % for
TSMixer), and bit-for-bit repeatability across hardware is not
guaranteed for either path.

Wall-clock for the complete run: 1481 s (24.7 min), of which model
fitting 1476 s; per-model times are in the results table. The run was
timed while an unrelated training job shared the machine, so an idle
machine should be somewhat faster. Hardware: Apple M5 Pro (15 cores,
24 GiB), macOS 25.5.0. `uname -a`:

```
Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64 arm
```

Software: Python 3.12.13, darts 0.41.0, pmdarima 2.1.1 (installed, not
used: the statsmodels SARIMAX route was taken), statsmodels 0.14.6,
prophet 1.3.0, lightgbm 4.6.0, holidays 0.96, torch 2.10.0, pandas
2.3.3, numpy 2.4.4, matplotlib 3.10.9. No packages were installed.

Deviations and notes. (1) darts 0.41.0 has no `PatchTSTModel`; the
`patchtst` slot runs `TSMixerModel`, the first alternative the study
names, and keeps the slot name in all outputs. (2) SARIMA is estimated
on the most recent 52 weeks of the fitting history rather than the full
4.75 years (about 3 minutes per candidate fit on the full history versus
45 s), with weekday and holiday level shifts as calendar regressors. (3)
LightGBM forecasts recursively so that it is a genuine 168-hour-ahead
forecast; its rolling-window features end one hour before each row so
the target never enters its own features. (4) No external data were
used or wanted; the one moment the temptation arose, the 6 January
errors, is discussed above and is a calendar issue, not a weather one.
(5) The data passed every integrity check; no rows were dropped or
imputed.
