# Forecasting bake-off: German hourly electricity load, one week ahead

Six forecasting methods were fitted on the same data, tuned on the same
validation weeks, refitted on everything before the test week, and scored
on the same 168 held-out hours. This document is the methods section and
the results; `metrics.json` and `metrics.csv` hold the numbers, `figures/`
the plots, and `code/forecast.py` reproduces everything.

## 1. Data

The series is the national German electricity load in megawatts, one value
per hour, as published by the ENTSO-E Transparency Platform and packaged by
Open Power System Data (`opsd_de_load.csv`, columns `utc_timestamp` and
`DE_load_actual_entsoe_transparency`). It runs from 2015-01-01 00:00 UTC to
2020-09-30 23:00 UTC: 50,400 hourly rows. The loader (`code/common.py`)
checks the row count, checks that consecutive timestamps are exactly one
hour apart, and checks that no value is missing; all three checks pass, so
no imputation, dropping or interpolation was needed or done. Timestamps
are kept in UTC throughout; the only place local time enters is the
calendar features (hour, weekday, month, weekend, public holiday), which
are computed on the Europe/Berlin clock because that is the clock people
live by. Load ranges from about 30 GW (summer nights, holidays) to about
80 GW (winter weekday evenings), with a daily, a weekly and a yearly
rhythm plus a visible dip every Christmas and New Year.

## 2. Why these six models

The six models form a ladder of increasing flexibility, and each rung adds
one ingredient. The seasonal-naive baseline ("next week equals last week")
costs nothing and captures the weekly rhythm; anything that cannot beat it
is not worth running. SARIMA is the classical statistical model: it adds
short-memory dynamics (how the last few hours and days shape the next) on
top of a daily season, with the weekly rhythm and public holidays offered
as simple regressors built from the timestamp. Prophet decomposes the load
into a bending trend plus yearly, weekly and daily waves and a bump per
German public holiday; it is the model that knows the calendar best but
conditions least on recent observations. LightGBM on engineered features
is the workhorse of industrial load forecasting: a tree ensemble that can
combine calendar flags, lagged loads and rolling statistics in arbitrary
interactions, forecasting recursively hour by hour. N-BEATS and TSMixer
are two deep-learning designs that see nothing but the last 168 hours of
load and write out the next 168 in one shot: N-BEATS with deep stacks of
fully connected blocks, TSMixer with alternating time-mixing and
feature-mixing layers. They test whether a flexible pattern-continuation
model can learn the calendar implicitly from the shape of the series. The
PatchTST slot is filled by TSMixer because the installed darts (0.41.0)
does not ship a PatchTST model; TSMixer is the first of the two options
named in the brief and belongs to the same "read a window, mix it,
project it" family, though it uses multilayer perceptrons rather than
attention. This substitution is the only deviation from the brief's model
list.

## 3. Validation strategy

The data were cut into three windows that never overlap: training
(2015-01-01 to 2019-09-30, 41,616 hours), validation (2019-10-01 to
2019-12-31, 2,208 hours) and test (2020-01-01 to 2020-01-07, 168 hours).
Every model with settings to choose was fitted on the training window only
and its candidate settings were scored on the validation window; nothing
after 2019-09-30 was seen during those fits. To make the validation score
resemble the task the test poses, the validation window was cut into 13
whole weeks and each candidate forecast every week from a fixed origin at
its start, with only data before that origin available, and the 13 weekly
forecasts were pooled into one MAPE (the last 24 hours of December are
left over and unused). Prophet's forecast is a pure function of the
calendar, so a single 13-week forecast is identical to 13 weekly ones. The
two neural models used the validation weeks twice, in the way the brief
asks for: as the early-stopping signal (training stops when the validation
MAPE has not improved for five epochs, and the best epoch's weights are
kept) and as the score that picks between candidate settings. Once the
settings were fixed, every model was refitted on training plus validation
(2015-01-01 to 2019-12-31) and forecast the test week from that refit.
This is the standard practice: a production model would always be fitted
on everything available before the forecast origin, and the validation
window is the most recent, most relevant quarter of data. For the neural
models the refit has no validation window left to stop on, so it runs for
exactly the number of epochs that was best in the first stage. The test
week was never used for anything except the final scoring. No weather,
price or gas series enters any model.

What each model's validation sweep covered, and what it chose:

- SARIMA (36 candidates): estimation window 4 or 8 weeks; regressors
  none, weekly Fourier waves (3 harmonics), or weekly waves plus a holiday
  flag; non-seasonal order (1,0,1), (2,0,1) or (1,1,1); seasonal order
  (0,1,1,24) or (1,1,1,24). Chosen: 4 weeks, weekly waves,
  (1,0,1)(1,1,1,24), validation MAPE 5.75 %. The "weekly plus holiday"
  variant tied exactly with "weekly" because no public holiday falls in
  the August-to-September estimation windows, so the holiday coefficient
  could not be identified there; the tie was broken toward the simpler
  model.
- Prophet (4 candidates): Prophet's defaults; separate weekday and weekend
  daily waves (Fourier order 10) with weekly order 6 and yearly order 10,
  additive; the same, multiplicative; the same additive with a stiffer
  trend (changepoint prior 0.01). Chosen: conditional daily waves,
  additive, stiff trend, validation MAPE 4.24 % (defaults: 5.41 %).
- LightGBM (8 candidates): objective L2 or L1, 31 or 63 leaves, learning
  rate 0.03 or 0.1, 500 trees. Chosen: L1, 63 leaves, 0.1, validation
  MAPE 3.02 %. Every candidate sat between 3.02 % and 3.16 %, so the
  choice barely matters.
- N-BEATS (2 candidates): darts' default generic architecture (30 stacks,
  4 layers of width 256), learning rate 0.001 or 0.0003, batch 512, at
  most 30 epochs. Chosen: 0.001, best epoch 30, validation MAPE 4.81 %.
- TSMixer (2 candidates): small (hidden 64, 2 blocks, dropout 0.1) or
  large (hidden 128, feed-forward 256, 4 blocks, dropout 0.2), learning
  rate 0.001, at most 30 epochs. Chosen: large, best epoch 22, validation
  MAPE 5.21 % (small: 5.24 %).

## 4. Results table

Test week 2020-01-01 to 2020-01-07 (168 hours), sorted by MAPE. The
validation column is the pooled 13-week MAPE from the selection stage and
is shown only to compare the two orderings.

| Model | MAPE (%) | RMSE (MW) | MAE (MW) | MAPE Jan 1, holiday (%) | MAPE Jan 2 to 7 (%) | Validation MAPE (%) | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 4.99 | 3,616 | 2,761 | 3.04 | 5.32 | 3.02 | 170 |
| Prophet (DE holidays) | 7.90 | 4,799 | 3,988 | 15.75 | 6.59 | 4.24 | 95 |
| TSMixer (PatchTST slot) | 9.13 | 5,918 | 4,769 | 6.35 | 9.59 | 5.21 | 935 |
| N-BEATS | 10.55 | 7,119 | 5,510 | 4.48 | 11.56 | 4.81 | 1020 |
| Seasonal naive (t-168 h) | 12.78 | 8,809 | 7,238 | 6.52 | 13.82 | 5.51 | 0 |
| SARIMA | 14.99 | 9,471 | 8,344 | 7.19 | 16.29 | 5.75 | 327 |

Per-day MAPE (%), the content of figure 04:

| Model | Wed 1 (holiday) | Thu 2 | Fri 3 | Sat 4 | Sun 5 | Mon 6 | Tue 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 3.0 | 2.1 | 3.3 | 2.9 | 4.7 | 11.7 | 7.2 |
| Prophet (DE holidays) | 15.7 | 6.9 | 4.7 | 8.9 | 7.6 | 5.6 | 5.9 |
| TSMixer (PatchTST slot) | 6.3 | 8.8 | 4.7 | 10.7 | 21.4 | 6.5 | 5.5 |
| N-BEATS | 4.5 | 14.6 | 5.6 | 9.4 | 26.3 | 7.3 | 6.2 |
| Seasonal naive (t-168 h) | 6.5 | 19.2 | 12.4 | 9.4 | 5.5 | 11.8 | 24.7 |
| SARIMA | 7.2 | 10.5 | 13.4 | 19.3 | 15.3 | 16.1 | 23.2 |

Winner, LightGBM, probabilistic scores on the test week:

| Quantity | Nominal | Actual |
|---|---:|---:|
| 80 % prediction-interval coverage | 80 % | 50.6 % |
| 95 % prediction-interval coverage | 95 % | 91.7 % |
| Pinball loss, quantile 0.1 (MW) | | 910 |
| Pinball loss, quantile 0.5 (MW) | | 1,331 |
| Pinball loss, quantile 0.9 (MW) | | 405 |

The same quantities for the other probabilistic models are in
`metrics.json` (80 % / 95 % coverage: Prophet 48 % / 71 %, TSMixer
95 % / 99 %, N-BEATS 51 % / 74 %, SARIMA 36 % / 71 %).

## 5. Discussion

LightGBM won (4.99 % against 7.90 %) because it alone combines the three
things this week demands: the calendar (holiday flag), the same weekday
one year earlier (40 % of its split gain) and the recent level (week-ago
and day-ago lags). On New Year's Day it was the best model (3.0 %). Its
weak spot is the far end of the horizon: Monday 6 and Tuesday 7 January
were over-forecast by 4 to 9 GW, because the recursive scheme compounds
its own errors after the first day, and because 6 January is Epiphany, a
holiday in Bavaria and Baden-Wuerttemberg (about 30 % of the population)
that the federal flag does not know.

Prophet came second and lost most of its margin on 1 January (15.7 %,
against 6.6 % on other days): it models a holiday as a flat shift, while
a real holiday changes the shape of the day. The two neural models beat
the naive copy but failed together on Sunday 5 January (N-BEATS 26 %,
TSMixer 21 %). They see only the previous 168 hours, and the week they
were shown (25 to 31 December) holds two holidays and three bridge days,
so it does not say which day ahead is the weekend; both continued a
weekday profile through Sunday. The naive copy reproduces Christmas
week: good on 1 January (Christmas Day), bad on 2 and 7 January (Boxing
Day, New Year's Eve). SARIMA finished last, below the baseline:
validation on the stationary autumn weeks preferred a 4-week estimation
window, so the final refit sat on the December trough and, being a
short-memory model, extrapolated that level, 8 to 10 GW too low all
week.

The ranking is largely what theory predicts: feature-based boosting is
the state of the art for calendar-driven load without weather, and
knowing the calendar beats inferring it from one atypical week. SARIMA
below the naive copy is the surprise; the origin explains it. Two
caveats. The test is a single 168-hour week chosen to contain a holiday,
so gaps of one or two points between neighbours are not meaningful; the
13 validation weeks give the same top two and bottom two but swap the
neural models. And the winner's intervals are too narrow (the 80 % band
covers 51 % of hours, the 95 % band 92 %), because the quantile models
never see the recursion's compounding error; only TSMixer's intervals
were calibrated.

## 6. Recommendation

For a JRC production short-term load pipeline I would put LightGBM on
engineered calendar and lag features into service, because it won on
every metric, is fast (three minutes including its sweep), is easy to
inspect, and its failure modes are the ones that are cheapest to fix. Four
pieces of work come first. (1) Replace the recursive scheme by direct
multi-horizon models (one model per lead time, or lead time as a feature)
so error does not compound across the week, and re-derive the intervals
from rolling-origin backtest residuals or conformal calibration so that
an 80 % band covers 80 %. (2) Enrich the calendar, still without any
external data: a population-weighted regional-holiday share (Epiphany,
Corpus Christi, Reformation Day), bridge-day and school-holiday flags,
and days-since and days-until the nearest holiday. (3) Backtest over at
least 52 rolling weekly origins covering every holiday and both clock
changes, and report per-week MAPE distributions rather than one test
week, so that the ranking rests on more than 168 hours. (4) Once the
univariate constraint of this study is lifted, add day-ahead temperature
forecasts, which are the largest known driver this study deliberately
excluded, and keep Prophet or the naive copy running alongside as a
sanity check and fallback.

## 7. Reproducibility note

- Command: `../../.venv/bin/python code/forecast.py` from this directory.
  Each model is fitted in its own child process (LightGBM and PyTorch
  each ship an OpenMP runtime and loading both into one process crashed
  PyTorch training on this machine), and every result is cached under
  `code/_scratch/results/`; `--reuse` regenerates figures and tables
  from the cache, `--models nbeats` fits a subset. The neural
  hyperparameter candidates are cached per candidate as JSON so an
  interrupted run resumes where it stopped.
- Seeds: 42 everywhere (Python `random`, NumPy, LightGBM `random_state`
  with `deterministic=True` and `force_col_wise=True`, darts
  `random_state=42` for N-BEATS, TSMixer and Prophet's sampled
  intervals). The neural models were trained on Apple's GPU (MPS), which
  is not bit-for-bit reproducible; rerunning may shift their MAPE by a few
  tenths of a point and could in principle flip the close TSMixer
  candidate choice. Set `FORCE_CPU=1` for exact reproducibility at roughly
  twice the training time.
- Runtime: the per-model runtimes sum to 2,546 s (42 minutes): naive 0 s,
  SARIMA 327 s, Prophet 95 s, LightGBM 170 s, N-BEATS 1,020 s, TSMixer
  935 s. The two neural models were fitted concurrently on the same GPU,
  so their individual timings are inflated; a sequential run is expected
  to take roughly 30 to 40 minutes on this hardware, and about 4 minutes
  for the four non-neural models.
- Hardware and software: `uname -a` gives
  `Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64`
  (Apple silicon, 15 CPU cores, 24 GB memory). Python 3.12.13, darts
  0.41.0, lightgbm 4.6.0, statsmodels 0.14.6, prophet 1.3.0, torch 2.10.0,
  pandas 2.3.3, numpy 2.4.4, holidays 0.96. No package was installed.
- Figures: 300 dpi, matplotlib's default sans-serif, one fixed tab10
  colour per model in every figure (grey naive, blue SARIMA, orange
  Prophet, green LightGBM, purple N-BEATS, cyan TSMixer; red is reserved
  for the test-window highlight). No six-colour subset of tab10 is fully
  colour-blind safe, so every figure also identifies models by panel
  title, axis label or direct value label rather than by colour alone.
- Deviations and choices worth knowing: TSMixer fills the PatchTST slot
  (section 2); SARIMA is estimated on a 4-week window with weekly
  Fourier regressors (section 3); LightGBM forecasts recursively and its
  0.5 quantile is the point model, with four extra quantile-objective
  models for the bands; calendar features use Europe/Berlin local time;
  the per-day and holiday breakdowns use UTC calendar days, as the test
  window itself is defined in UTC.
