# Forecast plan and independent challenge

## Original plan

Forecast 168 hourly German loads from 2020-01-01 00:00 through
2020-01-07 23:00 UTC. Exclude all observations on or after the forecast
cutoff before modelling. Write consumer tests first. Compare a calendar-only
LightGBM, a horizon-safe weekly/annual-lag LightGBM, their fixed equal blend,
and seasonal baselines. Select on expanding-window first-week-of-January
backtests in 2017, 2018, and 2019; retain late-2019 weeks as separate
robustness checks. Refit on all eligible pre-2020 history and deliver CSV,
backtest results, executable code, and methods report. Do not claim calibrated
uncertainty from only three New Year transitions.

The operator explicitly requested unattended end-to-end completion, so there
was no separate human approval gate. One independent Claude peer challenged
the plan before implementation and was resumed for deliverable verification.

## Independent peer's plan challenge (full substantive response)

Ground truth checked: the file is two columns (`utc_timestamp`,
`DE_load_actual_entsoe_transparency`), hourly UTC from 2015-01-01 00:00,
50,400 data rows, which matches 2015 through September 2020 with few or no
gaps. I read no 2020 values.

### Leakage: the plan is mostly safe by construction, with three places to pin down in tests

1. Lags are horizon-safe as stated: for a 168-hour block from origin o,
   lag 168 at the last target hour resolves to o itself, never past it.
   That holds only if the lag features are built from the raw series and
   the 2020 rows are dropped before any shift, rolling, or fill. A test
   should assert the training frame's max timestamp is
   2019-12-31 23:00 and that no feature column for the target week is
   derived from any row with timestamp > origin.
2. LightGBM tree count. If early stopping or any validation split touches
   the fold's scored week, that is leakage into selection. Fix the number
   of trees a priori or early-stop on a slice strictly before the fold
   origin. State which in the report.
3. Calendar distances must wrap across the year boundary. "Days to Jan 1"
   on Dec 28 must be -4, not +362. This is the most common bug in this
   feature and it silently degrades exactly the week you care about.
   One unit test on Dec 28 and Jan 3 pins it.

### New Year difficulty: the target week is transitional and the baselines are weak

- Jan 1 2020 is a Wednesday holiday, Jan 2-3 are bridge days with heavy
  vacation, Jan 6 is regional Epiphany (roughly a quarter of national load
  in BW, BY, ST), Jan 7 is the first normal day. Load ramps from holiday
  level to full working level within the week.
- Weekly naive uses Dec 25-31 (Christmas week) and will be biased low;
  last-4-weeks median uses December workdays and will be biased high on
  Jan 1-3. Both are easy to beat, so "beats baselines" proves little.
- The lag 8736 analog (52 weeks, weekday aligned) is the strong signal,
  but it maps holiday Wed Jan 1 2020 onto ordinary Wed Jan 2 2019.
  The model must learn that interaction from very few examples: with
  folds starting 2017, the 2017 fold's training set has analog lags only
  for 2016 rows (one January). Expect the lag model's fold scores to be
  unstable across 2017/2018/2019, and report them per fold, not just the
  mean.
- Year trend inside a tree model cannot extrapolate: 2020 is unseen and
  will be binned with 2019. That is acceptable, but say so in the report
  rather than implying a fitted trend.

### Validation fairness

- Three folds (Jan 1-7 of 2017, 2018, 2019) start on Sun, Mon, Tue; the
  target starts Wed. Weekday coverage is decent, holiday-on-weekday
  structure differs each year, so the mean MAE across three folds is noisy.
  Selecting among five candidates (two models, blend, two baselines) on
  three folds of 168 hours is a small-sample choice. Keep it, since the
  plan forbids tuning, but present per-fold MAE and skill relative to the
  best baseline so the reader can see whether the winner wins in every
  fold or only on average.
- Autumn 2019 weeks are correctly framed as sanity checks, not selection.
  Make sure the script cannot accidentally include them in the selection
  mean.
- Data quality at the origin: naive and lag 168 both depend on Dec 25-31
  2019 being complete. The audit should count missing hours in the last
  4 weeks of 2019 and in each fold's lag windows, and state the fill rule.
  ENTSO-E transparency load has known gaps in early 2015; those affect
  the 2016 analogs only.

### Minimal changes worth making

1. Add one stronger baseline: weekday-aligned same-week-last-year
   (lag 8736) scaled by the ratio of the trailing 4-week mean before origin
   to the corresponding 4-week mean one year earlier. It is three lines,
   holiday-aware by construction, and is the honest bar for this week.
   If LightGBM does not beat it on the folds, the blend should include
   or defer to it.
2. Fix tree count or a pre-origin early-stopping slice explicitly
   (item 2 above).
3. Unit tests: year-boundary wrap on the distance features; max training
   timestamp assertion; forecast.csv has exactly 168 hourly UTC rows from
   2020-01-01 00:00, no NaN, values in a plausible band (say 25 to 85 GW,
   checked against pre-2020 min/max, not 2020 values).
4. Error bands: three folds give only three residuals per horizon hour.
   Pool residuals by local hour-of-day across folds instead, and keep
   the "uncalibrated, empirical" label.

Nothing here requires broadening the model search. The plan as written is
sound; the risks are the wrap bug, a hidden early-stopping leak, and
overstating a win against weak baselines.

## Author's independent evaluation and implementation decisions

- **Accepted:** strict pre-origin slicing, fixed tree count, signed
  year-boundary distances, explicit UTC output contract, per-fold reporting,
  and separate diagnostic folds. These are implemented and tested.
- **Accepted with a correction:** added the scaled lag-8736 baseline.
  Rejected "holiday-aware by construction": 52-week alignment preserves
  weekday, not fixed-date holiday status. The implementation and report
  call this weekday-aligned, not holiday-aligned. The blend is selected
  against all three baselines, rather than assumed superior.
- **Qualified:** the peer uses "origin" for the last observed hour.
  The implementation uses `start` for the first forecast hour. Thus the
  last target's lag-168 source is `start - 1 hour`. The test asserts this
  exact instant and all lag sources strictly before `start`.
- **Qualified:** the full source audit, rather than inference from row
  count, establishes zero gaps and duplicates in the pre-2020 input.
  The peer's general early-2015 gap statement is not a finding about this
  supplied file. No load filling was needed.
- **Qualified:** the share of national load affected by Epiphany and a
  universal "first normal day" were not independently established. Neither
  is used as a numerical assumption. The model uses a date indicator and
  learns effects from historical loads; the report discloses no regional
  load weighting or school-calendar data.
- **Qualified:** the year feature cannot linearly extrapolate. However,
  the last training timestamp is already 2020-01-01 00:00 in local CET,
  so the peer's literal claim that local year 2020 is entirely unseen is
  not true. The report makes only the supported non-extrapolation claim.
- **Accepted:** avoid presenting three holiday transitions as hundreds of
  independent samples. No prediction intervals are supplied; pooling by
  hour would not resolve between-year dependence or selection optimism.
- **Implementation refinement:** the lag model retains training rows with
  absent annual lag features using LightGBM's native missing-feature
  handling, so valuable early January examples are not discarded merely
  because a 366-day reference predates the source. It omits only the first
  672 hours, ensuring its four weekly lag values are present.

The completed implementation and resulting forecasts are the original work
reviewed in the subsequent deliverable-verification round. See
`forecast_load.py`, `forecast_report.md`, and `verification.md`.
