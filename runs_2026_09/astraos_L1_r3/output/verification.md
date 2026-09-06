# Verification and independent review

Date: 2026-09-06. Forecast information cutoff: 2020-01-01 00:00 UTC.

## Executed checks

- Consumer tests were written before the implementation. The initial run
  failed because the forecast module did not yet exist. A timezone mismatch
  in the synthetic test fixture was corrected before the forecast run.
- Final pytest result: **11 passed**. Tests include UTC horizon/shape,
  rejection of invalid forecasts and incomplete history, future-value
  exclusion, local holiday/date boundaries, horizon-safe lag values,
  invariance to appended future observations, and selection excluding
  diagnostic folds.
- Final Ruff format and lint checks pass.
- The final script completed all eight fixed-origin backtests and the final
  forecast. It was rerun after the report revisions; `forecast.csv` was
  byte-identical to the previously reviewed forecast.
- The author separately recomputed all 192 saved metric values across
  48 model/fold groups with Python's standard library, checked all stored
  validation labels against pre-2020 source observations, checked the
  selected model, daily summaries, 168-hour UTC sequence, finite positive
  forecasts, a 25 to 85 GW plausibility range, and input fingerprint.
  All assertions passed.
- Strict mypy **does not pass in the supplied environment**. Its initial
  cached invocation raised an internal error. A no-cache run completed
  and reports missing `pandas` type stubs in both Python files. A separate
  `holidays.__version__` export warning was corrected by using
  `importlib.metadata.version`. No type-stub packages were installed.
  Runtime tests and forecasting are unaffected; full static typing is
  not claimed.
- Neither modelling, model selection, nor these verification checks used
  actual target-week 2020 loads.

Forecast SHA-256:
`8f7b23e8af8f78d39bdc9b7b49c40b9be555b8c627e4aad9df4e8e104c5c4232`.

The input and final forecast remain local files. No commit, push,
publication, or external load-data upload was performed.

## Independent Claude-peer verification: original report

The following is the substantive review before the small report/test fixes.
Its line references name that reviewed revision, not the later report
insertion. Claims disputed by the author are evaluated immediately after
this section; do not treat the findings as unqualified project facts.

### Independently verified (all pass)

- Input SHA-256 matches `run_metadata.json` and the report. 43,824
  pre-2020 rows, complete hourly 2015-01-01 to 2019-12-31 23:00 UTC,
  range 31,307 to 77,549 MW, as the report states.
- `forecast.csv`: exactly 168 rows, columns
  `utc_timestamp, forecast_load_mw`, index equal to 2020-01-01 00:00
  to 2020-01-07 23:00 UTC hourly, all finite, within the historical range.
  Mean 55,792.69 MW, 9.373173 TWh, min 37,292.76 at Jan 1 05:00 UTC,
  max 72,659.51 at Jan 7 16:00 UTC, all matching the report and the
  coordinator's numbers. `daily_summary.csv` matches a resample of
  `forecast.csv` to the cent.
- Metrics: recomputed all 48 rows of `backtest_metrics.csv` from
  `backtest_predictions.csv` (MAE, RMSE, MAPE, bias); max deviation 5e-7.
  Actuals in `backtest_predictions.csv` equal the pre-2020 history at
  every row. `equal_blend` equals the mean of the two LightGBM columns
  at every hour. `weekly_naive`, `four_week_median`, and
  `scaled_annual_naive` recomputed from history and match at every fold.
  January means: blend 2471.83, lag 2559.26, calendar 2594.05, scaled
  annual 4364.73; skill 43.4% versus best baseline; RMSE column is the
  mean of fold RMSEs as the report says.
- Leakage: `load_history` drops timestamps at or after the cutoff before
  `to_numeric` (forecast_load.py:50-52); `backtest` passes
  `history.loc[history.index < start]` and `forecast_candidates` filters
  again (lines 154, 204); every lag is at least 168 h so the last target
  hour's lag 168 resolves to the origin minus one hour; robustness folds
  all end at or before 2019-12-31 23:00; no early stopping, 650 trees
  fixed. The contamination test (test line 102) genuinely catches a
  training-set leak (I traced that `available` = full history would fit
  on the 1e12 rows).
- Calendar: target week local hours run 01:00 Jan 1 to 00:00 Jan 8 CET;
  Jan 1 flagged national holiday, `holiday_kind` 2; Jan 6 `epiphany` 1,
  Monday; `new_year_distance` 0 to 6, `christmas_distance` 7 to 13.
  `holidays` 0.96 `DE` without subdivision gives the 9 nationwide days
  per year, plus the one-off 2017 Reformation Day (10 in 2017), which
  is correct. Annual analogs of the target week: lag 8736 maps to local
  Jan 2 to 9 2019 (weekday-aligned, Jan 1 holiday maps to ordinary Wed
  Jan 2 2019), lag 8760 to Jan 1 to 8 2019, lag 8784 to Dec 31 2018
  to Jan 7 2019.
- 11 tests pass, ruff passes, report has no em dash, `2f20c7...529c`
  checksum agrees.

### Findings

**No material blockers.** Four minor items, none changing the forecast.

1. **Minor, report completeness (forecast_report.md:102-113).** The blend
   wins the selection mean but never wins a January fold outright: 2017
   lag 2873 vs calendar 4694 (blend 3592), 2018 calendar 1646 vs lag 3370
   (blend 2432), 2019 all three within 50 MW. The blend's advantage comes
   from anti-correlated fold errors of its components, which is exactly
   why a fixed 50:50 blend is reasonable, but the per-fold table shows
   only the winner. One sentence stating this, or a per-fold table for
   the three learned candidates, would make the selection honest at a
   glance rather than only discoverable in `backtest_metrics.csv`.
2. **Minor, limitation not disclosed (forecast_load.py:100,
   report:56-62).** Epiphany fell on a weekday in training only in 2015
   (Tue), 2016 (Wed), 2017 (Fri); with the 730-day half-life those rows
   carry weights of about 0.18 to 0.36. The Jan 6 2020 forecast
   (59.7 GW Monday, below Jan 3's 60.3 and Jan 7's 64.1) rests on thin,
   down-weighted evidence. Worth one sentence.
3. **Minor, test adequacy (test_forecast_load.py).** Untested behaviours
   a mutation would survive (hardening, not defects): sample-weight
   direction (flipping `age_days` sign at forecast_load.py:178 upweights
   old data with no test going red); the `target.max() >= CUTOFF` fold
   guard at line 198; `christmas_distance` and `holiday_kind` values;
   the `lag <= 672` weekly-column membership at line 127. Also no test
   asserts the written `forecast.csv` values lie within the pre-2020
   min/max band. None of these affected this run because FOLDS and the
   weight formula are correct as written.
4. **Trivial (forecast_load.py:409).** The em-dash guard contains the
   literal character, so any owner-side em-dash grep over the source
   trips on it. An escaped code point avoids that.

### Could not verify

- The strict mypy claim (ran nothing under mypy; the coordinator's own
  note already discloses the pandas-stubs gap).
- Runtime of the full run and that the frozen artifacts were produced
  by the frozen `forecast_load.py` (I did not rerun the training;
  timestamps are consistent, 03:56 source, 03:58 outputs).
- Whether the supplied series is an as-of-2019 vintage; the report
  already disclaims this.

## Author's evaluation and fixes

1. **Partially accepted.** Added a per-January-fold MAE table for all
   three learned candidates, generated directly from the metrics.
   **Rejected** the review's assertion that the blend never wins a
   January fold. In 2019 its MAE is 1391.369185 MW, below the lag model's
   1434.748708 and calendar model's 1441.690546. The source is
   `backtest_metrics.csv`, the three learned rows for 2019-01-01.
   Also did not import "anti-correlated fold errors": component biases
   have matching signs across these folds, and no such correlation
   analysis was performed. The report states only that the lowest-error
   candidate changes by year and selection uses the three-year mean.
2. **Accepted, with qualification.** Added a sentence that weekday
   Epiphany examples are scarce and downweighted, making 6 January
   particularly uncertain. The calendar model includes the three older
   weekday dates; the lag model drops January 2015 with its first
   672 training hours, so its evidence is thinner still. No unverified
   load-share number or exact sample count was added to the report.
3. **Partially accepted as test hardening, not a forecast defect.** Added
   direct assertions for Christmas distance and the New Year holiday
   kind. The independent artifact checks already establish this run's
   output range. The remaining suggestions identify non-exhaustive unit
   coverage, not wrong calculations in the delivered forecast. No
   unrequested modelling redesign or production-hardening programme
   was undertaken, and no claim of exhaustive mutation coverage is made.
4. **Accepted.** Replaced the literal character in the prose guard with
   `chr(0x2014)`. This preserves the check while keeping literal-character
   scans clean.

The final full run, tests, and forecast SHA assertion passed after these
changes. None changed the forecast values or model selection. The independent
peer's focused follow-up is recorded in `review_followup.md`.
