# Independent forecast review and disposition

Astra authored the forecasting code and report. A separate Claude/Fable peer reviewed the code, tests and outputs read-only. The following is the peer's returned report, followed by the orchestrator's independent evaluation. The reviewer did not inspect January 2020 actual load values.

## Original peer report

Tests pass (7 passed, 1.13 s). Outputs, run log and backtest files checked; I did not open any 2020 target values (backtest_predictions.csv max timestamp is 2019-12-31 23:00Z, and I only read the 2020 rows of forecast.csv, which are model output).

### Verification summary

**Leakage / causality: clean.**
- `load_history` (forecast.py:43-48) masks rows `< cutoff` before `to_numeric`, and the completeness check (line 51-53) pins the series to end at 2019-12-31 23:00Z. `historical_rows` 43824 = 5 y x 8760 + 24 (2016 leap): correct.
- `predict_candidates` (line 167) and `build_tree_features` (line 140) both re-truncate at `origin`, so backtests are causal independently of the caller. Lags 168..672 and 8736..8784 plus rolling means anchored at t-168 (line 157-159) mean the last target hour (origin+167h) reads origin-1h at most. Test at test_forecast.py:58-67 pins this for the tree path.
- The ridge level correction (line 180-186) uses only `train.index` (< origin) with holidays and Dec 21-Jan 6 excluded, so Christmas does not drag the level.
- Selection windows (all end by 2019-10-30) and calibration windows (2019-11-06 onward) do not overlap; calibration is done only on the already-selected model, so widths are not selection-contaminated.

**Calendar / timezone: correct.** Easter-derived dates for 2019/2020 verified (Good Friday 2020-04-10, Ascension 05-21, Whit Monday 06-01); Reformation 2017 one-off is correct; Epiphany is treated as regional-only (line 93, 130), which is right for a national series. `effective_dow=6` on holidays, weekday features come from `Europe/Berlin` local time, DST test (test_forecast.py:80) passes. Target week locally: Wed 01 Jan holiday, Mon 06 Jan Epiphany flag, first stamp 01:00 CET.

**Output schema / arithmetic: consistent.** forecast.csv: 168 unique hourly rows 2020-01-01 00:00Z to 01-07 23:00Z, positive, 7 columns. Bands equal forecast ± metadata half-widths (2692.9 / 5155.6 MW) to within 0.08 MW (independent rounding). daily_summary: 7 UTC days x 24 h, `energy_mwh` recomputed from forecast.csv to 0.0 difference; total 9,300,151.6 MWh matches metadata and the daily sum; MW summed over hourly steps is MWh, and the OPSD ENTSO-E column is MW, so units are right. Peak 71,847.1 MW at 2020-01-07 16:00Z, min 38,731.9 MW at 01-01 06:00Z match the CSV. Model-selection table recomputes from backtest_metrics.csv; blend wins both groups (autumn 1029 vs 1048 ridge; New Year 2174 vs 2382), so the 0.75/0.25 weighting is not load-bearing.

### Bugs (none blocking)

1. forecast.py:222-226, cosmetic: bands are computed from unrounded `predicted`, then rounded separately, so `upper_80_mw - forecast_mw` is not exactly the stated width (drift up to 0.08 MW). Round once or derive bands from the rounded forecast.
2. forecast.py:163, harmless but misleading: `year`, `day`, `utc_offset` are passed to the tree as features. `year=2020` lies outside the 2017-2019 training range and is simply treated as 2019; it adds no information but is a silent extrapolation feature. Same class: the ridge `trend` (line 111) extrapolates linearly one year past the fit window.

No incorrect code paths found in fitting, selection, or output writing.

### Limitations to state alongside the forecast

1. **Interval undercoverage in the target regime.** Calibration weeks are Nov/Dec Wednesdays; pooled calibration coverage is 80%/95% by construction. Applied to the three New Year selection weeks the same widths cover only 68% (80% band) and 94% (95% band); the empirical New Year q80 is ~3.7 GW vs the shipped 2.7 GW. The 80% band is optimistic for a New Year week (advisory: the New Year weeks were used for selection, so this is a pessimistic-but-indicative check, not a clean one).
2. **Bias, not just spread.** Blend errors are strongly serially correlated (lag-1 autocorr 0.94) and week-level biased: Dec 18 week -2.9 GW, Dec 25 week -2.2 GW (over-forecast), Jan 2018 week -2.4 GW, Jan 2017 +1.5 GW. Symmetric bands around the point forecast hide this; a whole-week energy error of ±2-3 GW x 168 h (roughly ±0.4-0.5 TWh on 9.3 TWh) is realistic.
3. **Thin evidence for the special days.** In the 3-year fit window, Epiphany fell on a weekday only in 2017 (Fri), so the Mon 6 Jan 2020 dip (59.6 GW mean vs 63.8 GW on Tue 7 Jan) rests mainly on the `regional_epiphany` and `winter_date` dummies with one weekday example. The New Year holiday-hour dummies (line 121) have three observations each.
4. **Effective sample size.** Model selection: 6 weeks, 5 candidates; calibration: 5 weeks with near-unit autocorrelation, so effectively ~5 independent error draws. The `interval_caveat` in metadata already says this; I would add the regime-coverage numbers above.
5. **No weather / temperature.** Not stated as a bug; the metadata caveat mentions it. A cold snap in early January would move load by several GW outside the bands.
6. The 2017-01-01 selection origin trains on only two years (data starts 2015), undisclosed in metadata.
7. Day framing: daily summary is UTC days (`date_utc` labelled). Local German days shift by one hour; the Jan 1 local day misses its first hour (2019-12-31 23:00Z), which is outside the horizon by the stated assumption. Fine as framed, but worth one sentence in the deliverable.

### Improvements (non-blocking)

- Test gaps: no test covers `national_holidays` Easter arithmetic, `predict_candidates` leakage end-to-end, or that `make_forecast_table` bands equal forecast ± width. The tree-leakage test covers only the tree path.
- Report coverage per regime (New Year weeks vs autumn) in `backtest_metrics.csv` or metadata rather than only pooled widths.
- Consider asymmetric or bias-adjusted bands, or at least report the per-week bias table.

Sanity check performed: re-ran the suite (7 passed), recomputed daily energy, totals, peak/min, band widths and selection scores from the shipped CSVs, and verified the holiday and DST calendar by executing `calendar_frame`/`national_holidays` directly. No code or outputs were altered.

result: Astra forecast.py verified clean of leakage, calendar and arithmetic errors (7 tests pass, outputs internally consistent); two cosmetic issues, no blocking bugs; main caveat is the 80% band covers only ~68% of New Year backtest hours and week-level biases of 2-3 GW are not captured by symmetric bands.

## Orchestrator evaluation and actions

### Claimed bugs

1. **Rounding: acknowledged, not a defect requiring a change.** Predictions and interval endpoints are independently rounded to 0.1 MW. Sub-0.1 MW endpoint differences are expected presentation rounding and negligible relative to errors measured in thousands of MW. No model or CSV change.
2. **Year/trend extrapolation: disagree with the stated diagnosis.** The orchestrator executed `calendar_frame()` on the last two training timestamps. The final training observation, 2019-12-31 23:00 UTC, is already 2020-01-01 00:00 in Berlin, so the tree training features do contain local `year=2020`. Even without that row, a tree's inability to extrapolate is a model limitation, not evidence that a historical year feature adds no information. `day` and `utc_offset` take ordinary in-range values. The ridge trend is continuous elapsed time, not a discrete year category: the final target is only **168 hours**, not one year, after the final training observation. No code change.

### Limitations

1. **Accept the regime-coverage warning.** Recomputed from archived blend residuals: New Year coverage is **67.857% / 93.849%** for the nominal 80% / 95% bands; New Year absolute-error q80 is **3739.96 MW**. Added these caveats to README. Do not label this diagnostic "pessimistic": selection dependence and regime change do not establish a direction of statistical bias. The bands remain explicitly approximate; no target actuals were used to retune them.
2. **Accept persistent bias; do not promote an illustration into an interval.** Independently recomputed December 18 bias **-2881.97 MW**, December 25 **-2164.29 MW**, and within-week lag-1 error correlation **0.9410**. Added bias examples to README and archived `weekly_bias.csv`. No statistically calibrated weekly-energy uncertainty is claimed.
3. **Accept limited special-day evidence, but not the unmeasured attribution.** Epiphany is represented and the samples are sparse. No feature attribution was performed to establish that the Jan 6 versus Jan 7 difference is mainly caused by particular dummies; weekday patterns and lagged loads also differ. No such causal attribution appears in the deliverable.
4. **Accept small, dependent calibration sample.** Five weeks and 840 dependent hourly residuals are disclosed. Do not assert an effective sample size of exactly five without an estimation method.
5. **Already handled.** No weather information is used, and the report warns about temperature-driven distribution shifts. The size of a hypothetical cold-snap effect was not estimated.
6. **Accept clarification.** README now explicitly states that the January 2017 fold has only two years of training history. `fit_window_years=3` is a maximum lookback, not an assertion that each fold has three available years.
7. **Already handled.** README specifies the UTC horizon, its CET equivalent, and the one-hour difference from a German-local calendar week.

### Suggested improvements

The existing seven tests and peer inspection were sufficient to verify this deliverable. Additional Easter/date unit cases and a full-model future-value mutation regression would be useful for a maintained forecasting package; they were not added to this one-off analysis. Per-regime interval diagnostics and weekly bias were computed and archived as `interval_diagnostics.csv` and `weekly_bias.csv`. Asymmetric interval redesign was not undertaken after inspecting these diagnostics, and no additional calibration-validation claim is made.

### Final state

Forecasting code and hourly forecast values are unchanged from the reviewed run. Documentation was clarified, the initial whole-file input audit was distinguished from the forecasting pipeline's strict numeric cutoff, and independently recomputed diagnostic tables were added. Parent-side checks reconfirmed 168 unique contiguous hourly rows, mean **55,358.045 MW**, energy **9,300,151.6 MWh**, and peak **71,847.1 MW** at **2020-01-07 16:00 UTC**.

## Final narrow peer recheck

The same independent reviewer then checked the documentation folds and chart. Its returned report follows:

Narrow check complete. Everything I could recompute matches the README and the chart.

**README folds, verified against outputs/interval_diagnostics.csv, outputs/weekly_bias.csv and a fresh recompute from backtest_predictions.csv:**
- New Year coverage 67.9% / 93.8%: diagnostics give 67.857 / 93.849; my own recompute gives the same. Correctly labelled as a regime diagnostic on selection weeks, not independent validation.
- Dec 18 / Dec 25 weeks "over-forecast by an average 2.882 GW and 2.164 GW": weekly_bias.csv holds actual-minus-forecast -2881.97 / -2164.29 MW, so the sign wording (over-forecast) is right.
- Jan 2017 fold two years of history: correct (history starts 2015-01-01; a 3-year window from 2017-01-01 reaches back to 2014).
- Scope item 4 (whole-file checksum and audit covering coverage, missingness, value range; no target-period score): internally coherent and consistent with `load_history`. I cannot verify what the initial audit did beyond that statement; it does disclose that 2020 values were examined in aggregate at the audit stage, which is a stated fact rather than a discrepancy.
- Other numbers re-derived: New Year MAPE 4.11% (4.109), calibration pooled MAE 1.708 GW / MAPE 3.21% (1707.7 / 3.213), 840 = 5 x 168 errors, band widths 2.693 / 5.156 GW, selection table, daily GW table, peak/min hours and CET conversions, Python 3.14.6. All match.

**Plot (outputs/forecast.png, plot_forecast.py):** MW/1000 to GW is applied to line and both bands (lines 42-48), y-label "Load (GW)" is right, x ticks are placed at 12:00 UTC with UTC formatter and weekday labels (1 Jan 2020 = Wed, correct), footnote "UTC (Germany is UTC+1)" is correct for January, "five held-out 2019 weeks" matches the calibration design, subtitle origin matches CUTOFF.

**Only non-substantive note:** the chart annotates "Peak 71.8 GW" (`.1f` on 71.847) while the README says 71.85 GW. Same value, different rounding; not a discrepancy of substance.

On the refutation of my earlier item 2: accepted. The last training stamp 2019-12-31 23:00Z is local 2020-01-01 00:00 CET, so `year=2020` is in training, and the ridge trend extrapolates at most 168 h. That item was overstated.

Sanity check performed: read README.md, plot_forecast.py and the rendered PNG, recomputed every quoted statistic from the shipped CSVs, and confirmed no 2020 target values were read (only forecast outputs and pre-2020 backtest rows).

result: README and chart folds verified: New Year coverage 67.9%/93.8%, Dec 18/25 biases, two-year Jan 2017 history, MAPE/MAE figures, GW units and UTC labels all match the diagnostics; no substantive discrepancies, only a 71.8 vs 71.85 GW rounding difference between chart and README.
