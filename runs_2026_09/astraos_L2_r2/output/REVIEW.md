# Review record

## Original plan

Build a fixed-origin, 168-hour UTC forecast for 1–7 January 2020. Compare two
fixed-parameter histogram gradient-boosted tree models, calendar-only and
calendar plus historical load, using the five pre-2020 validation weeks named
in the script. Use German-local calendar features, national holidays,
Christmas/New Year features and a January 6 flag. The initial lag list was
168, 336 and 8,736 hours. Select by mean validation MAE, refit on pre-2020
history, then evaluate the untouched January actuals. Save hourly predictions,
metrics, validation results and a two-line matplotlib plot. Write consumer
tests first, including a held-out-data mutation test.

## Independent plan review

A non-author Claude agent returned the following review before implementation.
Its harness reported Fable 5.1; that backend identity was not separately
verified. The reviewer did not write code.

> Ground truth checked: AGENTS.md only mandates pandas and matplotlib; CSV is
> 50,400 rows, 2015-01-01 00:00 to 2020-09-30 23:00 UTC with explicit +00:00
> offsets, which is exactly the complete hourly count (43,824 h for 2015-2019
> plus 6,576 h for Jan-Sep 2020). The data premise in the plan holds.
>
> Findings on the plan (no material flaw in the leakage or validation design;
> a few necessary small fixes):
>
> 1. Determinism, necessary. `HistGradientBoostingRegressor` has
> `early_stopping='auto'`, which switches ON at n_samples > 10,000 and holds
> out a random 10 percent of the fit rows. Not leakage (all rows are
> pre-cutoff), but predictions then depend on `random_state`. Set
> `early_stopping=False` with a fixed `max_iter` (the model is then fully
> deterministic at this data size, binning subsampling only starts at 200k
> rows), or pin `random_state`. Without this the "predictions unchanged when
> later actuals change" test and the reproducibility claim are fragile.
>
> 2. Contiguity between history and future, necessary. The plan validates
> "history before forecast start" and "future contiguous 168 h" but not that
> `future[0] == history.index[-1] + 1h`. The lag-168 feature for the last
> forecast hour is exactly the last history row, so a gap silently produces
> NaN lags or a misaligned lookup. Enforce the adjacency check and build lags
> by timestamp lookup (`history.reindex(future_index -
> pd.Timedelta(hours=lag))`), never by positional `shift` on a concatenated
> frame. Add a test asserting lag168 of the final forecast hour equals the
> final history value.
>
> 3. Fair comparison of the two variants, small. The calendar-only model can
> train from 2015-01-01; the lag model only from roughly 2015-12-31 (first row
> with a valid 8736 h lag). Either choice is defensible, but state which
> training rows each variant uses in the validation table (row counts per
> variant per fold), so the selection is not confounded by training length.
> Simplest: use the same rows (drop rows without full lags) for both.
>
> 4. Lag 8736 alignment, small optional improvement. 364 days aligns weekday
> but not calendar date: for 2020-01-01 (Wed, holiday) it points to 2019-01-02
> (Wed, bridge day). A 365-day (8760 h) lag hits the same holiday. Since the
> target week is Jan 1 to Jan 7 with Jan 1 and Jan 6, consider a third variant
> adding 8760 and let the same five-fold validation decide. Not required;
> the holiday indicators plus training history of prior Jan 1s carry most
> of this signal.
>
> 5. The five validation weeks are well chosen: the 2018-01-01 and 2019-01-01
> folds share the target's structure (lag-168 window is Christmas week), so
> the lag model's weakness there is measured rather than discovered on the
> test week. Note 2020-01-01 is a Wednesday and all five folds start Monday
> or Tuesday; no fix needed since horizon position is not a feature, just
> mention it in the report.
>
> 6. Freeze the protocol in code: hyperparameters, variant list, and fold
> origins as module constants; final Jan 2020 metrics computed once after
> selection, in the same script run. Persist in metrics.json: selected
> variant, hyperparameters, training row count, last training timestamp,
> per-fold table, and the baseline metrics beside the model on the test week.
>
> 7. Definitions to pin so tests are unambiguous: bias sign (forecast minus
> actual), per-day = UTC calendar day (spec says UTC dates), WAPE = sum|e| /
> sum|y|, MAPE in percent.
>
> 8. Regional holidays beyond Jan 6 (Corpus Christi, All Saints, Reformation
> Day outside 2017) are not in `holidays.DE()` without a subdiv and do affect
> load in training weeks. Out of scope for this target week; state the
> limitation rather than extend.
>
> Everything else (lags >= 168 so no gap needed between training end and
> origin, Berlin-local calendar via tz_convert, UTC parsing with `utc=True`,
> one-call week forecast, sha256/version provenance, two-line plot with
> linestyle encoding) is sound as planned.

## Author evaluation and dispositions

1. **Accepted.** Both early stopping and the seed are explicitly set.
   This removes the random validation split; pinned dependency versions and
   repeat-run comparisons address reproducibility separately.
2. **Accepted.** `fit_forecast` enforces history/future adjacency and rejects
   overlap. `lag_features` uses timestamp lookups. Tests cover the gap,
   overlap and last-hour lag boundary.
3. **Accepted.** Both variants use the identical fitting index after the common
   warm-up. Training row counts are saved per fold and tested for equality.
4. **Partially accepted.** Add the 8,760-hour lag to the existing lag variant,
   rather than introducing a third model. Keep the comparison small and fixed
   before observing test metrics. The warm-up becomes 8,760 hours. Explicitly
   document that 365-day shifts are not universally calendar-date aligned.
5. **Accepted as a limitation, no model change.** Validation origins are Monday
   or Tuesday, while the test origin is Wednesday. Features represent the
   actual target hour and weekday, not just its position within a horizon.
6. **Accepted.** Protocol constants and complete selection evidence are saved.
   Later executions are reproducibility checks, not additional tuning rounds.
7. **Accepted.** Formulas and bias sign are documented, with hand-computed
   test examples. Daily metrics use UTC days.
8. **Accepted.** The regional-holiday limitation is recorded, not expanded into
   a new feature set after looking at the test result.

## Verification scope and limitations

The implementation has executable tests for the leakage boundary and a full
CLI round trip. The CLI test independently recomputes metrics using standard
Python arithmetic, rather than calling the production scoring function, and
matches every actual and baseline timestamp to the source CSV. It also checks
the selected model against the recorded validation scores.

The plot palette passed the supplied validator. The rendered PNG was inspected
for geometry, labels, units and line distinguishability. The original "forecast
made before 1 January" subtitle was revised to explicitly call this a backtest,
so the figure does not imply that the forecast was actually issued in 2019.

The initial mypy attempt failed inside its SQLite cache with
`OperationalError: unable to open database file`. With the documented
`--no-sqlite-cache` option it ran, identifying a dynamic holiday alias and
untyped matplotlib date constructors. The holiday import now uses the concrete
`Germany` class. This scoped strict check passes:

```bash
../../.venv/bin/mypy --strict --ignore-missing-imports \
  --untyped-calls-exclude matplotlib.dates --no-sqlite-cache \
  --cache-dir .mypy_cache forecast_load.py test_forecast_load.py
```

Missing third-party type stubs and matplotlib's untyped date constructors are
excluded from that check. This is not a claim that their APIs were fully
statically checked.

**No independent post-implementation peer approval is claimed.** The required
Claude dispatch health check failed with exit 1 because
`/Users/doob/cliproxyapi/config.yaml` was not readable. No routing configuration
was changed and no alternative permission route was used. The completed plan
review is not being substituted for an implementation review. Runtime tests,
metric recomputation and chart inspection remain the implementation evidence.
