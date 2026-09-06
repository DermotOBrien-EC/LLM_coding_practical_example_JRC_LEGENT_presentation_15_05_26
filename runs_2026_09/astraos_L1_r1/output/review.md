# Independent review and disposition

The forecasting implementation was authored by the main Astra session and
reviewed by one non-author Claude/Fable peer. No reader fleet was used.
The report below refers to the pre-format source line numbers. No model was
retuned after the review and no target-week actual loads were inspected.

## Original peer report

### What I actually verified (all read-only, no 2020 load values inspected)

- Ran `test_forecast.py` + `test_forecast_artifacts.py`: 8 passed.
- Reran `forecast_model.py` into a temp dir: all four CSVs byte-identical to the delivered ones, metadata identical except the `versions` block (which also matched the live venv: numpy 2.4.4, pandas 2.3.3, sklearn 1.7.2, lightgbm 4.6.0, holidays 0.96).
- Recomputed `training_sha256` from `load_history`: matches metadata. Training = 43,824 rows, 2015-01-01 00:00 to 2019-12-31 23:00 UTC, no NaNs, all positive.
- Independent leakage probe: overwrote every post-origin load with a constant and reloaded; history identical. The artifact test does the stronger version (non-numeric sentinel, full rerun, byte-identical outputs). Both pass.
- Calendar: target maps to 2020-01-01 01:00 CET through 2020-01-08 00:00 CET; per-local-day features are correct (Jan 1 national holiday/non-workday, Jan 4-5 weekend, Jan 6 epiphany=1 and national=0, Jan 8 00:00 local is weekday 2 workday). `holidays.Germany` without subdiv returns the 9 national days only, so the separate Epiphany flag is needed and present.
- Lags: min lag 168 h; `target[-1] - 168h` = 2019-12-31 23:00, the last training hour. No target-window actual can enter any candidate. Backtest folds train strictly before each fold origin (`forecast_model.py:192`) and score against pre-2020 actuals from the same history object (`:193`).
- Output contracts: 168 unique consecutive UTC hours, columns `utc_timestamp,germany_timestamp,forecast_mw,lower_90_mw,upper_90_mw`, lower<forecast<upper everywhere, `germany_timestamp` equals the tz-converted UTC column, daily CSV recomputes from forecast.csv, `model_comparison.csv` equals the mean of `backtest_metrics.csv`, peak/min/energy in metadata match forecast.csv.
- README numbers: all table values, 72.4% baseline improvement (1 - 2155.718/7823.683 = 0.7245), per-year MAEs, offsets, 9.009 TWh, boundary times, and the sign conventions (bias = forecast - actual in `score()`; residual = actual - forecast in `backtest_predictions.csv`) all check against the artifacts. Zero em-dashes.
- Level plausibility using pre-2020 folds only: Jan 1-7 actual weekly means were 57.7/54.9/55.9 GW (2017/18/19); the 2020 forecast weekly mean of 53.6 GW is consistent with the selected model's -309 MW mean bias, a mildly declining series, and the weekend falling on Jan 4-5.
- ruff check: clean. ruff format: one reformat pending (coordinator already knows). mypy --strict (with `--cache-dir=/dev/null`, the "INTERNAL ERROR" on the first run was only the sandbox refusing the cache DB): 11 errors, detailed below.

### Blockers

None. I found no leakage path, no calendar misalignment, no output-contract violation, and no metric miscalculation.

### Caveats (should be disclosed or fixed, ordered by weight)

1. **Interval coverage is heterogeneous across folds and the README states only the pooled number** (`forecast_model.py:214-215`, `README.md:106-116`). Applying the pooled 5/95 offsets (-3761/+4754 MW) back to the selected model's own fold residuals gives coverage of 75.6% (2017), 94.0% (2018), 99.4% (2019). Lag-1 residual autocorrelation is 0.85-0.96. The README already says "not independently calibrated" and "correlated", but a reader would reasonably assume something near 90% in a typical week; the honest statement is "76% to 99% across the three selection folds". Medium.

2. **Model selection on three folds is rank-unstable** (`backtest_metrics.csv`). MAE ranks by year: lagged_boosting 1/4/1, calendar_boosting 2/2/3, calendar_ridge 4/1/4, ensemble 3/3/2. The prespecified rule was followed correctly and the README says the top-two gap is small; I only note that the 52 MW gap between calendar_boosting and the ensemble is well inside fold-to-fold noise, so "selected" should not be read as "demonstrably best". Medium, disclosure only.

3. **Project quality gate does not pass** (`pyproject.toml:28-30` sets `strict = true`; global rules require `mypy --strict`). 11 errors: sklearn untyped (5), `pandas-stubs` not installed in the venv, `holidays.Germany` not visible to the type checker (`forecast_model.py:53`; `holidays.country_holidays("DE")` is the typed path), `holidays.__version__` not exported (`:260`), and `tmp_path` fixtures unannotated (`test_forecast.py:18,31`). Runtime is unaffected. Also `scikit-learn` is imported directly (`forecast_model.py:13-18`) but is not a declared dependency in `pyproject.toml:6-15`; it is only present transitively via `u8darts[all]`. Medium for project hygiene, nil for forecast correctness.

4. **Leap-year misalignment of the year-lags in the 2017 fold** (`forecast_model.py:23,144-150`). `target - 8760h` from 2017-01-01 lands on 2016-01-02 because 2016 had 8784 hours, so `lag_8760` means "calendar date minus one day" in that fold and in every training row spanning 2016, while it means the same calendar date for the 2020 origin. Affects only `lagged_boosting` and the ensemble, neither selected. Low.

5. **Composite calendar codes are fed to LightGBM as ordinal numerics** (`forecast_model.py:92-95,151-164`): `hour_kind`, `month_hour`, `year_end_block`, `epiphany_hour` are one-hot only for ridge (`:123-128`); the boosting models receive them as integers. Trees can still isolate values, so this is a modelling choice with no correctness defect, but it is not the "calendar features" a reader might picture. Low.

6. **Fold heterogeneity in training length**: the 2017 fold trains on two years and produces the worst residuals for every model; those residuals enter the pooled band at equal weight while the delivered model has five years. Partly conservative, partly why item 1 is skewed. Low.

7. **Robustness of the loader to post-origin garbage is partial** (`forecast_model.py:32`): unparseable post-origin *load values* are tolerated (tested), unparseable post-origin *timestamps* would still raise in `pd.to_datetime` on the full column. Not relevant to the supplied file. Low.

8. Reproducibility was verified on this machine only; LightGBM byte-identity across platforms/BLAS builds is not guaranteed, so the README's "complete reproducible pipeline" (`README.md:131`) should be read as same-environment reproducibility. Low.

### Things I checked and found correct that a reader might doubt

- The weekly-naive baseline's -7.4 GW bias is the Christmas-week copy effect the README describes, not a bug.
- `trend_years` extrapolation is 0.0001 years beyond training range; irrelevant for trees, negligible for ridge.
- Ridge level correction (`forecast_model.py:134-136`) uses only the last 56 days before the origin, excluding Dec 20-31 and holidays, so no target data.
- `daily_forecast.csv` is UTC-day aggregation; README says so and warns against reading it as German civil days.

Sanity check before closing: I re-read the ask (leakage, calendar timing, validation, uncertainty claims, code correctness, output contracts, README included per coordinator) and confirmed every command I ran either filtered to pre-origin rows or masked/overwrote 2020 load values without printing them.

result: Non-author Claude-peer review of the 7c2cbf load forecast finds no blockers (leakage, calendar, validation and output contracts all verified by rerun and independent probes); caveats are the undisclosed 76-99% per-fold band coverage, rank-unstable 3-fold selection, a failing mypy --strict gate with an undeclared scikit-learn dependency, and a leap-year lag misalignment in an unselected model.

## Main-session evaluation and changes

1. **Accepted.** Recomputed coverage directly from the saved residuals and
   metadata offsets: 75.595238%, 94.047619%, 99.404762%. Lag-1 correlations
   recompute to 0.961715, 0.850635 and 0.882027. Added per-fold coverage to
   README, explicitly describing it as reuse of the selection data, not an
   independent coverage check. No forecast or interval was retuned.
2. **Accepted, already substantially disclosed.** The saved metrics confirm
   the unstable rankings and small top-two gap. README already states the
   small difference, variation across years and model-selection limitation.
   Retained the prespecified selection rule rather than changing it after
   seeing the scores.
3. **Partially fixed; environment work deferred explicitly.** Added `Path`
   types to both fixtures, replaced the dynamic holiday class lookup with
   `country_holidays("DE")`, and read the package version through
   `importlib.metadata.version`. Executed a calendar-equality assertion for
   2015–2020: the two holiday constructors give identical calendars. Applied
   ruff formatting. Missing pandas/sklearn typing information is an
   environment limitation and remains disclosed in README; no parent
   dependency manifest or shared environment was modified. Standalone
   scikit-learn installation is now explicitly documented. Strict typing is
   not claimed to pass.
4. **Accepted as a modelling caveat, not a timestamp error.** The lag is
   explicitly 8,760 hours, not a same-calendar-date lookup. That definition
   works as implemented and uses no future observations. README now spells
   out the fixed-hour rather than leap-adjusted interpretation. No alteration
   to an unselected candidate after model selection was justified.
5. **Accepted as a modelling choice.** README now states that trees receive
   numeric composite codes and ridge receives one-hot features. No claim of
   ordered physical meaning or categorical LightGBM handling is made.
6. **Accepted as a limitation of expanding-window validation.** README now
   notes the different amounts of training history. Retained genuine earlier
   forecast origins rather than borrowing later data to equalize history.
7. **No change needed for this input.** A timestamp must be parseable to
   determine which side of the cutoff it belongs to. Failing on malformed
   timestamps is safer than silently guessing or dropping records. The
   documented guarantee is independence from post-origin load values, not
   tolerance of an arbitrary damaged CSV.
8. **Accepted.** README now bounds byte-identical reproduction to the supplied
   environment and makes no cross-platform guarantee.

Two details of the review narration are corrected here rather than silently
carried forward. The executable artifact test compares **five** CSV files,
not four. Also, the final horizon endpoint is 168 hours beyond the last
training observation, approximately 0.0192 years, not 0.0001 years; the latter
is only the first forecast hour's gap. Neither detail changes the verified
leakage boundary or any forecast value. The review's load-level plausibility
comment is qualitative, not additional validation evidence.

No claim of passing strict static typing is made. Runtime tests, deterministic
CSV reproduction and the independent leakage checks are the forecast's
execution evidence.

## Original peer recheck report

Recheck complete. Everything I looked at is read-only and no 2020 load values were touched.

**Forecast unchanged: confirmed.** All five CSVs are byte-identical to the rerun I made from the pre-fold code, and `forecast_metadata.json` is unchanged except that it still carries the pre-fold `versions` block (file mtime 03:51, code mtime 04:01). That is fine: the artifact test's poisoned full rerun runs the post-fold code and asserts the five CSVs byte-identical, and it passes, so the post-fold code produces the same forecast. The only note is that the delivered metadata was not regenerated after the fold; since the only behavioural changes are the holiday constructor (equal calendars) and the version-string source (same installed package), this is provenance trivia, not a defect.

**Code folds, each verified:**
- `forecast_model.py:54` now uses `holidays.country_holidays("DE", ...)`. I independently compared it with `holidays.Germany` over 2015-2020: identical dicts, 55 entries.
- `forecast_model.py:6,275` reads the holidays version via `importlib.metadata.version`.
- `tmp_path: Path` annotations on `test_forecast.py:20,33` and `test_forecast_artifacts.py:44`.
- ruff check clean, ruff format reports all three files formatted.
- `mypy --strict` on all three files: exactly 8 errors, all `import-untyped` for pandas and sklearn; the `holidays.Germany`, `__version__`, and fixture errors are gone. README states this honestly rather than claiming green.
- 8 tests pass (19 s).

**README folds, each present and numerically correct:** per-fold coverage 75.6/94.0/99.4 with the "descriptive, not an independent coverage test" qualifier (`README.md:117-120`); differing training history (`:120`); fixed-hour lag semantics (`:75-76`); numeric tree codes vs one-hot ridge (`:76-78`); same-environment reproduction bound (`:151-152`); scikit-learn transitive-only with standalone install note (`:152-154`); deferred stubs (`:157-161`). Zero em-dashes in README.md and review.md.

**review.md corrections to my report: both correct.** The artifact test compares five CSVs (I had compared four in my own rerun and wrote "four"). The horizon's last hour is 168 h past the last training observation; I recomputed the `trend_years` gap as 0.00011 years at the first hour and 0.01916 at the last, so "~0.0192 years" is right and my 0.0001 described only the first hour. Neither changes any conclusion; the extrapolation is still negligible for ridge and irrelevant for trees.

**Dispositions:** all eight match my findings' weight. Item 7 (malformed post-origin timestamps raise) is correctly declined; item 3's deferral of the parent `pyproject.toml` (scikit-learn is still undeclared there, I checked) is explicit in both README and review.md, which is the acceptable form of a deferral.

**Blockers:** none. No fold touches model fitting, selection, or interval estimation.

Sanity check: re-read the coordinator's list of changes against the diff regions I inspected; every listed change is present, nothing unlisted changed in the model code beyond formatting, and the forecast outputs are provably unchanged.

result: Recheck confirms all applied folds faithfully address the peer-review findings, the forecast CSVs are byte-identical to the pre-fold output, tests/ruff pass, mypy is honestly disclosed at 8 import-untyped errors, both corrections to my report are right, and there are no blockers.
