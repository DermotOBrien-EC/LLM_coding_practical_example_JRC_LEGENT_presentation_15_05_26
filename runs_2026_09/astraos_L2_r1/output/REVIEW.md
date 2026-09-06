# Independent review and verification

## Original work

The pre-implementation plan is preserved in `PLAN.md`. The reviewed deliverable
was `forecast_load.py`, `test_forecast_load.py`, `README.md`, both requirements
files, the generated CSV/JSON outputs, and the final PNG comparison plot.
The author froze these files before the independent implementation review.
No forecast settings were changed after observing target-week results.

## Plan challenge

The independent Claude peer approved the plan with four bounded caveats:

1. Explicitly disable automatic early stopping and pin the iteration budget.
2. The previous-week baseline copies Christmas week, so add a 52-week baseline
   and state why the comparison is imperfect. Include a lagged trailing
   seven-day mean.
3. Compute calendar features in Berlin time. Distinguish Epiphany and
   Christmas/New Year's Eve from national holidays, and account approximately
   for bridge days.
4. Test the forecast-origin boundary, keep target actuals out of features and
   selection, and define signed error as forecast minus actual.

Author evaluation: accepted. Those requirements were implemented. Previous-day
and two-days-prior holiday features represent bridge-day context; exact regional
load weights and bridge-day effects are explicitly not claimed. The peer's
proposed direction did not require target-week actuals.

## Implementation review: complete findings and confirmation

The independent peer returned the following report. Formatting is normalized;
no finding has been omitted.

> Verification complete. What I checked: read `forecast_load.py`,
> `test_forecast_load.py`, README, both requirements files, all five outputs
> and the PNG; recomputed every headline number with a stdlib-only script
> against the raw CSV (independent of pandas and the author's code); ran the
> 17 tests, Ruff check/format and strict mypy with `.venv312` (all green);
> probed the `holidays` library's national-vs-Epiphany semantics.
>
> **Verdict: accept with limitations. No blockers, no misleading claims found.**
>
> Confirmed correct, with grounding:
>
> - Leakage boundary: `split_data` keeps `index < FORECAST_START`;
>   `forecast_week` refuses history at or past the origin and requires it to
>   end exactly one hour before; the latest hour any target feature can touch
>   is origin+167h-168h = 2019-12-31 23:00 UTC. Rolling means are trailing and
>   shifted 168h (test at line 76 pins it). `early_stopping=False` literal,
>   `fitted_iterations: 200`.
> - All 168 hours: forecast.csv rows are 2020-01-01 00:00 through 01-07 23:00,
>   continuous, actual column byte-equal to the raw CSV.
> - Baselines: previous_week equals raw[t-168h] and previous_52_weeks equals
>   raw[t-8736h] for all 168 rows; 52-week source is Wed 2019-01-02,
>   weekday-aligned as README states.
> - Metrics: my recomputation matches metrics.json and README to every printed
>   digit (MAE 2592.06, RMSE 3190.54, MAPE 4.7945, WAPE 4.854, ME +2534.81;
>   improvements 64.19% and 52.73%). Daily MAEs and validation mean MAEs
>   (2214.71 / 2274.67 / 3347.15 / 5378.71) reproduce from validation.csv and
>   daily_metrics.csv. Training rows 35064 = 43824 - 8760 (2016 leap year
>   accounted).
> - Fold selection: each fold trains on `history < origin`, windows all
>   precede 2020, 15 leaves wins on mean MAE; frozen before target scoring.
> - Chart: two distinguished lines (solid blue, dashed orange), legend, UTC
>   day ticks, aligned error panel, subtitle numbers match metrics.json, no
>   text overlaps at 2160x1260.
>
> Mutation question: a mutant that trains on, or builds lag features from,
> target-week loads is caught by the end-to-end poison test (forecast_mw must
> be identical after every 2020 load becomes 999999) and by the
> `build_features(extended)` equality at test line 79-80. A mutant that adds a
> target-week fold is caught by the validation.csv equality. The only survivor
> I can construct is contrived: scoring candidates on the target week without
> recording it, where the poisoned argmin happens not to flip. Low.
>
> Findings (all low):
>
> 1. `select_model`'s guard (line 219-220) is exercised only indirectly; a
>    one-line test that `select_model(data)` raises on 2020 rows would pin it.
>    Optional.
> 2. The upward bias (+2.5 GW) was already visible pre-freeze in 4 of 5
>    validation folds (mean_error +1.4 to +3.1 GW, validation.csv). README does
>    not say so; adding one sentence would strengthen the honesty claim, and
>    no configuration change is warranted post-freeze.
> 3. `pd.to_datetime(..., utc=True)` would silently treat naive timestamps as
>    UTC. Irrelevant to this file (all rows carry +00:00), noted for reuse.

## Author evaluation of the findings

1. **Accepted as an optional test-coverage gap, not a current defect.** The
   guard is present and was independently inspected. The end-to-end test
   already checks unchanged historical selection and forecasts after all
   2020 values are replaced. A direct guard test could make future regression
   diagnosis more precise, but no claim is made that every possible mutation
   is detected. No code change was needed for this scoped forecast.
2. **Accepted and disclosed here.** Overprediction was also visible during
   historical validation, not just on the target week. For the selected
   model, the signed mean error in the five folds was approximately +2,847,
   +106, +1,506, +1,464 and +3,073 MW. This suggests systematic bias worth
   studying on a fresh validation design. The frozen target forecast was not
   recalibrated after its error became known. The README already prominently
   reports the target-week bias.
3. **Accepted as a reuse limitation.** The supplied input explicitly uses UTC
   offsets, so no timestamp ambiguity affects this run. An alternative CSV
   with timezone-naive timestamps would be interpreted as UTC, not Berlin
   local time. Such input must be converted correctly before reusing the
   script. No broader timezone-inference capability is claimed.

The review's phrase "actual column byte-equal" is stronger than the numerical
check needed here. Author verification established exact numeric equality for
every actual value after parsing, and exact timestamp alignment. It does not
rely on the two CSV files using identical textual formatting.

## Author's separate verification

- Tests were written before the implementation and initially failed to import
  the not-yet-created forecasting module. The completed suite has **17 passing
  tests**, including the full pipeline's future-data replacement test.
- Ruff checks and formatting pass. Strict mypy passes with the exact command
  in the README. Two matplotlib date helpers lack typed signatures; their
  individual calls carry narrow `no-untyped-call` ignores.
- The initial mypy invocation failed because its SQLite cache could not be
  opened. Using `--cache-dir=/dev/null` and explicitly matching Python 3.12
  resolved the environment issue. The initial `uv` installation route could
  not canonicalize the environment's interpreter; the working environment
  was installed with that interpreter's `ensurepip` and `pip` instead.
- `pip check` reports no broken requirements.
- A stdlib-only recomputation matched all headline metrics to the JSON and
  checked every actual/baseline value against the source at its precise
  timestamp. The complete 168-hour index and input SHA-256 also match.
- Model selection was recomputed from the five scores per candidate in
  `validation.csv`; the selected candidate matches the JSON. The daily report
  has seven rows, each aggregating exactly 24 UTC hours.
- The chart palette passed the supplied color validator. Visual inspection
  caught title/subtitle and footer/axis-label overlaps in the first render;
  spacing was corrected before the final independent review. The final PNG
  was opened and inspected again.
- A fresh full run from another working directory reproduced all three CSV
  files, the JSON report, and the PNG **byte for byte**. SVG byte identity was
  not required because that format can contain generation metadata.

No code or forecast outputs changed in response to the final review. No
commits, pushes, external publications or persistent-memory writes were made.
