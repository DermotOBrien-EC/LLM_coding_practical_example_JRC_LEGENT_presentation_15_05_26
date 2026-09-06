# Review and execution record

## Authority and original protocol

The operator explicitly requested unattended, end-to-end delivery. No approval
or response from a human was assumed. No package installation, commit, push,
or modification outside this run directory was requested or performed by the
study code. Temporary files and plotting/Darts caches are redirected beneath
this directory.

The original implementation plan was to verify the exact input grid, create
consumer-facing leakage tests, implement one module per model, select using
rolling seven-day validation, refit on all pre-test observations, and score a
single fixed-origin test week only after every forecast exists. The primary
comparison is MAPE; absolute metrics remain in MW. Forecast tables, candidate
scores, versions and input digests supplement the requested figures and short
methods transcript.

## Independent Claude-peer protocol review

The implementation author is the Astra session. An independent Claude peer
reviewed the proposed method and installed-library contracts before the full
model run. Its verdict was that the specified protocol had no leakage blocker.
Its recommendations and the author's decisions were:

1. **Recursive LightGBM uncertainty:** accepted the underlying finding that
   quantile fits applied only to a single simulated median history omit
   uncertainty in the history itself. Instead of dropping the prescribed lag
   and rolling features, the implementation simulates 1,000 load paths, each
   updating its own features. Conditional quantiles are sorted, interpolated
   on a normal-score axis, and explicitly extrapolated in the tails. These
   intervals still assume independent conditional draws and are not claimed
   to include all temporal dependence or parameter uncertainty.
2. **Validation interval calibration for all models:** deferred as additional
   scope, not a required selection criterion. No interval calibration is
   performed using the test week. Test coverage and width are provided for
   every probabilistic model, beyond the requested winner-only summaries.
3. **Weekly structure in daily SARIMA:** accepted. Three weekly Fourier pairs
   computed from weekday/hour accompany the daily seasonal AR term. Annual
   terms and a holiday dummy were not added; the statistical comparison keeps
   a small prespecified candidate set.
4. **Use Europe/Berlin calendar time:** retained UTC consistently with the
   supplied series, test partitions and Prophet's regular hourly grid. The
   daylight-saving limitation is stated rather than hidden. No extra timezone
   covariate was introduced after seeing scores.
5. **Richer Prophet seasonality candidate:** accepted. The second candidate
   changes daily/weekly Fourier orders as well as the changepoint prior.
   Prophet does not update a recent-load state, so predicting validation once
   is equivalent to slicing its unchanged fitted calendar curve by origin.
6. **Gaussian neural forecast parameters:** accepted. The point mean and
   quantiles come from the public likelihood-parameter API rather than noisy
   samples. The custom epoch evaluator is checked against that public path.
   The suggestion to omit the final 24 validation hours was not adopted:
   all 2,208 hours enter the shared hourly-weighted selection criterion.
7. **Regional German holidays:** not adopted because the requested indicator
   is federal. The write-up distinguishes Jan 6's regional status and does not
   mislabel the weekend within Jan 2–7 as working days.
8. **SARIMA final refit:** the review suggested that merely extending the
   filter could suffice. Rejected: the user explicitly requires parameter
   refitting on Train + Validation. The implementation calls a new full fit.
9. **Mixed-unit metric plot:** accepted normalization to baseline = 100, with
   original percentage/MW values on bars. A shared raw scale for MAPE and MW
   would be misleading.

The default-stack N-BEATS model is retained. PatchTST is unavailable in Darts
0.41.0, so the sixth slot is explicitly labeled TransformerModel; TSMixer was
not used because it is not transformer-based. These choices precede test
scoring. The annual LightGBM lag is the permitted 364-day offset (8,736 hours).
The prompt's example about an 8,760-hour lag timestamp is not used as a
calculation: 8,760 hours before 2020-01-01 00:00 UTC is 2019-01-01 00:00 UTC.

## Runtime corrections before test scoring

- The first Prophet launch failed before fitting because the Darts constructor
  keyword is `suppress_stdout_stderror`, not `suppress_stdout_stderr`. The
  installed constructor source was read, the keyword corrected, and the full
  selection/refit/prediction run completed. The failed log is preserved as
  `prophet_run.log`; the successful retry is `prophet_retry.log`.
- The first SARIMA launch fit its first candidate but failed at validation
  state extension: `low_memory=True` removes `predicted_state`. The fit now
  retains state (`low_memory=False`) so `extend` can advance through observed
  validation history without refitting. The failed log is `sarima_run.log`.
- The original six consecutive tab10 colors failed the palette validator's
  adjacent color-vision separation/chroma checks. The fixed subset/order
  blue, orange, purple, green, pink, red passes those checks. Lower-contrast
  orange/pink marks have explicit labels and numeric tables as a second
  access channel. Forecast comparisons are faceted by named model.

## Implementation and deliverable verification

The initial five consumer tests were written first and failed because the
implementation did not yet exist. After implementation, all five passed:
exact data/splits, validation partitioning, metric units and quantiles,
current/future-target feature invariance, and agreement between training and
recursive feature calculations. The modeling source was handed to the same
independent Claude peer for a read-only review. Prophet and SARIMA were
explicitly withdrawn from that frozen subset while the two runtime API
corrections above were applied; a narrow reread is required afterwards.

### Modeling-code review result

The independent Claude peer read the frozen modeling source and the relevant
Darts internals, ran the five tests, and exercised LightGBM simulation with a
fake-model oracle. It returned **no blockers**. It confirmed:

- Models receive only the pre-test slice; validation contexts stop before each
  forecast origin and rolling features exclude the current observation.
- Fresh neural builds, fresh scaling and five fresh LightGBM fits implement
  the required final refits. N-BEATS architecture metadata matches installed
  defaults; training-window counts and the recorded early-stop epochs agree.
- The neural callback's raw Gaussian mean is the same quantity produced by
  the public Darts likelihood-parameter forecast path. A runtime assertion
  also checks their equality after refitting.
- Each LightGBM simulation updates its own history. Feature order and rolling
  sample standard deviations match the one-step feature builder.
- Quantile dimensions, MW scaling, coverage indices and pinball formulas are
  correct, and save-time checks reject non-finite or crossed final quantiles.

The peer requested disclosures about quantile crossing, the difference
between LightGBM's point path and marginal median, package versions,
concurrent preliminary runtimes and Christmas-week naive normalization.
All are accepted. The short report includes the two LightGBM diagnostics,
versions are recorded by the report generator, and a separate complete
sequential execution replaces contention-affected preliminary timing.
Christmas-week copying is explicit in the discussion. All five probabilistic
models receive an additional coverage/width/pinball CSV, not only the winner.
The peer's suggestion that rougher simulated features explain observed
quantile crossings was not accepted as an established causal result. The
report says independent innovations *can* produce unrealistic roughness.

### Corrections after modeling review, still before test scoring

The selected SARIMA training fit converged; the initial final refit exhausted
60 iterations. Its final optimization budget was therefore increased to 180
iterations, without changing the selected order or looking at test scores.
Training-candidate budgets remain 60; non-converged candidates are explicitly
ineligible for selection. This is a numerical convergence correction, not a
hyperparameter change selected on test performance. Final convergence is
reported from the actual optimizer result.

Saved model metadata now includes hashes of the actual model source and
shared numerical utilities. `--report-only` rejects source mismatches instead
of giving stale forecasts the current code's provenance. Code was formatted
with Ruff and unused imports removed. Ruff checks and five pytest tests pass.
A strict mypy attempt failed with an internal error in installed mypy 1.20.2
(`mypy_check.log`), so strict static typing is **not** claimed as verified. No
package was installed or updated to resolve that tool failure.

Prophet, SARIMA, the source-digest addition and the new reporting/verifier
modules were then submitted for an independent narrow reread. The full
sequential execution runs the same published one-command entry point.
### Reporting review, findings and resolution

The independent reviewer found two reporting defects, both accepted:

1. Original finding: the validation table used the minimum score across all
   candidates, while SARIMA selection allowed only converged candidates. A
   synthetic ineligible candidate with MAPE 5.00 made the table misreport a
   selected converged candidate whose MAPE was 6.30. The actual data happened
   not to trigger this, but the code was wrong. Resolution: filter by
   `item.get("converged", True)` and label SARIMA selection as "best converged
   candidate". The peer confirmed both the code and rendered 6.32% row.
2. Original finding: the unconditional sentence "Holiday weakness is therefore
   measured rather than assumed" overclaimed when few or no models had worse
   Jan 1 error. Resolution: say whether Jan 1 is harder is a measured
   per-model result, not an assumption. The peer confirmed the rendered count
   of three and the removal of the overclaim.

The remaining reviewed reporting code was confirmed faithful: UTC test slice,
per-day scores, all-model interval statistics, quantile-crossing percentage,
version discovery, data/figure digests, pixel dimensions and the independent
verifier's formulas. No modeling or selection settings changed after test
scoring. The metric-chart legend was moved to empty space at the upper right
following visual inspection, and every figure was regenerated through the
report entry point, not edited by hand.

### Final execution and independent closure

The complete sequential command finished successfully. Its fit/forecast phase
was 580.9 seconds; the original full command, including rendering, was 582.74
seconds, comfortably below 30 minutes on the recorded machine. The final
layout-only refresh records its own rendering cost added to the unchanged
fit/forecast duration in metrics.json. SARIMA's selected final fit converged
at iteration 63. The unsuccessful second training candidate remains disclosed
and was not eligible for selection.

The final independent Claude-peer check recomputed these LightGBM results
from the raw input CSV and saved forecast, matching metrics.json at relative
tolerance 1e-10:

- MAPE 5.114978%, RMSE 3,639.16 MW, MAE 2,774.11 MW.
- Jan 1 MAPE 5.41%; Jan 2–7 MAPE 5.07%.
- 80% coverage: 133/168 = 0.791667; 95% coverage: 164/168 = 0.976190.
- Pinball losses q10/q50/q90: 470.63 / 1,213.33 / 501.34 MW.

It also verified the six-model ranking, the rounded 60.0% relative MAPE
reduction, LightGBM's worst day (Jan 6), final optimizer convergence and
runtime provenance. Its final verdict was **no remaining blocker** and no
contradiction among tables, generated discussion and archived artifacts.

Final local checks: five pytest tests pass; Ruff lint and formatting pass;
all six models' five point metrics and daily metrics independently recompute;
winner interval metrics independently recompute; all eight required PNGs
exist at 300 dpi and their prescribed 11x6 or 6x6-inch dimensions; model
source and figure digests match. All eight figures were visually inspected,
including the refreshed metric-chart legend. The transcript contains 1,175
words, including a 309-word Discussion, and no em-dash prose. Strict mypy
remains unverified because of the tool internal error documented above.
