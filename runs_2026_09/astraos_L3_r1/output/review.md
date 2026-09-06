# Independent review and response

## Protocol review

A separate Claude Fable reviewer read the task, CSV and installed library APIs before the experiment produced test scores. Its verdict was that the design was sound, subject to the following concrete challenges.

1. **Darts early stopping is not validation MAPE by default.** Accepted. `ValidationMAPE` evaluates all 14 actual-history contexts at every training epoch, computes the median forecast MAPE over all 2,208 hours, and selects the best epoch. The public Darts prediction and callback forward path are checked for numerical agreement after refitting.
2. **Prophet's frozen fit forecasts validation from 1 to 92 days ahead.** Accepted. Prophet has no state update. Its validation scores select only its own seasonality mode and are not used to compare model classes. The transcript discloses this distinction.
3. **Recursive quantiles must share one median path; rolling variance may collapse; yearly lag drops the first year of feature targets.** Accepted. LightGBM recursively uses its median only, including for all five conditional quantile fits. Crossing correction preserves the median. Its training targets start after the 8,760-hour warmup; the source observations are not imputed or dropped from the dataset. The lag is exactly 8,760 hours, not weekday-aligned.
4. **Use the same training stride during selection and refit, and distinguish a quantile median from sampled forecasts.** Accepted. Both neural fits use stride 168, not a recent-only sample cap. All forecast origin hours are midnight UTC; N-BEATS retains its default 30 stacks. Quantiles come from the public likelihood-parameter prediction interface.
5. **Prophet's UTC calendar blurs local midnight and DST.** Accepted and disclosed. LightGBM's calendar features use Europe/Berlin; Prophet's regular input index and built-in holidays use UTC dates. No timezone shift or imputation is applied to the load grid.
6. **The final validation block is only 24 hours.** Accepted. Pool hours equally, not block means. Preserve per-block scores for inspection.
7. **Naive winner intervals require a separate calibration layer.** Accepted as a contingency. Native naive uncertainty remains absent. If it wins, validation signed-error quantiles supply separately labeled auxiliary intervals, never fitted on test.
8. **Rename the substituted model's machine key to transformer.** Partially accepted. Human-facing labels and `actual_class` explicitly say TransformerModel. Retain `patchtst` as the compatibility key because the requested metrics schema enumerates that key; it never means PatchTST actually ran.
9. **Bound SARIMA search and disclose Gaussian fixed-parameter intervals.** Accepted. Two explicit daily orders are tested on a weekly-pre-differenced series. No holiday regression is added to this classical comparator. This tests a different information representation, not whether SARIMA could benefit from known-calendar regressors.
10. **Explain bridge days, Jan 6's regional status, and mixed-unit metric bars.** Accepted. Jan 2–7 is called the remaining six days, not working days. Metric bars have a single dimensionless baseline-relative axis with raw %/MW labels; numerical tables remain the authoritative raw-scale view.

## Execution issue

The first run failed before any model selection or test scoring: statsmodels `low_memory=True` removed the predicted states required by `extend()`. The fixed implementation retains predicted states while omitting unused filtered/smoothing arrays. The original exception is preserved in `training_attempt1.log`. No model specification changed in response to test results.

## Implementation review

The independent Claude reviewer read all model modules and their consumer tests, checked the relevant Darts/statsmodels source, and inspected forecast shapes without computing test scores. Verdict: no material task violation; six minor findings.

1. **The substitute module depends on the orchestrator's `study_nbeats` import alias.** Accepted as an entry-point limitation, not a model defect. The reproduction note now explicitly names `forecast.py` as the entry point; standalone per-model execution is not promised.
2. **Prophet's `yhat` point differs from its sampled q0.5.** Accepted and disclosed. Mean point error and predictive median pinball loss are different quantities. Prophet did not win, so this does not affect the winning model's diagnostics.
3. **The plan said Prophet prior strength was selected, but code selects seasonality mode.** Accepted and corrected in `plan.md`; the prior is fixed. The transcript matches the implementation. No fitted model changed.
4. **Stride 168 is a self-imposed neural training restriction, not required by measured runtime.** Accepted. The discussion explicitly warns against inferring architecture superiority from this budgeted comparison. The protocol is not changed after test scoring.
5. **Tests only run from `code/`.** Disagreed with the literal claim: the recorded command `../../.venv/bin/python -m unittest discover -s code -p 'test_*.py'` runs successfully from the study root; unittest discovery supplies the import path. The transcript gives that exact invocation.
6. **A `fitted_prophet` extras filter is unused.** Accepted as harmless unused handling. It has no execution or output effect; no scope-expanding refactor was needed.

The reviewer confirmed causal shifted features, recursive median history, complete validation coverage, best-epoch fresh refits, probability-column ordering, the corrected SARIMA state retention, and the loader that prevents local `prophet.py` from shadowing the installed package.

## Deterministic verification

- Seven consumer tests pass, including current/future-load perturbation, recursive lag propagation, federal/local-calendar boundaries, median-preserving crossing correction, split counts and known metric values.
- A complete independent rerun of selection, fitting, refitting, forecasting and reporting produced byte-identical outputs for all 12 test/validation forecast CSVs. The actual subprocess took 86.24 seconds. Evidence: `artifacts/reproducibility.json` and `reproduction.log`.
- The separate `verify_study.py` recalculates every test metric, winner coverage and pinball loss from CSV forecasts without using the reporting metric functions. JSON/CSV agreement, validation argmin and selected epochs, source/code hashes, all eight image dimensions (300 dpi), quantile ordering and discussion length pass.
- All eight figures were visually inspected. Initial overlaps in the square metric/heatmap figures were corrected, then both were rendered and inspected again.

## Final study review

The independent Claude reviewer approved the final study with no material misleading claim. They independently recomputed all six metric rows, the winner coverage and pinball losses, per-day errors, interval width, and validation-selection scores directly from the CSVs, without calling the study's metric or verification functions. All values matched. They checked that recorded code hashes match the reviewed code and that both optional figures are required by the actual top two.

Final review notes and dispositions:

1. **The neural window counts could be stated more plainly.** Accepted as optional context: selection uses 246 windows, refit 259, about eight gradient steps per epoch. The transcript already discloses stride 168 and calls the budget a self-imposed handicap. No further compression-sensitive prose change was made after approval.
2. **Parts of the generated discussion and metric title are specific to this result.** Accepted as a scope limitation. This is a study of the supplied immutable CSV and recorded configurations, not a general-purpose narrative generator. Changing the dataset, package versions or model specifications requires fresh interpretation of the prose, even if code still runs. The same-input reproduction is verified.
3. **Two-candidate grids are modest, not comprehensive tuning.** Accepted; section 3 explicitly names the two choices per model. No broad optimum or architecture-ranking claim is made.
4. **Per-model runtimes include different operations, including Prophet decomposition preparation.** Accepted. The table measures each model's complete implementation workload, not comparable fitting-only latency. The reproduction note distinguishes per-model workflow time from total wall time.
5. **Hour-of-day boxes have only seven observations; some bar labels touch the baseline reference line.** Accepted as disclosed/acceptable. The residual subtitle gives the sample count; the labels remain legible. These are descriptive plots, not significance tests.
6. **The verifier's Jan 1 calculation uses the first 24 rows.** Already handled by its preceding exact timestamp-index assertion; the reviewer confirmed this is sound and not a defect.

The reviewer did not run training or the verification commands themselves; their numerical check was independent recomputation, while the orchestrator supplied the actual execution evidence. No model specification or selected configuration was changed after looking at test scores.

Additional verifier robustness check: three mutations in disposable copies were rejected at the intended assertions: a wrong reported MAPE, crossed quantiles, and widened intervals with stale coverage. The original outputs were untouched. Evidence: `mutation_checks.json`.
