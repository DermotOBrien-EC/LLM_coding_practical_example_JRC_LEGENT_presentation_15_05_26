# Verification and review record

## Independent model review: not completed

A single Claude/Fable judgment reviewer was launched to challenge `artifacts/plan.md`, then asked to check the implemented validation and forecast-isolation paths. The brief explicitly required read-only work and no additional agents. Subsequent messages identified the daily training-stride amendment, SARIMA state-retention and variance-freezing corrections, and the unexplained live epoch-cap change. A final message requested a bounded report from completed reads, without further scope expansion.

No findings or completed review were returned. The task was stopped after the first completed run and output checks were available. The later quantile-protocol repair and final rerun were therefore not independently model-reviewed either. Its only returned text was: "Reading the plan and AGENTS.md first, then checking the data and any existing code they reference."

This is an incomplete review, not approval. The verification below was performed by the author through executable checks and visual inspection; it is not a replacement claim of cross-vendor review. A publication or deployment decision should obtain that missing independent review.

## Author-detected issues and dispositions

| Issue | Evidence | Disposition |
|---|---|---|
| Low-memory SARIMA results could not advance the validation origin | `attempt1.log`: `predicted_state` was None inside statsmodels `extend` | Fit with retained filtering states; successful full rerun |
| Concentrated residual scale could be recalculated while filtering validation data | `attempt2.log` and installed statsmodels filtering behavior | Use an explicit fitted sigma2 parameter, freeze the entire parameter vector, assert equality after each extension, and test the behavior |
| Weekly neural sampling would anchor examples to one weekday | Darts training-stride source documents end-anchored slicing | Use a 24-hour stride covering all weekdays before fitting |
| PatchTST unavailable in installed Darts | Installed Darts 0.41.0 catalogue inspection | Use actual TransformerModel; report the substitution, never label its results as real PatchTST |
| Neural cap changed from 30 to 2 on disk | Live-file change notification and final source/run records | Preserve the changed value; disclose that two epochs cannot trigger five-epoch patience and do not represent convergence; writer/intent not established |
| Mixed-unit grouped error bars would be misleading | MAPE has percentage units while RMSE/MAE have MW units | Plot ratios to each baseline metric and retain raw numeric %/MW labels |
| Asterisk ambiguous between the transformer substitution and holiday on heatmap | Visual inspection | Write `Transformer (sub.)` in the heatmap row label |
| LightGBM candidates and final model used different quantile sets | `quantile_protocol_red.log`; sorting changed the median at 9 of 168 forecast hours in `quantile_path_check.json` | Fit all five quantiles during candidate validation as well as the final refit; add a crossing-quantile consumer witness and rerun every model |
| Diagnostic tools created caches outside the requested directory | Parent `../../.pytest_cache/v/cache/lastfailed` names this test file; parent `../../.ruff_cache` was created during Ruff checks | Disclose the unintended writes; leave those caches untouched and explicitly route subsequent tool caches locally |

## Completed checks

- All eight consumer tests passed, including fixed SARIMA variance, no-gradient deep validation, and identical LightGBM quantile sets and median extraction in validation and final forecasts.
- Darts public prediction outputs agreed with the neural validation callback's predictions within the asserted numerical tolerance.
- `code/verify_outputs.py` recomputed all six metric rows, winner coverage and pinball losses from saved forecasts without calling the implementation's scoring functions. It checked CSV/JSON agreement, quantile ordering, all eight expected PNGs, exact dimensions, 300 dpi, input SHA-256, and the discussion word count.
- The main report is under 1,000 whitespace-delimited words, with a 328-word discussion.
- `ruff format` and `ruff check` passed. E402 exceptions are explicit only where local namespace or cache setup must precede imports.
- All eight figures were opened and inspected for labels, units, legends, layout and color consistency; the revised heatmap is explicitly labeled.
- The final six-model run completed in under nine minutes, including initial rendering; its exact timing is in `run_manifest.json`. Model forecasts were all saved before scoring in that rerun. The correction was driven by the protocol-identity failure, not by test-error optimization; the superseded run is archived under `pre_quantile_protocol_fix/`.

`artifacts/verification.json` contains the machine-readable output-check summary. The initial plan is retained unchanged as provenance; the amendments above and `artifacts/methods.md` describe what actually ran.
