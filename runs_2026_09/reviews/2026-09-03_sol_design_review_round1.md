FINDINGS
  F1. [BLOCKER] [Q1] Level is confounded with compute exposure, recovery policy, wave membership, and shared resource pressure.
      Where: DESIGN.md:60-65, 74-75, 126; scripts/launch_wave.sh:13-23
      Why it matters: L3 receives three times the wall-clock budget and may resume, while arbitrary parallel waves share CPU and a rate-limit window. A level or model grouped with other CPU-heavy runs can time out, hit tool deadlines, or be rate-limited differently; contention therefore can change outputs, not merely runtime.
      Fix: Give every cell the same cap and resume rule; pre-register exact balanced/randomised wave membership and launch order; limit heavy L3 concurrency and CPU threads; define a pre-wave rate-limit eligibility threshold.

  F2. [MAJOR] [Q1] Opus 5 and May comparisons are single-session historical case comparisons, not estimates of model effects.
      Where: DESIGN.md sections 1, 2, 4, and 7
      Why it matters: Opus 5 has one observation per level, while May additionally differs in CLI version, interaction mode, unknown user configuration, possible operator intervention, and default effort. A better or worse number cannot be attributed to model generation.
      Fix: State explicitly that only the new Fable runs provide replicated within-model level comparisons; Opus 5 is descriptive, and May is an annotated historical reference with no causal or repeatability inference. Treat default effort as part of each model’s tested configuration.

  F3. [MAJOR] [Q1] Headline-forecast selection and process classifications are not operationally pre-registered.
      Where: DESIGN.md:81-103, 122-125
      Why it matters: Two operators could choose different columns when several forecasts exist, or disagree whether code merely containing validation/interval logic counts as “used/produced.” Choosing after seeing MAPE creates outcome-dependent selection.
      Fix: Define a decision table before launch: the headline is the forecast explicitly selected in the final message or metrics file; if ambiguous, score and report all without selecting one. Define “fitted,” “validation used,” “interval produced,” and “written document” using executable/file-level evidence, plus missing-outcome handling.

  F4. [MAJOR] [Q2] The one-percentage-point median rule is not evidence that accuracy is “no longer” bought by specificity.
      Where: DESIGN.md:111-112
      Why it matters: Six stochastic observations can happen to have close medians despite material uncertainty, and missing/unrecoverable forecasts can make the median selectively observed.
      Fix: Replace it with: “If all three forecasts in both cells are recoverable, report whether |median L1 MAPE − median L3 MAPE| ≤ 1 percentage point as descriptive closeness in these six runs; do not interpret this as equivalence or absence of an accuracy effect.” Do not apply the rule when either cell is incomplete.

  F5. [MAJOR] [Q2] “Specificity is doing work” and “same accuracy” are not supported by the stated checklist rule.
      Where: DESIGN.md:113-117; DESIGN.md section 1, Q2
      Why it matters: One checklist difference may be stochastic, while continuous MAPEs will almost always differ numerically; no threshold defines “same.”
      Fix: Report checklist completion as per-item proportions (0/3–3/3) and call differences “observed in these runs,” not a general effect. Predefine repeatability as exact agreement in a model-class taxonomy and, if desired, a MAPE range threshold such as max−min ≤1 percentage point; otherwise report the range without a repeatable/not-repeatable label.

  F6. [BLOCKER] [Q3] The wall-clock alarm kills only the Claude parent process, not its spawned computation tree.
      Where: scripts/run_headless.sh:68-76
      Why it matters: At the cap, an N-BEATS or TSMixer subprocess can survive as an orphan and continue modifying the run directory after `run_meta.json` records completion, while also consuming CPU needed by other cells.
      Fix: Use a macOS-compatible supervisor that creates a new process group/session, sends TERM and then KILL to the entire group on timeout, waits for every child, and records a distinct timeout status.

  F7. [MAJOR] [Q3] The launcher does not enforce the pre-registered executable or input bytes.
      Where: scripts/run_headless.sh:15-16, 39-46, 51-65, 69-75; DESIGN.md:28-30, 37-58
      Why it matters: An inherited `CLAUDE_BIN` can select another build; prompt command substitution removes all trailing newlines; prompt/data hashes are recorded but never checked against fixed expected values; and the copied `AGENTS.md` is not hashed. The child also receives LOGNAME and SHELL despite the design saying only HOME, USER, PATH, and LANG.
      Fix: Fail closed on exact version and pre-registered data/prompt/AGENTS hashes; record the executable path and AGENTS hash; preserve prompt bytes using stdin or a sentinel technique; reconcile the documented and actual environment allowlists. Quoting otherwise preserves internal whitespace and newlines.

  F8. [MAJOR] [Q3] `launch_wave.sh` reports success to callers even when runs fail, and its specification parsing is insufficiently guarded.
      Where: scripts/launch_wave.sh:13-25; scripts/run_headless.sh:33-38
      Why it matters: The final `echo` returns status 0 regardless of `fail`, so automation can accept an incomplete wave. For exact controlled triples, `set -- $spec` splits correctly under bash, but it also performs glob expansion, ignores extra fields, and can exit after launching earlier jobs when fields are missing. Duplicate specs can race through the non-atomic existence check.
      Fix: Parse and validate exactly three fields without glob expansion, reject duplicate run names, create run directories atomically, trap interruptions and terminate/wait for children, and exit nonzero whenever `fail > 0`. The individual run exit code itself is currently preserved correctly.

  F9. [MAJOR] [Q4] Missing, truncated, or resumed result events can silently produce inconsistent token and tool totals.
      Where: scripts/summarize_runs.py:78-87, 113-162
      Why it matters: Malformed JSON lines are silently discarded and absence of a result becomes plausible-looking zeros. If resumed attempts are appended, tool calls span all attempts but result usage/cost is overwritten by the final result, mixing scopes.
      Fix: Count and report parse failures; require and flag terminal result events; distinguish “unavailable” from zero; store per-attempt summaries and explicitly aggregate usage/cost/tool calls. Also record every rate-limit/error event rather than only blocked counts and maximum utilisation.

  F10. [MAJOR] [Q4] `AGENTS.md` read detection has both false positives and false negatives.
      Where: scripts/summarize_runs.py:123-135
      Why it matters: `ls AGENTS.md`, `echo AGENTS.md`, or reading `NOT_AGENTS.md` marks a read, while `cat *.md` can genuinely read the file without containing the literal name. This can reverse the conclusion about whether workspace context influenced a run.
      Fix: Resolve Read-tool paths exactly to the run’s root `AGENTS.md`; classify Bash commands by read-capable utility and expanded argument paths. Where shell indirection cannot be resolved reliably, report “possible/ambiguous” rather than yes/no. Record the turn of the first confirmed read.

  F11. [MINOR] [Q4] Figure and artefact counts do not represent the labels or the “same set of artefacts” question.
      Where: scripts/summarize_runs.py:30-36, 90-97, 168-176, 190-200
      Why it matters: The Markdown column says `png` but counts PNG, PDF, and SVG; JPEG/HTML figures are missed; a report PDF is counted as a figure; and extension counts cannot distinguish two different file sets.
      Fix: Emit an exact relative-path manifest with size and digest, exclude harness files only at their known root paths, and report separately defined counts for PNGs, other image formats, plot-like HTML, and documents. Rename the current column to `figures` if retaining its present definition.

  F12. [BLOCKER] [Q5] The design does not audit test leakage or test-based model selection.
      Where: DESIGN.md:90-103
      Why it matters: An L1/L2 run may fit several candidates, inspect test errors, and put forward the best test performer, while L3 selects on validation. Recomputed MAPE would still be numerically correct but not comparable because one result contains test-set selection optimism.
      Fix: Add a blinded code/log audit per run recording whether test observations entered fitting, feature construction, hyperparameter or model-class selection, and the number of candidates compared on test. Classify the test as final-scoring-only, leaked, or indeterminate before interpreting accuracy.

  F13. [MAJOR] [Q5] Q2 needs provenance measurements for stochasticity, execution, artefact identity, and workspace exposure.
      Where: DESIGN.md section 5
      Why it matters: Equal MAPEs or file counts can mask different seeds, hyperparameters, failed/unexecuted code, or different artefacts; a late `AGENTS.md` read cannot explain earlier decisions.
      Fix: From code, files, and the session log, record random seeds/determinism settings, chosen hyperparameters, internal retries, attempted versus successfully fitted models, command failures, exact artefact manifests and validity, and first-`AGENTS.md`-read position relative to the first implementation/model-fitting action.

NO-FINDING AREAS
  None — questions 1 through 5 each have at least one finding.

SUMMARY
  Do not launch yet: resource/attempt confounding, incomplete process-tree termination, and absent test-leakage auditing are blockers.
  Keep Opus 5 and May results explicitly descriptive, and replace the equivalence/causal reading rules with bounded observational wording.
  Per-run quoting and exit capture are otherwise sound; aggregate launch status and post-run accounting still require correction.