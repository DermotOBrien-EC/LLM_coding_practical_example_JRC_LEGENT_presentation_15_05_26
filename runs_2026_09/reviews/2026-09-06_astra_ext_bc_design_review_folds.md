# Fold record: GPT-6-Astra review of Extensions B and C (design and harness), 2026-09-06

Review: `2026-09-06_astra_ext_bc_design_review.md` (read-only, `codex exec`,
high reasoning effort, brief beside it; the reviewer read `git diff 0e7f0d1
b1f1fb4` and DESIGN.md sections 12 and 13). Orchestrator (Fable 5.1)
evaluated each finding against the files before acting, before launch.

| # | Severity | Verdict | Action |
|---|---|---|---|
| 1 | major | Accepted. The driver took two waves and would have silently skipped C's third. | Rewritten to take an ordered list of waves, validated against the manifest before anything starts; the exit status covers every wave. |
| 2 | major | Accepted. Three reps do not bound stochastic variation. | Section 13 now says the reps show the observed September spread and that harness, date, configuration and sampling are not separated. |
| 3 | major | Accepted. B's classifier and possible Claude subagents run on the gateway's Claude credentials; running both L3 waves at once would exceed section 7's two-L3 limit. | Section 13 states the dependencies and that the quota coupling is not established; the launch sequence keeps the two L3 waves apart (C's L3 wave starts only after B's L3 wave and C's wave 2 have ended) and discloses up to ten light sessions at once. |
| 4 | minor | Accepted. An empty completion count passed the `= 0` test; old logs could count. | The count fails closed on any non-numeric result and counts only logs written after the driver started. |
| 5 | minor | Accepted. Every gate-check failure mapped to exit 5. | `launch_wave.sh` exits 5 only for a closed gate and 6 for a failed check; section 13 describes the retry as bounded and pre-wave. |
| 6 | minor | Accepted. The reader's brief would have described the frozen prompt alone for B runs. | `assess_fleet.sh` reads `run_meta.json` and names the suffix in the brief for B runs. |
| 7, 8, 9 | note | Confirmations (suffix delivery, pins, model ids, scan, membership, name uniqueness, gate observation, launch-time preservation); the figure-builder configuration for the new tags is pending post-processing. | Recorded; the figure builder is updated when the runs are scored. |
| 10 | note | Accepted. | Section 12 says the mixed GPT-5.5 L3 cell cannot become a three-run B cell or meet the recoverability requirement, and that B is fresh attempts on cells selected by their A outcome. |
