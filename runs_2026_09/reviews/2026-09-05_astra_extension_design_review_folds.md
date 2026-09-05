# Fold record: GPT-6-Astra review of Extension A (design and harness), 2026-09-05

Review: `2026-09-05_astra_extension_design_review.md` (read-only,
`codex exec`, high reasoning effort, brief in
`2026-09-05_astra_extension_design_review_brief.md`; the reviewer read
commit `8144202` against `999e602` because the diff was already committed).
Orchestrator (Fable 5.1) evaluated each finding against the files before
acting, before the canary.

| # | Severity | Verdict | Action |
|---|---|---|---|
| 1 | blocker | Accepted. `resume_run.sh` cleared the environment and passed no model, so a gateway L3 resume would have called Anthropic with an OpenAI id and used up the one permitted resume. | The resume script reads `route` and `model` from `run_meta.json`, rebuilds the gateway environment and passes `--model` for gateway runs, and does so before the attempt counter is written. |
| 2 | major | Accepted. The key is necessarily in the agent's environment and in the supervisor's argv; "never reaches the sandbox" was too strong, and `~/cliproxyapi` was readable. | Section 11 wording narrowed to "not written to run_meta.json or the logs"; the seatbelt now denies `~/cliproxyapi` (section 3 amendment, dated). |
| 3 | major | Accepted. `CLAUDE_CODE_SUBAGENT_MODEL` is a default that an agent definition's own model overrides. | Section 11 says "default, not a guarantee" and commits to auditing subagent models from the session logs. |
| 4 | major | Accepted. The gate scanned every session log regardless of route, and the classifier is a Claude dependency. | `launch_wave.sh` skips the gate for waves made only of gateway tags; section 11 (e) and (f) state both dependencies. |
| 5 | major | Accepted. "Void" was undefined for this arm. | Section 11 (f) now states the wave 2 precedent as the rule: void only when the session ends on an API error before its first tool call (archived, relaunched once under the same rep); otherwise counted as it stands. |
| 6 | major | Accepted. | `launch_extension.sh` stops after a wave A with no completed run and exits non-zero if any run failed. |
| 7 | minor | Accepted. | Section 11 states the `bypassPermissions` versus `auto` departure, describes the concurrency cap as concurrent execution, and qualifies the effort value. |
| 8 | minor | Accepted. | The figure footnote now says "the September Claude L3 forecasts"; the extension L3 runs get their own audit before any such claim. |
| 9 | note | Accepted as procedure. | Section 11 gains a post-processing scope: derived files regenerate deterministically and must diff empty for the Claude runs; the reader runs on named extension runs only. |
| 10, 11 | note | Confirmations (27 runs, interleaving, unchanged Claude invocation, bash 3.2 expansion, identifier scan; the diff was already committed). | Commit range recorded in DESIGN.md section 10. |
