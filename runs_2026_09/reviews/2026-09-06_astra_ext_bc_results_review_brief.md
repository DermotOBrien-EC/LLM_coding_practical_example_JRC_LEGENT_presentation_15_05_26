# Review brief: Extension B and C results write-up (read-only)

You are reviewing the write-up of two finished extensions against the raw
material they summarise. Read-only: do not modify, create or delete
anything; you may run read-only shell commands (cat, grep, python3 for
parsing JSON and CSV). Work from the repository root.

## What to review

1. `runs_2026_09/RESULTS.md` sections 9 and 10 (new). Sections 1 to 8 were
   reviewed earlier and must not be contradicted or silently restated.
2. The `runs_2026_09/scoring.json` entries whose `model_tag` is `astraos`,
   `solos`, `gpt55os`, `opus48`, `opus47`, and the `opus5` entries for reps
   2 and 3, against the run directories `runs_2026_09/<run>/` (each holds
   `session.jsonl`, `final_message.md`, `summary.json`, `output/`),
   `runs_2026_09/results.csv` and the independent reader's assessments in
   `runs_2026_09/_assess/<run>.json`.
3. `runs_2026_09/DESIGN.md` sections 12 and 13 (the pre-registration these
   results must be read under), its section 9 launch-record entries dated
   2026-09-06, and section 5.2 (the audit rule).
4. The seminar slide text in
   `runs_2026_09/reviews/2026-09-06_deck_bc_slides.qmd.txt`, and the
   figures it shows: `runs_2026_09/figures/exp-rerun-mape.png`,
   `exp-openai-mape.png` and `exp-extension-outcomes.png`, built by
   `scripts/build_rerun_figures.py` from `results.csv` and `scoring.json`.

## Questions, in priority order

For each, state what you expected, what you found, and the file and line.

1. Recompute every number and count in sections 9 and 10 and on the slides
   from `scoring.json`, `results.csv`, `summary.json` and the run
   directories: the per-cell ranges and spreads, the counts of runs that
   produced a forecast, the wall-clock figures, the process proportions
   (validation, intervals, methods document), and the audit-class counts.
   Name any that does not reproduce.
2. The audit classes under DESIGN.md 5.2 read literally. Is each class
   right, and is each `leak_scope` accurate to the session log or the code
   line it cites? Pay particular attention to:
   - the two Opus 4.7 L3 runs, recorded `leaked` because the winner's
     features are built on the full series (the May 2026 construction);
   - `opus47_L1_r2` and `gpt55os_L2_r1`, recorded `leaked` because features
     or hyperparameters were changed after a test-week score was read;
   - `opus5_L1_r3` and `opus5_L2_r2`, recorded `leaked` because the target
     week was summarised and cited before the features were designed, where
     Extension A's `sol_L1_r1` and `gpt55_L1_r2` were recorded
     `indeterminate` for summarising it without a traceable consequence: is
     that distinction defensible on the logs, and is it drawn consistently?
   - `astraos_L3_r3` and `solos_L2_r1`, where the reader called a leak
     (a timing probe fitted on a window containing the test week; a
     robustness run from a later origin) and the write-up declines it;
   - `opus48_L3_r1`, where a non-winning model was repaired after its test
     score and the run is not reclassified.
3. Are the section 10.1 claims about May 2026 supported? Specifically that
   the May L1 session fitted no learned model, that all three September
   Opus 4.7 L1 sessions did, and that the design does not separate harness,
   date, configuration and sampling (DESIGN.md section 4). Is anything
   claimed about causes that the evidence cannot carry?
4. Is the Extension B reading sound: that the note removed the stopping
   rather than changing the models' judgment, and that every run resolved
   the ambiguity the same way? Check the agents' own stated assumptions in
   their reports, and name any run that resolved it differently.
5. Route and harness disclosures: the machine sleep during Extension C, the
   concurrency actually used, the gateway cooldown that voided four runs
   and the one resume, the agents that built their own environments and
   downloaded packages, and the claim that the twelve Claude runs and the
   27 Extension A runs are unchanged. Verify the last one with `git diff`.
6. Anything in sections 9 or 10 or on the slides stated more strongly than
   the evidence supports, or that contradicts sections 1 to 8.
7. What did you not check, and why.

## Output

A numbered list of findings, most serious first: severity (blocker, major,
minor, note), the claim or line, expected versus actual, and a one-sentence
suggested fix. End with the list of things not checked. No preamble.
