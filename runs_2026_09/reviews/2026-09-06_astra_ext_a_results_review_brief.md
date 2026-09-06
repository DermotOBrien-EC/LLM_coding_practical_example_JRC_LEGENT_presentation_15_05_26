# Review brief: Extension A results write-up (read-only)

You are reviewing the write-up of a finished extension against the raw
material it summarises. Read-only: do not modify, create or delete
anything; you may run read-only shell commands (cat, grep, python3 for
parsing JSON and CSV). Work from the repository root.

## What to review

1. `runs_2026_09/RESULTS.md` section 8 (new; sections 1 to 7 are the
   Claude arm and were reviewed earlier).
2. The 27 entries of `runs_2026_09/scoring.json` whose `model_tag` is
   `astra`, `sol` or `gpt55`, against the run directories
   `runs_2026_09/<tag>_L<n>_r<k>/` (each holds `session.jsonl`,
   `final_message.md`, `summary.json`, `output/`), `runs_2026_09/results.csv`
   and the independent reader's assessments in `runs_2026_09/_assess/<run>.json`.
3. `runs_2026_09/DESIGN.md` section 9, the three launch-record entries dated
   2026-09-05 22:29 onwards, and section 11 (the pre-registration these
   results must be read under).
4. The seminar slide text in
   `runs_2026_09/reviews/2026-09-06_deck_extension_slides.qmd.txt` (a copy
   of the slides that quote these results; the figure it shows is
   `runs_2026_09/figures/exp-extension-outcomes.png`, built by
   `scripts/build_rerun_figures.py` from `results.csv` and `scoring.json`).

## Questions, in priority order

For each, state what you expected, what you found, and the file and line.

1. Every number and count in section 8 and on the slide: recompute it from
   `scoring.json`, `results.csv`, `summary.json` and the run directories
   (20 of 27 stopped, 9 of 9 Astra asked, the seven MAPEs, the wall-clock
   minutes, the classifier denials, the notional costs, the AGENTS.md read
   turns). Name any that does not reproduce.
2. The audit classes of 8.3 under DESIGN.md 5.2 read literally: is each of
   the seven classes the right one, and is each `leak_scope` accurate to
   the session log (line numbers cited in `scoring.json`)? In particular
   `sol_L1_r2` (leaked at the selection stage), `gpt55_L1_r3` (an
   intermediate file that copied the test actuals) and `gpt55_L3_r2`
   (leaked headline). Where the write-up and the reader disagree
   (`sol_L1_r1`, `gpt55_L1_r2`: indeterminate versus final_scoring_only),
   is the write-up's reasoning stated and defensible?
3. The claims about the route: the Fable subagent inside `sol_L1_r2` (24
   turns, 8 Bash, 2 Write, nothing in the run directory), the blocked
   probe in `sol_L1_r3`, "no other model id in any other session", and the
   statement that the log records no source for the agents' knowledge of
   the operator's tooling. Verify against the session logs.
4. Is anything in section 8 or on the slide stated more strongly than
   section 11 (h) allows (descriptive only; model plus translation layer),
   or more strongly than the evidence supports? Is the slide's caption
   faithful to 8.5?
5. Does section 8 change, contradict or silently restate anything in
   sections 1 to 7 or in DESIGN.md sections 1 to 11? It should not.
6. What did you not check, and why.

## Output

A numbered list of findings, most serious first: severity (blocker, major,
minor, note), the claim or line, expected versus actual, and a one-sentence
suggested fix. End with the list of things not checked. No preamble.
