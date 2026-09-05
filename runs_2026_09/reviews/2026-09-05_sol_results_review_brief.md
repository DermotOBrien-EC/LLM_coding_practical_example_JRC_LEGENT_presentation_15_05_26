# Review brief: September 2026 re-run write-up and deck slides (read-only)

You are reviewing a finished write-up against the files it is built from.
Read-only: do not modify, create or delete anything. You may run shell
commands that only read (ls, cat, head, grep, python3 for parsing JSON and
CSV). Do not fit any model. Work from the repository root.

## What to review

1. `runs_2026_09/RESULTS.md`, sections 1 to 6.
2. `runs_2026_09/DESIGN.md`: the paragraph beginning "Correction, 2026-09-05"
   in section 5.2, the "Print-mode keep-alive" bullet and the amended
   attempt policy in section 3, and the last entry of section 9.
3. The seminar deck's re-run slides, a verbatim copy at
   `runs_2026_09/reviews/2026-09-05_deck_rerun_slides.qmd.txt` (Quarto
   markdown; the slides are the `##` headings). The deck's claims must
   agree with RESULTS.md and with the files below.

## Ground truth to check against

- `runs_2026_09/results.csv` (recomputed MAPE per run), `scoring.json`
  (per-run decisions and audit notes), `summary.json` (harness facts:
  turns, wall-clock, cost, tokens, AGENTS.md read turn), `RESULTS_table.md`.
- Per run: `runs_2026_09/<run>/output/metrics.json`, `transcript.md`,
  `code/`, `derived/` where present, `session.jsonl` (large; grep it).
  The four L3 runs are `fable51_L3_r1`, `fable51_L3_r2`, `fable51_L3_r3`
  and `opus5_L3_r1` (two attempts: `session.jsonl`, `session_resume1.jsonl`).
- May 2026 reference: `runs/L3/code/lightgbm_features.py`,
  `runs/L3/metrics.json`, `runs/L3/transcript.md`; `runs/L1/metrics.csv`,
  `runs/L2/metrics.csv`.
- The frozen prompt `prompts/L3.md` and `runs/L3/AGENTS.md`.
- Independent per-run readings: `runs_2026_09/_assess/<run>.json`.

## Questions, in priority order

For each, state the expected behaviour or value, what you actually found,
and the file and line you found it in.

1. Numbers. Every MAPE, range, median, count, turn count, minute count,
   dollar figure and token figure quoted in RESULTS.md sections 1, 2, 3, 5
   and 6 and in the deck copy: does it match `results.csv`, `summary.json`,
   `scoring.json` or the run's own `metrics.json`? Report every mismatch,
   however small, with both values.
2. The May L3 correction (DESIGN.md 5.2, RESULTS.md section 4, deck). The
   claim is that `runs/L3/code/lightgbm_features.py` builds features on
   the concatenated train, validation and test series (line 111) and
   predicts test hours from that frame (line 143), so `lag_24h`,
   `rollmean_24h` and `rollstd_24h` for test hours after 1 January are the
   test week's own actual loads. Confirm or refute from the code itself,
   including `_build_features` and how the rolling columns are computed.
   Is "six of the seven days" the right count?
3. The September L3 claims. For each of the four L3 runs: (a) the winner
   was LightGBM by test MAPE and also had the lowest validation MAPE, with
   the validation numbers quoted in RESULTS.md section 4; (b) the LightGBM
   forecast is recursive (own predictions fed into the 24-hour lag), not
   built from test actuals; (c) the run produced validation-based
   selection, intervals and a methods document; (d) the coverage figures
   quoted in section 4. Cite lines.
4. The `test_selected` versus `leaked` rulings in RESULTS.md section 4,
   including the orchestrator's decision to keep `fable51_L3_r3` at
   `test_selected` despite its labelled day-ahead supplement. Is the
   reasoning consistent with DESIGN.md section 5.2 as written, and with
   how the same rule was applied to the May L3 run? Say plainly if you
   think the two rulings are inconsistent.
5. Every claim in the deck copy that is not in RESULTS.md, or that
   compresses a RESULTS.md claim: does the compression overstate? In
   particular "the beginner prompt caught up", "honest, week-ahead, and
   worse", "every L3 session did all of it", "both models caught the leak
   the May run had missed", "read AGENTS.md within its first seven
   commands", "97 to 129 minutes".
6. Anything in RESULTS.md that contradicts DESIGN.md, or one part of
   RESULTS.md that contradicts another.
7. What did you not check, and why.

## Output

A numbered list of findings, most serious first. For each: severity
(blocker, major, minor, note), the claim as written, the expected value or
behaviour, the actual value or behaviour with file and line, and a
one-sentence suggested fix. End with the list of things not checked. No
preamble.
