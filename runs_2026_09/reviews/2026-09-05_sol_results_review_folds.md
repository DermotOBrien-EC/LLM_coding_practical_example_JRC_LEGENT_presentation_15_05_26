# Fold record: GPT-5.6-Sol review of the write-up and deck, 2026-09-05

Review: `2026-09-05_sol_results_review.md` (read-only, `codex exec`, high
reasoning effort, brief in `2026-09-05_sol_results_review_brief.md`, deck
snapshot in `2026-09-05_deck_rerun_slides.qmd.txt`). Reviewed surface:
`RESULTS.md`, the 2026-09-05 additions to `DESIGN.md`, and the seminar
deck's slides 43 to 49. The orchestrator (Fable 5.1) evaluated each finding
against the files before acting.

| # | Severity | Verdict | Action |
|---|---|---|---|
| 1 | blocker | Accepted. Re-read `runs/L3/code/lightgbm_features.py`: the rolling features are computed on the series shifted by one hour (line 46), so test actuals enter from the second test hour (167 of 168 hours); `lag_24h` from the second day (144 hours). "Day-ahead for six of seven days" understated it. | Reworded in RESULTS.md (section 2 footnote, section 4), DESIGN.md 5.2, the figure footnote, the deck's May L3 note and accuracy caption. The r3 supplement is now described as "the May construction", not "day-ahead". |
| 2 | blocker | Accepted. DESIGN.md 5.2 has no headline-only limitation, and the May ruling was made on the literal text; keeping r3 at `test_selected` on a headline-only reading was a post-hoc narrowing. | `fable51_L3_r3` reclassified `leaked` in `scoring.json` with a new `leak_scope` field ("supplement only", headline untouched); RESULTS.md section 4 carries both readings and says which the tables use; the deck's column is renamed "target week entered the headline" and its verifier counts headline leaks only; `RESULTS_table.md` labels the case. |
| 3 | major | Accepted. Every L3 run is `test_selected` because the frozen prompt names the winner by test MAPE, which its own DO NOT block forbids; "the whole DO NOT block" and "no peeking" overstated. | RESULTS.md section 3 rewritten to say what was delivered and that the test week decided the class ranking the prompt asks for; deck takeaway and card reworded ("no test-week inputs"). |
| 4 | major | Accepted. Eleven of twelve headlines were recomputed, two of them from pickles read into derived CSVs; Opus 5 L2 is agent-reported. | RESULTS.md preamble rewritten; deck slide 45 bullet now says "re-scored from what the agent left on disk (one exception, marked)". |
| 5 | major | Partly accepted. The causal wording ("forces", "because") is dropped: the runs differ in strategy and no ablation isolates it. "Caught up" stays as the slide title: it describes L1's 3.3 to 4.5 against the May research-grade number, which is what the slide compares, and the caption and card state the strategy difference. | RESULTS.md L1-versus-L3 bullet and the deck's accuracy card reworded; median difference corrected to 1.06. |
| 6 | major | Accepted for the September text: intervals were produced and their coverage measured, none calibrated. The May slide's "calibrated intervals" describes what the prompt asks for and is Dermot's original wording, left as is; its coverage figures now carry the note from finding 1. | "calibration" replaced by "intervals with measured coverage" in the deck card and RESULTS.md section 3 and 4. |
| 7 | major | Accepted as a reporting matter. Fable L2 r3 read `AGENTS.md` in the same fourth command that ran its first Python; the design's definition has no intra-command ordering. | RESULTS.md section 3 cell now reads "2/3 strictly before; in r3 the same 4th command also ran its first Python"; the deck's "within its first seven commands" is unchanged. |
| 8 | minor | Accepted. The figures were input plus cache-read tokens with cache creation excluded; Opus L1 output is 60,490. | Relabelled in RESULTS.md section 5; 61 k corrected to 60 k. |
| 9 | minor | Accepted. | Deck card now says "71 to 129 minutes". |
| 10 | minor | Accepted. | Deck slide 45 bullet reworded to the seatbelt-confined sandbox with its staged files and "no repository context". |
| 11 | minor | Accepted. 5.025 minus 3.963 is 1.062. | 1.07 corrected to 1.06. |
| 12, 13 | note | Confirmations of the per-run L3 claims, the recursion evidence, the cost total (102.16 dollars) and the AGENTS.md turns. | RESULTS.md section 5 now states 102 dollars rather than "about 100". |

Not re-reviewed after folding: the reworded passages are new claim
material and were checked by the orchestrator against the same files
(`results.csv`, `scoring.json`, `summary.json`, the May and r3 code lines
cited), and the deck was re-verified with `scripts/verify_deck_numbers.py`.
