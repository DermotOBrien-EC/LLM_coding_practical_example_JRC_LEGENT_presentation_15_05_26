# Fold record: GPT-6-Astra review of the Extension B and C write-up, 2026-09-06

Review: `2026-09-06_astra_ext_bc_results_review.md` (read-only, `codex exec`,
high reasoning effort, brief beside it; the reviewer read RESULTS.md sections
9 and 10, the 44 new scoring entries, the run directories, the reader
assessments, DESIGN.md sections 9, 12 and 13, and the slide snapshot
`2026-09-06_deck_bc_slides.qmd.txt`). 11 majors, 5 minors, 2 notes, no
blocker. Every finding was checked against the files before acting.

| # | Severity | Verdict | Action |
|---|---|---|---|
| 1 | major | Accepted; verified (session lines 200, 204, 214, 219). `gpt55os_L2_r3` swept eight lag sets against the target week and printed each one's target MAPE before adopting the winner, and it did run historical validation. | Reclassified `leaked` with a feature-selection scope, eight candidates, validation true; Extension B's validation count becomes 18 of 20. |
| 2 | major | Accepted. The orchestrator had exempted work outside the headline's causal chain; 5.2 as written has no such restriction and the supplement precedent of section 4 already records a leak confined to a supplement. | `astraos_L3_r3` (SARIMAX timing fits), `solos_L2_r1` (later-origin robustness runs) and `opus48_L3_r1` (a losing model repaired after its test score) are now `leaked`, each with a scope naming what was and was not affected. Extension B is 3 / 10 / 6, Extension C 5 / 11 / 8. |
| 3 | major | Accepted; verified. The Claude arm is not bare: of 24 L1 and L2 runs, 8 validated, 4 produced intervals, 4 wrote a methods document. | The blanket claim is gone from section 9 and from the discipline slide; both now give the proportions and say no Claude run was given the note. |
| 4 | major | Accepted. Section 2 declines the causal readings that section 10 attributed to it, and the third Opus 4.7 L3 run differs from its siblings in more than the recursion. | Section 10.1 and 10.2 rewritten: the cell removes the model as an explanation rather than proving a cause, the run pair is an association with its confounds named (800 versus 600 trees), and the invented attribution to section 2 is removed. |
| 5 | major | Accepted; verified (session line 8210, `lightgbm_features.py:198`, transcript line 38). | `solos_L3_r3` is no longer described as complete in substance: its delivered code and transcript disagree because a regeneration was killed by the cap, and the reported numbers are the earlier pass. |
| 6 | major | Already fixed before the review returned, on the same evidence: the third Opus 4.7 L3 run finished after the slide snapshot was taken. | The slides and section 10 carry all three runs and state each row's construction; the recursive range reads from 3.9. |
| 7 | major | Accepted. | Section 9 and the slide say none stopped again and 19 of 20 produced a scored headline, with the twentieth ending while fitting. |
| 8 | major | Accepted; recounted (12 + 27 + 20 + 24). | 83 counted sessions, not 90, in the results header and on the setup slide; voids and resumes are counted separately in the launch record. |
| 9 | major | Accepted; verified against both scripts. May's climatology is an estimated grouped mean and two of the three September runs are of the same family. | Section 10.1 names the four methods instead of asserting a move from no learning to learning. |
| 10 | major | Accepted; verified (`work/fetch_temp.py`, `out/README.md`). Opus 5 L1 r3 downloaded ERA5 temperature for fourteen cities and used the target week's own weather as a known predictor, its own README calling it a perfect-weather assumption. | Recorded in the scoring entry and in section 10.4 as a material external input; its 5.35 is stated as not comparable with the other L1 numbers. |
| 11 | major | Accepted. | The dagger now marks one mechanism only, the winner's own inputs containing test-week observations, and says so; the other leak scopes are named per run in `RESULTS_table.md`; the footnote states that numbers without a forecast file are the agent's own. |
| 12 | minor | Accepted; recomputed from unrounded values. | 2.68 is no longer called the study minimum (Fable L2 r1 is 2.298); the widest cell claim is scoped to the extension; spreads corrected to 4.37 and 0.21. |
| 13 | minor | Accepted in part. | Astra L2 r1 reads 4.79; the figures round from the recomputed values. The stale reason strings in `results.csv` are regenerated from `scoring.json` by the scorer and are left as the record of what was checked when. |
| 14 | minor | Accepted; verified. | `opus48_L3_r3` records four models fitted and validation true; `opus47_L3_r1` records eight candidates and a leak scope that includes raising N-BEATS epochs after reading a test score. |
| 15 | minor | Accepted; verified against git and `run_meta.json`. | The pre-registration is 00:43:05, nine minutes before the first run; the review folds were in the working tree at launch and committed at 01:07; `opus47 L3 r1` was resumed once for 29 seconds, which section 9 of the design record and the scoring entry now say. |
| 16 | minor | Accepted; verified (`fable51_L2_r3` attempted installations in the original arm). | Section 9 lists the runs that built their own environments across arms and drops the "no earlier run" claim. |
| 17, 18 | note | Confirmations of the principal leakage findings and of 26 recomputed headlines, the three-versus-sixteen L3 split, the wall-clock ranges, the void archives and the unchanged Claude and Extension A material. | Retained with the narrower wording the reviewer asks for. |
