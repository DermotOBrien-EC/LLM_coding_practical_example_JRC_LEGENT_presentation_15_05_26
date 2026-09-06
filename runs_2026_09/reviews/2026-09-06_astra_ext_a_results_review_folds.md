# Fold record: GPT-6-Astra review of the Extension A write-up, 2026-09-06

Review: `2026-09-06_astra_ext_a_results_review.md` (read-only, `codex exec`,
high reasoning effort, brief beside it; the reviewer read RESULTS.md
section 8, the 27 scoring entries, the run directories, the reader
assessments, DESIGN.md sections 9 and 11 and the slide snapshot
`2026-09-06_deck_extension_slides.qmd.txt`). Orchestrator (Fable 5.1)
verified each cited line before acting.

| # | Severity | Verdict | Action |
|---|---|---|---|
| 1 | major | Accepted. The literal rule says "only to score the finished forecast"; the Fable L1 r1 analogy does not change it. | `sol_L1_r1` and `gpt55_L1_r2` are `indeterminate`, with the final code's exclusion of the test week recorded in each note and in 8.3. |
| 2 | major | Accepted; verified in all five logs ("claude-sonnet-5 is temporarily unavailable (server error), so auto mode cannot decide"). | 8.1 and 8.4 report six blocked steps from classifier unavailability; the model-id claim is limited to assistant messages; the scoring notes say the same. |
| 3 | major | Accepted. 2.41 is outside the Fable L1 range; the improvements are 7.2 to 8.4 points; the slide dropped the 9.5 exception. | 8.2, 8.5 and the slide corrected. |
| 4 | major | Accepted. Section 4 has three leaked Claude headlines. | 8.3 names Opus 5 L1, Opus 5 L2 and Fable L1 r3, plus the Fable L3 r3 supplement. |
| 5 | major | Accepted. `n_candidates` is defined by 5.2 as candidates scored on the test window. | Set to 1, 102, 1, 1, 1, 0 and 7 for the seven forecast runs; the SARIMA first pass (100.16, fixed and re-scored after the test score was seen) is disclosed in the note and in 8.3; "agrees on every field" removed. |
| 6 | major | Accepted. | The outcomes figure has a fifth category for the agent-reported study, drawn in a different colour with the number in italics; 8.1 says six saved a forecast file and one reported its accuracy. |
| 7 | minor | Accepted; verified (Astra L2 r1 and r2: five turns, four tools; Sol's L3 reads after three or four Bash calls). | Corrected; positions are named as assistant-message indices. |
| 8 | minor | Accepted; verified by grep (11 Claude runs, 11 extension runs). | Corrected, availability separated from use. |
| 9 | minor | Accepted (six forecasts in wave A leave eleven stopped; GPT-5.5 L3 r3 made no tool call). | Section 9 entries corrected. |
| 10 | minor | Accepted; git shows 22:19:09 and 22:27:35. | Section 8 gives the commit times; a dated correction sits under the section 11 heading; cells and runs distinguished in 8.2, 8.5 and on the slide; the May reference in the gpt55_L3_r2 note is 3.43. |
| 11 | minor | Accepted. | Provenance "unresolved"; the follow-up "might" let the runs finish (it is now Extension B); the read claim narrowed to pre-existing files. |
| 12, 13, 14 | note | Confirmations of the numbers, the leak scopes, the subagent account and the preservation of sections 1 to 7. | Recorded. |
