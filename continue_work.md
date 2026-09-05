# continue_work.md

Updated: 2026-09-05 16:45 local (session that completed wave 2); the
sections below section 0 are the 2026-09-03 handoff and are superseded
where section 0 says so.

Read this with `runs_2026_09/DESIGN.md` (the pre-registered design, its two
review rounds, and the launch record) and `runs_2026_09/RESULTS.md`
(findings, plus sections 6 and 7 on what is unfinished). Those two files are
the substance; this file carries the decisions, the permissions and the
state that are not in them.

---

## 0. State on 2026-09-05 evening (read first)

**Done.** The owner asked (09:30 local) to "continue, finish off this work
and have my presentation ready, then commit and push to my dermot branch".
All four L3 runs completed the same day, the L3 cell is filled, the
write-up (`runs_2026_09/RESULTS.md`) and the design record are final, the
deck's slides 43 to 49 are updated and rendered, both repositories are
committed and pushed (this one to `master`, the deck to `dev_dermot`; no
merge, no PR). The write-up and slides were reviewed read-only by
GPT-5.6-Sol and every finding was folded or answered
(`runs_2026_09/reviews/2026-09-05_sol_results_review_folds.md`).

Headline: Fable 5.1 L3 5.03, 4.99, 5.53 (LightGBM won every six-model
bake-off; the only Fable cell repeatable in accuracy); Opus 5 L3 5.46
(resumed once). Two corrections came out of it: May's L3 winner (3.43) had
the test week's observed history in its rolling features from the second
hour and in its 24-hour lag from the second day, so it is `leaked` under
the design's own rule and flagged everywhere; and the harness summariser
now sums a session's result events (Opus 5 L2's turn and token counts
changed). Fable L3 r3 is recorded `leaked` for a labelled supplement with
its headline untouched (`leak_scope` in `scoring.json`).

**Extension A, in flight from 2026-09-05 22:30 local.** The owner asked
(22:10) for "the full experiment" with GPT-5.5, GPT-5.6-Sol and
GPT-6-Astra "using the same harness using poly with codex models", and
said the workshop is on the 11th. Pre-registered as DESIGN.md section 11
(commit 8144202): the same harness, routed to the local gateway
(`sol-proxy.sh`, 127.0.0.1:8317) with the `poly` launcher's environment,
tags `astra`, `sol`, `gpt55`, three runs per level each, 27 runs. Order:
Astra read-only review of section 11 and the harness diff
(`runs_2026_09/reviews/2026-09-05_astra_extension_design_review*.md`),
fold, then `bash scripts/launch_wave.sh ext_canary` (astra L1 r1, inspect
by hand: tool calls, classifier, result event, harvest), then
`nohup bash scripts/launch_extension.sh > runs_2026_09/_logs/extension.log`
(ext_waveA, 17 light runs six at a time; then ext_waveB, nine L3 runs two
at a time, roughly nine hours; marker `EXTENSION_DONE`). Afterwards:
`assess_fleet.sh` per run (the reader fleet is Astra now), `score_runs.py
discover` then `scoring.json` entries then `final`, `build_rerun_figures.py`
(the figure has six series; consider a second figure for the OpenAI arm),
a RESULTS.md section 8 for the extension, the deck (a new slide or two;
the verifier needs extending for the new tags), review, push. Codex quota
is the only cost; a gateway 429 shows up as an API error in the session
log (void for L1 and L2, one resume for L3).

Also optional: delete the sandboxes under `~/dev/energy_forecast_ws/`
after the extension is scored, and tell Andres the slides are on
`dev_dermot`. The seminar deck still carries the May date on its cover;
the workshop is on 2026-09-11.

The rest of this section is the in-flight record as written at 09:50 and
is kept for provenance.

The weekly credit window had come back (probe: seven-day 0.13, five-hour
0.34, overage disabled for lack of credits, so an exhausted window refuses
rather than charges). Wave 2 was relaunched at 09:45 local:

- `nohup bash scripts/relaunch_wave2.sh > runs_2026_09/_logs/wave2_relaunch.log`
  runs detached on the owner's Mac. Pair A: the single permitted resume of
  `opus5_L3_r1` (log `_logs/opus5_L3_r1.resume.log`, session log in
  `runs_2026_09/opus5_L3_r1/_resume/session.jsonl` until the script moves it
  to `session_resume1.jsonl`) beside a fresh `fable51_L3_r1`
  (`_logs/wave2_pairA_fable.*`). Pair B (`fable51_L3_r2` + `r3`,
  `_logs/wave2_pairB.*`) starts when both have ended and the usage gate
  passes; the gate reads the live five-hour utilisation, so pair B may wait
  for the 14:00 local reset. The driver ends with the line
  `WAVE2_RELAUNCH_DONE resume=<rc> pairA=<rc> pairB=<rc>`.
- Harness changes made for the relaunch, all disclosed in `DESIGN.md`
  sections 3, 7 and 9: `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0` in
  `run_headless.sh` and `resume_run.sh` (print mode no longer ends a session
  600 s after a turn that leaves background jobs running); the resume gate
  accepts that cut-off as an infrastructure failure; `waves.json` gained
  `wave2_pairA_fable` and `wave2_pairB`; the void 09-03 `fable51_L3_r*`
  directories moved to `runs_2026_09/_void_wave2_credit_refusal/`.

When the driver has finished, the remaining steps are, in order:

1. `scripts/summarize_runs.py`, then `scripts/score_runs.py` (discover, fill
   `scoring.json` for the four L3 runs, then `final`), then
   `bash scripts/assess_fleet.sh opus5_L3_r1 fable51_L3_r1 fable51_L3_r2
   fable51_L3_r3` (Sol, read-only) and re-check every load-bearing field
   against the logs, then `scripts/build_rerun_figures.py`.
2. Rewrite `RESULTS.md` sections 1, 2, 3, 6 and 7 for the L3 cell and the
   headline; fill `DESIGN.md` section 9's outcome sentence.
3. Deck (`~/dev/ai_seminar_jrc`, branch `dev_dermot`): copy the new
   `exp-rerun-mape.png` into `figures/`, update slides 45 to 49 (accuracy
   caption, discipline table rows for L3, the two "no September L3" notes,
   the three cards), render with `~/.local/bin/quarto render index.qmd`,
   run `.venv/bin/python scripts/verify_deck_numbers.py
   ~/dev/ai_seminar_jrc/index.qmd` from this repository, then commit and
   push `dev_dermot` (no merge, no PR; Andres merges).
4. Commit and push this repository's `master`; delete the
   `~/dev/energy_forecast_ws/*` sandboxes only after the L3 results are in
   git.

If a session is refused mid-run (`"status":"rejected"` in its session log)
the L3 resume rule allows one resume per run; `opus5_L3_r1` has used its
one. Do not launch anything else without the owner.

---

## 1. What the owner asked for, and what is done

The owner's request, in their words: re-run the workshop experiment "with
the current state of things", on the premise that "fable5.1 is smart enough
that it probably wont make a difference and what was covered in the workshop
i did before is only really valid for less inteligent models"; also test
Opus 5; also run Fable 5.1 three times "to see if it gives same output 3
times or if the prompt styles actually help for inteligent models or not";
then update the repo, update the slides, and push to the `dermot` branch of
`AndresLaverdeMarin/AI_seminar_jrc`, where Andres merges.

Done and pushed:

- **This repository (`master`, pushed):** the whole re-run harness, the
  pre-registered design, eight completed L1/L2 runs with full session logs
  and harvested outputs, independent per-run assessments, recomputed
  scoring, the deck figure and table, and the write-up.
- **`AndresLaverdeMarin/AI_seminar_jrc`, branch `dev_dermot` (pushed,
  commits `b4029b3` and `9460b6b`):** five new slides in the Practical Use
  Cases section (now slides 45 to 49 of 84), Tip 5 corrected to name
  `CLAUDE.md` for Claude Code with `AGENTS.md` for Codex and most others,
  and `PRESENTERS.md` renumbered. The owner's section now runs 38 to 51 and
  hands back on slide 51.

Not done: **the L3 cell has no September result.** See section 3.

## 2. The headline finding, so it is not lost

Recomputed test MAPE, each from the forecast file the run itself wrote:

| Level | May 2026 (Opus 4.7) | Fable 5.1, three runs | Opus 5, one run |
|---|---|---|---|
| L1, 10 words | 10.76 | 3.96, 3.28, 4.49 | 3.07 |
| L2, 46 words + 7-line AGENTS.md | 5.52 | 2.30, 5.35 (incomplete), 3.21 | 3.40 claimed, 3.67 on disk (incomplete) |
| L3 | 3.43 | not run | no forecast produced |

The owner's hypothesis is half right, and the half that fails is the
interesting half. **Accuracy:** yes, ten words now get a frontier model to
roughly where the 1,673-word prompt got Opus 4.7 in May, with holiday
features and quantile bands appearing unprompted. **Discipline: no.** No L1
or L2 run used a held-out window to choose between model classes, only one
of six Fable runs wrote a methods document, and three of eight runs let the
test week into a decision, including both Opus 5 runs. That is exactly what
the L3 prompt's DO NOT block forbids, and it is the argument the workshop
should now make: specificity has stopped buying accuracy on this task and
has started buying method validity.

Repeatability: same model class every time (LightGBM or a sibling tree
ensemble in all six Fable runs), but the number moves 1.2 points across L1
and 3.1 across L2. Under the design's own rule neither cell is "repeatable
in accuracy".

## 3. What is blocked, and on what

**Wave 2 (the four L3 runs) produced nothing.** Three Fable L3 sessions were
refused by the API with "You're out of usage credits" (the weekly overage
window read 1.01, status `rejected`). The Opus 5 L3 session ran 37 minutes,
fitted four of the six required models, and then ended on a turn where it
said it would wait for N-BEATS and PatchTST to finish training. Headless
print mode ends a session on any turn with no tool call, so the training
processes died with it and the orchestrator never ran.

Two blockers, in order:

1. **Credits.** The weekly window resets **2026-09-09 11:00 local**. The
   launcher's gate (`waves.json`) only reads the five-hour window, so it
   will happily launch into a dead weekly window; check
   `seven_day_overage_included` in a recent `rate_limit_event` by hand
   first, or extend the gate.
2. **The headless parking problem.** Three of twelve runs died this way.
   For L3 specifically the six-model bake-off with two deep forecasters is
   long enough that the agent will always park. Options, cheapest first:
   resume the existing Opus 5 session in place with
   `bash scripts/resume_run.sh opus5_L3_r1 "<reason>"` (the design permits
   exactly one L3 resume after an infrastructure failure); or drive L3
   interactively as the May runs did; or add a harness keep-alive. Do not
   silently change the prompt or the model to make it fit.

   **Its sandbox is deliberately still on disk:**
   `~/dev/energy_forecast_ws/dac166/` (1.3 GB, an APFS clone), with the
   four fitted models' pickles and the agent's `code/` tree intact. Every
   other run's sandbox was deleted after harvesting. Do not delete
   `dac166` until the L3 question is settled, and note that
   `resume_run.sh` currently checks for a timeout, a stop, or a rate-limit
   error in the first attempt's log; this run exited 0, so the check will
   refuse it and needs a one-line widening (or run the resume by hand from
   the session id in `run_meta.json`).

## 3b. Where you must be sitting to finish this

Everything needed to **re-run** L3 from scratch is in this repository. Three
things needed to **resume** the half-finished Opus 5 L3 session are not in
git and live only on the owner's Mac (`doob`):

- the preserved sandbox `~/dev/energy_forecast_ws/dac166/`;
- that session's Claude Code transcript,
  `~/.claude/projects/-Users-doob-dev-energy-forecast-ws-dac166-project-runs-dac166/19436657-a2eb-49bd-b881-03118ed3dd75.jsonl`;
- the project `.venv` (gitignored; `uv sync` rebuilds it elsewhere).

So: a new session **on that Mac** can resume and save the 37 minutes
already spent. A fresh clone on any other machine can only re-run L3 whole,
and additionally needs macOS (the harness uses `sandbox-exec` and APFS
clones), Claude Code exactly 2.1.259 or an edit to `EXPECT_VERSION` in
`scripts/run_headless.sh`, and `uv sync`.

## 4. Permissions and boundaries

- **Push:** the owner asked for the push to `dev_dermot`; both pushes above
  were made under that. Andres owns `AndresLaverdeMarin/AI_seminar_jrc` and
  merges; **do not merge, do not open a PR, do not touch `main`** without
  asking. This repository's `master` is the owner's own.
- **Spend:** the re-run has already cost roughly 45 dollars at list price
  across twelve sessions and exhausted the weekly allowance. Do not launch
  another wave without the owner saying so.
- **The frozen inputs are frozen.** `prompts/L*.md`, `data/opsd_de_load.csv`
  and `runs/L*/AGENTS.md` are pinned by SHA-256 in
  `runs_2026_09/pins.sha256`, and `run_headless.sh` refuses to start if any
  digest moves. Editing a prompt to "improve" a run destroys the comparison
  with May.
- **The May 2026 artefacts under `runs/L1`, `runs/L2`, `runs/L3` are
  historical evidence.** Never regenerate or tidy them.
- Two earlier launch attempts were voided for leaking the study into the
  agent's sandbox; their logs are under
  `runs_2026_09/_aborted_wave1_contaminated/` and
  `_aborted_canary_metadata_leak/`, gitignored, kept as provenance and
  excluded from analysis. Do not fold their content into any result.

## 5. Open items, priority-ordered

1. **Get one L3 result** (blocked on credits until 09-09). Resume
   `opus5_L3_r1` first; it is 37 minutes from done. Then decide with the
   owner whether the three Fable L3 runs are worth the spend, given that
   the L1/L2 finding already carries the talk.
2. **Fill the L3 cells** in `runs_2026_09/RESULTS.md`, the figure, the
   table, and the deck. `scripts/verify_deck_numbers.py` fails on any
   `L3PENDING` placeholder and checks every other quoted number against
   `results.csv`; run it after any deck edit:
   `uv run python scripts/verify_deck_numbers.py ~/dev/ai_seminar_jrc/index.qmd`.
3. **Tell Andres the slides are on `dev_dermot`** and that the L3 column is
   deliberately May-only for now.
4. Optional: the deck's older backup slides (61 to 84) still describe the
   May experiment only. They are consistent, just not updated.

## 6. Where things live

- Design, harness contract, limitations, launch record, review record:
  `runs_2026_09/DESIGN.md`.
- Findings and the unfinished-work analysis: `runs_2026_09/RESULTS.md`.
- Per-run: `runs_2026_09/<model>_<level>_r<rep>/` holds `session.jsonl`
  (the full stream log), `run_meta.json`, `output/` (what the agent wrote),
  `summary.json`, `changes.json` and the seatbelt profile actually used.
- Judgements about each run, with reasons: `runs_2026_09/scoring.json`.
  Independent reader's verdicts: `runs_2026_09/_assess/<run>.json`.
- Cross-model review ledger: `reviews.jsonl` at the repo root.
- The seminar deck: `~/dev/ai_seminar_jrc` (Quarto; `quarto render index.qmd`,
  and Quarto is installed via `uv tool` at `~/.local/bin/quarto`).

## 7. Harness facts worth knowing before touching it

- Claude Code does **not** auto-load `AGENTS.md`; it loads `CLAUDE.md`.
  Probe-verified on 2.1.259. The experiment keeps `AGENTS.md` because that
  is what May used and the agents do read it, but the deck now says this.
- `claude -p` takes the prompt on stdin, so the frozen bytes reach the agent
  unchanged; `--setting-sources project,local` keeps the operator's global
  `CLAUDE.md` and rules out of the session (probe-verified).
- Each run gets a fresh sandbox under `~/dev/energy_forecast_ws/<token>/`
  with an APFS clone of the venv, its metadata scrubbed, behind a macOS
  seatbelt profile. `run_headless.sh` refuses to start if a scan finds any
  study identifier in the sandbox. Both of those exist because earlier
  attempts leaked.
- The sandbox is a deny-list on one account, not a VM. The agent's home is
  the operator's home because subscription auth needs it.
- Two runs reached outside the sandbox in ways worth knowing: one found the
  `codex` CLI on PATH and ran its own peer reviews; one could not import the
  supplied venv and built its own under `~/.venvs`. Both are disclosed in
  `RESULTS.md` section 4.
