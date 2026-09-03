# continue_work.md

Updated: 2026-09-03 (session that ran the September 2026 re-run)

Read this with `runs_2026_09/DESIGN.md` (the pre-registered design, its two
review rounds, and the launch record) and `runs_2026_09/RESULTS.md`
(findings, plus sections 6 and 7 on what is unfinished). Those two files are
the substance; this file carries the decisions, the permissions and the
state that are not in them.

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
   `bash scripts/resume_run.sh opus5_L3_r1 "<reason>"` (its sandbox and
   session id are preserved, and the design permits exactly one L3 resume
   after an infrastructure failure); or drive L3 interactively as the May
   runs did; or add a harness keep-alive. Do not silently change the
   prompt or the model to make it fit.

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
