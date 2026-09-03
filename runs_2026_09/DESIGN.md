# Re-run of the three-level prompt experiment, September 2026

Pre-registered design. Revision 3, after two rounds of independent
read-only review by GPT-5.6-Sol (`reviews/`); the fold records are in
section 10. Written before any counted run was launched; the numbers in
`RESULTS.md` were filled in afterwards.

## 1. Questions

The May 2026 experiment ran one Claude Code (Opus 4.7) session per prompt
level and observed test MAPE 10.76 % (L1) → 5.52 % (L2) → 3.43 % (L3),
together with large differences in what each session produced around the
number (validation discipline, intervals, figures, write-up).

- **Q1 (level effect, Fable 5.1).** With three fresh sessions per level, does
  the specificity of the prompt and workspace still change what comes back,
  and where: in the accuracy number, in the process, or in both?
- **Q2 (repeatability, Fable 5.1).** Given the same prompt three times, does
  the model reach the same model class, a similar accuracy, and the same set
  of artefacts?
- **Q3 (a second model, descriptive).** What does Claude Opus 5 produce on the
  same three prompts, one run each?

Only the Fable 5.1 cells are replicated. The Opus 5 runs and the May 2026
Opus 4.7 runs are single sessions and are reported as descriptive reference
points; nothing in this design supports attributing a difference between
them and the Fable runs to the model generation (section 7).

## 2. Factors and cells

| Factor | Values |
|---|---|
| Level | L1, L2, L3. Prompts are `prompts/L1.md`, `prompts/L2.md`, `prompts/L3.md`, byte-identical to May 2026 and pinned by SHA-256 in `pins.sha256`. |
| Workspace | L1: data file only. L2: data file + the 7-line `AGENTS.md` copied from `runs/L2/AGENTS.md`. L3: data file + the 113-line `AGENTS.md` copied from `runs/L3/AGENTS.md`. Both `AGENTS.md` files are pinned. |
| Model | `claude-fable-5-1`, three runs per level. `claude-opus-5`, one run per level. Each model runs at its CLI default effort; the default is part of the tested configuration. |

Twelve counted runs. The May 2026 runs are not re-run.

## 3. Held fixed

- **Inputs.** `pins.sha256` pins the three prompts, the data file (50,400
  hourly rows, 2015-01-01 to 2020-09-30) and the two `AGENTS.md` files. The
  runner verifies the pins with `shasum -c` and refuses to start on any
  mismatch. Each run records the digests it used.
- **Build.** Claude Code 2.1.259, called by absolute path; the runner refuses
  to start if `--version` differs.
- **Sandbox.** Each session works in a neutral directory tree outside the
  repository, `~/dev/energy_forecast_ws/<token>/project/runs/<token>/`,
  built fresh per run: an APFS clone of the repository venv at
  `project/.venv` with its `pyvenv.cfg` prompt and entry-point shebangs
  rewritten to the clone, a generic `pyproject.toml` (same dependencies)
  and a matching `uv.lock`, the data file and (L2, L3) the frozen
  `AGENTS.md` in the working directory, and for L3 `project/prompts/L3.md`,
  which its `AGENTS.md` cites. The relative paths inside the frozen L3
  prompt and `AGENTS.md` (`../../.venv/bin/python`, `../../prompts/L3.md`)
  resolve exactly as they did in May. Before launch the runner scans the
  sandbox (outside the venv's library tree, and excluding the L3 files that
  legitimately name the study) for study identifiers and refuses to start
  if any is found.
- **Confinement.** The agent process runs under a macOS seatbelt profile
  (`seatbelt.sb`, kept with the run) that denies reading and writing
  everything under `~/dev` except this run's sandbox, everything under
  `~/.claude/projects` except this run's own entry, `~/.claude/history.jsonl`,
  every other run's staging area, and process listing. The prompt file and
  the supervisor are staged under neutral names, the logs are collected
  through anonymous pipes, and the harness keeps `run_meta.json`,
  `session.jsonl`, `stderr.log` and the inventories in the repository, so
  nothing in the agent's argv, open files or working directory names the
  run, the level or the model tag. This is a deny-list on one user account,
  not a virtual machine: the account's home remains the agent's home
  (subscription authentication needs it), so `~/.claude.json` and other
  home files are readable in principle. The summariser reports every
  command or tool path that reaches above the sandbox root as a warning
  signal, and such runs are disclosed.
- **Python environment.** The same `uv.lock`, Python 3.12.13, darts 0.41.0,
  lightgbm 4.6.0, torch 2.10.0 (CPU), in the cloned venv.
- **Prompt delivery.** The prompt file is piped to `claude -p` on stdin, so
  the bytes are unchanged, as the first and only user message. No operator
  interjection.
- **Permissions.** Auto mode (the classifier-gated mode the operator uses
  interactively) with prompts routed to nobody: any action that would have
  needed a human answer is denied instead. Denials are read from the
  session's terminal result event, so a run that ends without one (timeout,
  stop) has no denial count; that is stated where it happens.
- **Isolation of configuration.** The agent runs under an allow-listed
  environment (`env -i` with exactly HOME, USER, LOGNAME, SHELL, LANG, PATH)
  and with `--setting-sources project,local`, so the operator's user-level
  `CLAUDE.md`, rules, hooks, skills and effort setting are not injected.
  Verified with a probe session that reported no user-level instructions.
  The claude.ai MCP connectors attached to the account (Context7, Hugging
  Face, Spotify) remain available, as they were in May.
- **Process control.** Each session runs in its own process group under a
  supervisor with a 180-minute wall-clock cap, the same for every level. On
  timeout the group (the agent and every descendant that stayed in the
  group) is terminated and the run is marked `timeout`; the supervisor
  records whether anything in the group survived. A descendant that leaves
  the group (its own new session) is outside this control and would show up
  as a surviving process or as files changed after the run.
- **Harvest.** A full inventory of the sandbox (path, node type, size,
  symlink target, SHA-256 outside the venv, size and mtime inside it) is
  taken before the session starts and again after it ends; the diff is the
  record of what the agent added, modified and deleted. Added and modified
  files inside the working directory are copied to `output/`, elsewhere in
  the sandbox to `output_outside/`; deletions and any change inside the
  venv are listed in `harvest.json`. A modified data file or `AGENTS.md` is
  kept as evidence. If the harvest itself fails the run is marked
  `harness_failed_before_finalise`.
- **Attempt policy** (inherited unchanged from `RUNBOOK.md`, May 2026). L1
  and L2: first attempt only, never resumed. L3: may be resumed once, with
  `scripts/resume_run.sh`, which refuses unless the level is L3, the first
  attempt did not complete, it has not been resumed before, and its log
  shows a timeout, a stop, or a rate-limit or API error; the resumed session
  receives the single message "Continue.", is logged to
  `session_resume1.jsonl`, and is disclosed beside the result. The
  asymmetry is deliberate and is the same one the May runs used.
- **Wave membership, order and gate** (pre-registered in `waves.json`;
  `scripts/launch_wave.sh` accepts only a wave name from that file).
  `canary`: `fable51 L1 r1` alone, to validate the harness after the
  aborted launches in section 9. `wave1`, launched together: `fable51 L1
  r2-r3`, `opus5 L1 r1`, `fable51 L2 r1-r3`, `opus5 L2 r1` (light runs;
  in May, L1 and L2 fitted models that take seconds). `wave2`, two at a
  time in this order: `fable51 L3 r1` + `opus5 L3 r1`, then `fable51 L3 r2`
  + `fable51 L3 r3`. The launcher reads the latest rate-limit event in any
  session log on disk and refuses to start unless the five-hour window's
  utilisation is at or below 0.35 or its reset time has passed. Machine
  contention still exists and is disclosed as a limitation.

## 4. Known differences from May 2026

- Headless `-p` instead of the interactive terminal; the first message is
  the same text.
- The May runs inherited whatever user-level configuration the operator had
  at the time; that state was not recorded. The re-run isolates it.
- The May runs sat inside the repository (`runs/L1`, `runs/L2`, `runs/L3`),
  where the README, the other prompts and the other levels' `AGENTS.md`
  were reachable two levels up; whether any May session read them is not
  recorded. The re-run's sandbox removes that possibility.
- Claude Code has moved from the May build to 2.1.259, and each model's
  default effort is whatever the September build applies.
- Runs execute in parallel on one machine, so wall-clock and per-model
  `runtime_seconds` are not comparable with May.

## 5. Measurements per run and the decision table

### 5.1 Outcome

- **Headline forecast.** The forecast the agent itself put forward as its
  answer. When a run fitted one model, that model. When a run fitted several
  and chose among them, the headline is the class it selected **by
  validation** (a held-out window ending before 2020-01-01) if the log
  shows such a selection; that class's test score is the headline number,
  scored once. When the run chose by comparing test scores (the May L1 and
  the frozen L3 bake-off both do this), the class it put forward is still
  reported as its headline, the run is classified `test_selected` with the
  number of candidates (section 5.2), and where the validation-preferred
  class can be read from the run's own files its test score is reported
  beside the headline as the selection-free number. If the agent presents
  several forecasts without choosing one, no headline is assigned; every
  scorable forecast is reported and the figure shows them all as unselected.
- **Test MAPE**, recomputed by `scripts/score_runs.py` from the headline
  forecast file against the actuals in `data/opsd_de_load.csv`, on the 168
  hours the forecast is stamped with (timestamps converted to UTC). A
  forecast stamped in local time covers a week one hour off the UTC test
  window; the scorer reports the shift and the run is scored on the hours
  it was actually made for. The agent-reported number is recorded beside
  it; a difference above 0.05 percentage points is flagged and explained.
  The choice of file and column per run, with its reason, is recorded in
  `scoring.json` before any cross-run comparison is made.
- **Missing outcome.** If a run wrote no machine-readable forecast, the
  outcome is "not recoverable from disk"; the agent-reported number, if
  any, is shown in italics and with a hollow marker. Nothing is re-prompted
  or re-run to obtain a number.

### 5.2 Test-window audit (before any cross-run comparison)

Each run is classified from its code and session log, before its MAPE is
compared with any other run's, as one of:

- `final_scoring_only`: the test observations were used only to score the
  finished forecast;
- `test_selected`: two or more candidates were scored on the test window
  and the best was put forward (selection optimism; the number of
  candidates is recorded);
- `leaked`: test-window observations entered fitting, feature construction
  or hyperparameter selection;
- `indeterminate`: the log does not allow a call.

By this rule the May 2026 L1 run is `test_selected` (three baselines, best
reported), L2 is `final_scoring_only` (one model), and L3 is
`test_selected` with six candidates: the frozen prompt selects
hyperparameters within each class by validation but names the class with
the lowest **test** MAPE as the winner. In May the class with the best
validation MAPE (LightGBM, 2.02 % on 2019 Q4) was also the test winner, so
its selection-free number equals its headline, 3.43 %. The audit reader
sees the run's own final message, which usually states a MAPE, so this
audit is done before comparison, not blind.

### 5.3 Process, with file-level definitions

| Item | Counts as yes only when |
|---|---|
| Model class fitted | executed code produced a test-window forecast for it (a forecast file, a metric in a file, or the printed metric in the log); code that exists but never ran does not count |
| Held-out validation | executed code scored candidates on a window that ends before 2020-01-01 and the choice depended on it (quote the line) |
| Prediction intervals | a file or figure contains lower and upper bounds for the test window |
| Written document | the agent wrote a methods, results or README file (`.md`, `.txt`, `.pdf`, `.html`); the final chat message does not count |
| Read `AGENTS.md` | `yes`: a Read tool call on the file, or a read-capable shell utility naming it as an operand; `possible`: a glob such as `cat *.md`; `indeterminate`: the file system was reached through a variable, a script or an interpreter whose arguments cannot be resolved; `no`: none of these. The classifier is lexical and is a warning signal, not proof; the independent reader confirms from the log. The turn of the first confirmed read and the turn of the first implementation action (a Write/Edit tool call, or a shell command that runs Python, writes a file or redirects into one) are recorded so a late read is not mistaken for influence. Claude Code auto-loads `CLAUDE.md`, not `AGENTS.md`; a probe on 2.1.259 confirmed an `AGENTS.md` in the working directory is not injected into context. |
| Artefacts | the harvest diff and an exact manifest (path, bytes, SHA-256) of what the agent left behind, with separate counts for PNG, other image, HTML, document, Python, CSV and JSON files |
| Provenance | seeds or determinism settings, chosen hyperparameters, models attempted but not fitted, tool-call errors, permission denials (where a result event exists), rate-limit or error events, commands reaching outside the sandbox, turns, tokens, list-price cost, wall-clock |

The per-run classification is done by an independent reader (GPT-5.6-Sol,
read-only, strict JSON schema in `scripts/assess_schema.json`) and every
load-bearing field (headline, MAPE, validation, test-window audit,
`AGENTS.md` read) is re-checked by the orchestrator against the files and
the session log before it enters `RESULTS.md`.

## 6. Analysis and pre-stated reading rules

- **Per level** (Fable 5.1): each run's MAPE and the min-max range; the
  model class per run; each process item as a proportion of runs (0/3 to
  3/3); the test-window audit class per run.
- **Repeatability label.** A cell is called *repeatable in model class* only
  when all three runs put forward the same model-class taxonomy entry
  (naive / statistical / additive-seasonal / tree ensemble / neural /
  other), and *repeatable in accuracy* only when all three are recoverable
  and max minus min is at most 1 percentage point. Otherwise the range is
  reported without a label.
- **Level comparison, accuracy.** Only when every Fable run in both cells is
  recoverable: report whether the absolute difference between the L1 and
  L3 medians is at most 1 percentage point, as *descriptive closeness in
  these six runs*. This is not evidence of equivalence and is not read as
  the absence of an accuracy effect. If either cell is incomplete the rule
  is not applied. Where a cell mixes `final_scoring_only` and
  `test_selected` runs, the comparison says so.
- **Level comparison, process.** Differences in the per-item proportions are
  reported as *observed in these runs*, not as a general effect.
- **Opus 5 and May 2026** are shown beside the Fable cells as annotated
  single observations. No statistical test is reported anywhere.

## 7. Limitations

- n = 3 per Fable cell, n = 1 per Opus 5 cell, n = 1 per May 2026 cell.
- The May runs differ from the re-run in build, interaction mode, unknown
  user configuration, default effort and directory exposure; an Opus 5
  number differs from a Fable number in one session's worth of everything.
  Neither comparison isolates the model.
- Scoring L1 and L2 depends on the agent having written its forecast to a
  file.
- Confinement is a seatbelt deny-list, not a VM; the lexical detectors in
  the summariser are warning signals; the test-window audit is not blind.
- Headless print mode ends a session whenever the agent finishes a turn
  without a tool call, including when it has deliberately parked itself to
  wait for a background job or monitor; an interactive session would have
  been woken. A run that ends this way is scored on what is on disk and
  marked incomplete. This bit three of the twelve runs, and it is the
  reason the L3 cell has no result: the L3 prompt's six-model bake-off
  includes two deep forecasters whose training outlasts the agent's
  patience for waiting in a way that a headless turn cannot express.
- The account's usage credits are a hard external dependency. The
  launcher's gate reads the five-hour window; the weekly window is not
  gated and is what stopped wave 2.
- Other command-line agents installed on the machine (`codex`) are
  reachable from the sandbox, exactly as they were in May; a run that uses
  one is disclosed.
- Parallel execution on one machine can change runtime and, through the
  harness's per-command time limit, could in principle change what an agent
  does; the wave plan limits heavy concurrency to two and any tool timeout
  in a log is disclosed.

## 8. Files

- `pins.sha256`: pinned inputs; `waves.json`: waves, order, concurrency and
  the usage gate; `scripts/run_headless.sh`: one run (sandbox, seatbelt,
  inventories, harvest); `scripts/supervise.py`: process-group wall-clock
  cap with piped logs; `scripts/inventory.py`: inventories, diff, harvest;
  `scripts/launch_wave.sh`: a pre-registered wave; `scripts/resume_run.sh`:
  the single permitted L3 resume; `scripts/summarize_runs.py`: harness facts
  per run; `scripts/score_runs.py`: MAPE recomputation;
  `scripts/assess_fleet.sh` + `assess_schema.json` + `assess_brief_template.md`:
  independent per-run reader; `scripts/build_rerun_figures.py`: figure and
  table; `scoring.json`: the per-run headline decisions with reasons;
  `RESULTS.md`: the write-up.

## 9. Launch record

- **2026-09-03 14:00 local, wave 1 (in-repository layout), aborted after
  four minutes.** The first launch used run directories inside the
  repository with the harness files beside the data. Within four tool calls
  the sessions had listed the parent directories and read the repository
  README, this design document and their own `run_meta.json` (which names
  the model and level); two read their own live session log; one was
  heading for the May run outputs. All eight sessions were terminated
  before any had written a forecast. The logs are kept locally under
  `runs_2026_09/_aborted_wave1_contaminated/` (not committed) and are
  excluded from analysis.
- **2026-09-03 14:11 local, canary (first sandbox layout), completed in 8
  minutes, then voided.** The sandbox removed the repository files but the
  cloned venv's `pyvenv.cfg` still carried the original project name and
  the copied `pyproject.toml` still carried the description "Three-level
  prompt-engineering demo", which the session read. The run is kept locally
  under `runs_2026_09/_aborted_canary_metadata_leak/` (not committed) and is
  excluded from analysis. It is the reason for the metadata rewrite, the
  identifier scan and the seatbelt in section 3.
- **2026-09-03 17:04 local, canary (final harness), counted.** `fable51 L1
  r1`, completed in 6 minutes; recorded in `RESULTS.md`.
- **2026-09-03 17:11 local, wave 1, counted.** Seven runs launched together
  with the usage gate at 0.09. Two observations from the logs, both kept
  and disclosed rather than re-run: (a) `fable51 L2 r2` started a
  feature-selection backtest in the background, armed a monitor and said
  it would wait; in headless print mode a turn that ends without a tool
  call ends the session, so the run finished without the agent choosing
  its final feature set or reporting. Its files on disk are scored as they
  stand and the run is marked incomplete. Interactive sessions do wake on
  such events; this is a headless-mode difference from May. (b) `fable51
  L2 r1` found the `codex` CLI on the machine's PATH and, on its own
  initiative, ran GPT-5.6-Sol read-only reviews of its script from the
  home directory. The seatbelt applies to child processes, so those
  reviews could not read the repository, but `~/.codex` session logs
  (which hold this design's own review transcripts) were not on the deny
  list at the time; nothing in the log shows them being read. From wave 2
  onwards `~/.codex/sessions`, `~/.codex/log` and `~/.codex/history.jsonl`
  are denied as well. The `codex` CLI was also installed on this machine
  in May.
- **2026-09-03 21:50 local, wave 2, no result.** The gate opened on the
  five-hour reset and pair A launched. `opus5 L3 r1` ran 37 minutes,
  fitted four of the six required models, then ended on a turn where it
  said it would wait for the two deep models to finish training; no
  test-window forecast was written. The three `fable51 L3` runs were
  refused by the API with "You're out of usage credits" (weekly overage
  window at 1.01, status `rejected`), 95 seconds into the first and at
  launch for the other two. The L3 cell of this study is therefore empty.
  All four directories are kept with their logs; none contributes a
  number. See `RESULTS.md` sections 6 and 7.

## 10. Review record

Round 1 (GPT-5.6-Sol, read-only, before launch): 13 findings, 3 blockers.
Accepted and folded: uniform wall-clock cap and pre-registered waves with a
rate-limit eligibility threshold (F1, in part); descriptive-only wording for
Opus 5 and May (F2); the decision table and file-level definitions (F3);
the rewritten reading rules (F4, F5); process-group termination (F6);
pinned inputs, pinned build, stdin prompt delivery and the reconciled
environment list (F7); launcher validation and exit status (F8); parse
failure counts, missing-result flags and per-attempt aggregation (F9);
stricter `AGENTS.md` detection with an ambiguity class and first-read
position (F10); the artefact manifest and separated counts (F11); the
test-window audit (F12); provenance fields (F13). Retained after
consideration: the level-dependent resume rule, because it is the May
protocol and is disclosed rather than hidden (F1, in part).

Round 2 (GPT-5.6-Sol, read-only, after the aborted launches): 10 findings,
3 blockers. Accepted and folded: venv metadata rewrite, generic
`pyproject.toml`, identifier scan (F1); seatbelt confinement, neutral
staging, piped logs (F2, in part: a VM or a second account is out of scope
and the residual is stated in section 3); inventory-diff harvest with a
harness-failure status (F3, F4); detectors reclassified as warning signals
with an `indeterminate` class and narrowed wording (F5, F6, in part);
TERM forwarding, survivor check and a narrowed process-control claim (F7);
L3 reclassified as `test_selected` with the selection-free number reported
beside the headline (F8); the wave manifest, the mechanical usage gate, the
resume script, and honest wording on denials and blinding (F9, in part);
the corrected launch record (F10).
