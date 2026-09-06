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
  run, the level or the model tag. From Extension A (section 11) onwards
  the profile also denies the local gateway's directory `~/cliproxyapi`;
  the twelve Claude runs ran without that line, and none of their logs
  touches it. This is a deny-list on one user account,
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
  environment (`env -i` with exactly HOME, USER, LOGNAME, SHELL, LANG, PATH;
  from the wave 2 relaunch also the print-mode keep-alive variable below)
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
  Amended 2026-09-05, after wave 2 (section 9): a first attempt that print
  mode itself ended, recognisable by "Background tasks still running after
  600s; terminating" in `stderr.log`, counts as an infrastructure failure
  for this rule even though the harness recorded it as `completed` (the
  CLI exits 0 after killing the agent's background jobs). The other
  conditions are unchanged, and the one-resume limit stands.
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
  Added 2026-09-05 for the wave 2 relaunch (section 9): `wave2_pairA_fable`
  (`fable51 L3 r1` alone, run beside the permitted resume of `opus5 L3
  r1`) and `wave2_pairB` (`fable51 L3 r2` + `r3`), the same pairs in the
  same order as `wave2`, through the same gate.
- **Print-mode keep-alive** (added 2026-09-05, from the wave 2 relaunch
  onwards). Claude Code's print mode ends a session 600 s after a turn
  that leaves the agent's background jobs running, and kills those jobs;
  the CLI's own message names the switch
  (`CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0` waits indefinitely). The
  runner and the resume script now pass that variable, so a session waits
  for the agent's own background jobs and re-enters the model when one
  finishes, as an interactive session would. The 180-minute wall-clock cap
  still bounds every run. This is a harness change between wave 1 and the
  relaunched wave 2; the L1 and L2 cells were not re-run under it.

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

Correction, 2026-09-05 (after the L3 cell was filled; the paragraph above
is left as pre-registered). Reading the two September L3 implementations
beside May's showed that the May L3 winner is `leaked` under this rule,
not merely `test_selected`: `runs/L3/code/lightgbm_features.py` builds
its features on the training, validation and test series concatenated
(line 111, `pd.concat([trainval, test])`) and predicts each test hour
from that frame (line 143). Its rolling 24-hour and 168-hour features are
computed on the series shifted by one hour (line 46), so from the second
test hour onward they contain the test week's own actual loads (167 of
168 hours), and `lag_24h` (line 43) contains them from the second day
onward (144 of 168 hours): the forecast had the observed history of the
previous hour behind almost every test hour. The code's own comment
("only timestamps used for features") is wrong for the lag and rolling
columns. All four September L3 sessions saw the same prescribed feature
list and forecast the week recursively, feeding their own predictions
into those features; one of them (Fable r3) also scored its model the
May way, as a labelled supplement, and got 3.71 % against its 5.53 %
recursive number, which under this rule as written makes that run
`leaked` for the supplement while its headline stays untouched
(`scoring.json`, `leak_scope`). May's 3.43 % therefore stands in every
table as the May
reference, flagged, and is not read as the same task as the September
L3 numbers. The frozen prompt's own wording admits both readings ("all
required lags must be present in the data"), so this is a finding about
the prompt as much as about the May run.

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
- Headless print mode, as run in wave 1, ends a session 600 s after the
  agent finishes a turn without a tool call while its background jobs are
  still running, including when it has deliberately parked itself to wait
  for a job or a monitor; an interactive session would have been woken. A
  run that ends this way is scored on what is on disk and marked
  incomplete. This bit three of the twelve first attempts (two in wave 1,
  kept as they stand; the `opus5 L3 r1` first attempt, resumed). The
  relaunched wave 2 runs with the keep-alive of section 3, so the L3 cell
  and the L1/L2 cells were not produced under the same headless rule.
- The account's usage credits are a hard external dependency. The
  launcher's gate reads the five-hour window; the weekly window is not
  gated and is what stopped the first wave 2 launch.
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
  the single permitted L3 resume; `scripts/relaunch_wave2.sh`: the
  2026-09-05 relaunch driver (resume beside `wave2_pairA_fable`, then
  `wave2_pairB`); `scripts/launch_extension.sh`: the Extension A driver
  (`ext_waveA` then `ext_waveB`, section 11); `scripts/summarize_runs.py`: harness facts
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
  launch for the other two. The L3 cell was empty until the relaunch
  below. The three void `fable51 L3` directories are archived under
  `runs_2026_09/_void_wave2_credit_refusal/` (committed) and contribute no
  number; the `opus5 L3 r1` directory keeps its first attempt beside the
  resume. See `RESULTS.md` section 6.
- **2026-09-05 09:45 local, wave 2 relaunched.** The weekly window had
  come back (a probe session read the seven-day window at 0.13 and the
  five-hour window at 0.34 when the pair launched, with overage disabled
  for lack of credits, so a refusal, not a charge, is the failure mode).
  The first attempt's `stderr.log` had named the cause of the Opus cut-off
  ("Background tasks still running after 600s; terminating"), so the
  keep-alive of section 3 was added to the runner and the resume script,
  the attempt policy was amended to recognise that cut-off as an
  infrastructure failure, and the two relaunch waves were added to
  `waves.json`. Pair A: `opus5 L3 r1` resumed once with "Continue."
  (`resume_run.sh`, its preserved sandbox and session id intact) beside a
  fresh `fable51 L3 r1`. Pair B: `fable51 L3 r2` + `r3` through the gate.
  Outcomes are recorded in `RESULTS.md` section 6.
- **2026-09-05 22:27 local, Extension A canary, counted.** `astra L1 r1`
  alone through the gateway (section 11), after the Astra review of the
  extension was folded (commit `e68abb2`). The route worked: the session
  started on `gpt-6-astra` in auto mode with 28 tools and no connectors,
  three tool calls executed, no permission was denied, a result event
  arrived and the harvest ran. The model then did something no Claude
  session did on this prompt: after inspecting the data it asked whether
  "first week" meant UTC or German local time and ended its turn, 35
  seconds in, with no forecast written. Nobody answers in this harness, so
  the run counts as it stands: incomplete, no recoverable outcome, with the
  clarifying question recorded. The driver for `ext_waveA` and `ext_waveB`
  was started at 22:30.
- **2026-09-05 22:29 to 23:00 local, `ext_waveA`, 17 runs, all counted.**
  Six at a time through the gateway; every run ended with a result event
  and exit 0, no void, no API error, no rate limit. Eleven of the seventeen
  ended on a clarifying question or a plan with no forecast written (all
  five remaining Astra runs, Sol L2 r1 to r3, GPT-5.5 L2 r1 to r3); the six
  L1 runs of Sol and GPT-5.5 wrote a forecast. `RESULTS.md` section 8.
- **2026-09-05 23:00 to 2026-09-06 00:19 local, `ext_waveB`, 9 L3 runs,
  all counted.** Two at a time, interleaved by model. Eight ended on a
  question or a plan (Astra r1 and r2 after no tool call, 23 and 24
  seconds; Astra r3, Sol r1 to r3 and GPT-5.5 r1 after verifying the
  data; GPT-5.5 r3 after no tool call); GPT-5.5 r2 completed the study in
  61 minutes. No resume was
  needed. `RESULTS.md` section 8.
- **2026-09-06 00:30 to 02:00 local, post-processing.** Summariser, scorer
  (27 `scoring.json` entries), independent reader on the 27 named
  extension runs (Astra, `assess_fleet.sh` with names, never `all`),
  figure builder (the MAPE figure now carries the extension points and a
  second figure, `exp-extension-outcomes.png`, shows how each of the 39
  sessions ended). `git diff` of the twelve Claude runs' directories and of
  their rows in `results.csv` and `RESULTS_table.md` is empty, as section
  11 requires. Two facts found in the audit that section 11 had flagged as
  risks: `sol L1 r2` spawned a Claude Fable 5.1 subagent through the Agent
  tool's explicit model alias, and `sol L1 r3` tried the operator's gateway
  probe script, which the seatbelt blocked; both are in `RESULTS.md` 8.4
  and in `scoring.json`. Section 11 is not edited.
- **2026-09-06 00:50 local, Extensions B and C launched** after the Astra
  design review was folded (commit `635dc00`): `ext_b_waveA` (12 runs,
  six at a time) and `ext_c_wave1` (gate: fresh window, last observation
  0.51 with its reset in the past) at 00:50:35, with `ext_c_wave2` to
  follow in the same driver and `ext_c_wave3` held by a waiter until both
  `ext_b_waveB` and `ext_c_wave2` have ended (no more than two L3 sessions
  at once). At about 00:52 to 00:58 the network outage described in the
  section 12 amendment killed five of the six `astraos` sessions of the
  first batch (`_void_ext_b_outage_20260906/`, relaunched once as
  `ext_b_relaunch` after wave A); the four Opus 5 sessions retried through
  it and continued.

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

Extension A review (GPT-6-Astra, read-only, 2026-09-05 23:00 local, on
commit `8144202` against `999e602`;
`reviews/2026-09-05_astra_extension_design_review.md`, fold record beside
it): 11 findings, 1 blocker. Accepted and folded: the resume script
rebuilds a gateway run's route and model before touching the attempt
counter (F1); the credential wording narrowed and `~/cliproxyapi` denied
in the seatbelt (F2); the subagent variable described as a default with a
log audit (F3); the launcher skips the Claude gate for all-gateway waves
and the classifier's Claude dependency is stated (F4); the void versus
counted-as-it-stands rule written down (F5); the driver stops after a wave
A with no completed run and returns non-zero on any failure (F6);
permission mode, concurrency and effort wording (F7); the figure footnote
scoped to the Claude L3 runs until the extension audit (F8); the
post-processing scope stated (F9). F10 and F11 were confirmations.

Extension A results review (GPT-6-Astra, read-only, 2026-09-06 00:44 local,
on the working tree at commit `0e7f0d1` plus the slide snapshot;
`reviews/2026-09-06_astra_ext_a_results_review.md`, fold record beside it):
15 findings, 6 major, 5 minor, 4 notes, no blocker. Every major accepted:
two audit classes corrected to `indeterminate` (F1); the six blocked steps
were classifier unavailability, not decisions, and the model-id claim
narrowed (F2); the Sol L1 range and the slide's comparison corrected (F3);
the Claude-arm recap restored to three leaked headlines (F4); candidate
counts redefined as test-scored forecasts, with the SARIMA re-scoring in
GPT-5.5 L3 r2 disclosed (F5); the agent-reported L3 study distinguished
from the six saved forecasts in the figure and text (F6). Minor findings
folded: Astra maxima, AGENTS.md read positions, the bundled-skill counts,
the wave A stopped count and the GPT-5.5 L3 r3 verification claim, the
pre-registration timestamp, cells versus runs, the May reference, and the
three over-statements (F7 to F11). The principal numbers reproduced (F12);
the leak scopes and the subagent account were confirmed (F13).

Extensions B and C design review (GPT-6-Astra, read-only, 2026-09-06 00:44
local, on `git diff 0e7f0d1 b1f1fb4` and sections 12 and 13;
`reviews/2026-09-06_astra_ext_bc_design_review.md`, fold record beside
it): 10 findings, 3 major, 3 minor, 4 notes, no blocker. Folded before
launch: the driver takes an ordered list of waves, validates them, and
fails closed on a completion-count error (F1, F4); the chance-bound
sentence removed from section 13 (F2); the machine and quota dependencies
between B and C stated, and the two L3 waves kept apart (F3); gate-parser
failures no longer look like a closed gate, and the retry described as
bounded and pre-wave (F5); the reader's brief describes the delivered
prompt from `run_meta.json` (F6); the mixed GPT-5.5 L3 cell's limits
carried into section 12 (F10). F7 to F9 were confirmations, with the
figure-builder update for the new tags noted as pending post-processing.

## 11. Extension A, pre-registered 2026-09-05 22:30 local: three OpenAI models on the same harness

(Correction 2026-09-06: the commit that registered this section is
timestamped 22:19:09 local, the review fold 22:27:35, the canary's start
22:27:44; the heading's 22:30 was the time as written, not the commit
time. Nothing else in the section is changed.)

Written before any extension run was launched; the numbers go to
`RESULTS.md` afterwards. Nothing above this section changes.

- **Question Q4 (descriptive, cross-vendor).** On the same three frozen
  prompts, the same sandbox and the same Claude Code build, what do
  GPT-6-Astra, GPT-5.6-Sol and GPT-5.5 produce, three fresh sessions per
  level each, and how do the cells read under the rules of section 6?
  Twenty-seven counted runs: tags `astra` (`gpt-6-astra`), `sol`
  (`gpt-5.6-sol`), `gpt55` (`gpt-5.5`), each at L1, L2 and L3, reps 1 to 3.
- **Held fixed**, as in section 3: the pinned inputs, Claude Code 2.1.259
  by absolute path, the sandbox and its identifier scan (extended with the
  three new tags), the seatbelt, the prompt on stdin, auto permissions with
  prompts routed to nobody, `--setting-sources project,local`, the 180-minute
  cap, the print-mode keep-alive, the attempt policy (L1 and L2 never
  resumed, L3 once on an infrastructure failure), the harvest, the
  summariser, the scorer, the independent reader and the audit rules of
  section 5.
- **Route.** The agent's API traffic goes to the operator's local
  CLIProxyAPI gateway at `http://127.0.0.1:8317`, which serves these models
  to Claude Code under their Codex ids from the operator's Codex
  subscription. The runner adds, for these tags only, the environment the
  operator's `poly` launcher uses for the same models: `ANTHROPIC_BASE_URL`
  and `ANTHROPIC_AUTH_TOKEN` (the gateway's local key: it is in the agent's
  environment, as it must be, and is not written to `run_meta.json` or the
  logs; the seatbelt now also denies the gateway's own directory
  `~/cliproxyapi`, from this extension onwards), `CLAUDE_CODE_SUBAGENT_MODEL`
  set to the same model, which is a default for subagents rather than a
  guarantee (an agent definition that names a model overrides it; the
  session logs are audited for the models any subagent actually used),
  `CLAUDE_CODE_ALWAYS_ENABLE_EFFORT=1`,
  `CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY=3`,
  `CLAUDE_CODE_MAX_CONTEXT_TOKENS=272000` and `ENABLE_TOOL_SEARCH=false`.
  The launcher itself runs `bypassPermissions`; the runner keeps the Claude
  arm's `auto` mode with prompts denied, so this arm reproduces the
  launcher's environment, not the operator's interactive session. The
  runner refuses to start unless the gateway is up and lists the model, and
  a gateway-routed L3 resume rebuilds the same route and model before the
  attempt counter is touched.
- **Known differences from the Claude arm, disclosed rather than removed.**
  (a) The context window is pinned at 272,000 tokens; the Claude arm ran at
  the build's default for `claude-fable-5-1` and `claude-opus-5`. (b) The
  claude.ai connectors are disabled by the gateway credentials (the CLI
  says so on stderr: an auth source other than the claude.ai login takes
  precedence) and tool search is off; the Claude arm had the connectors
  available and no counted run used one. (c) At most three tool calls
  execute concurrently. (d) The effort parameter is enabled for these
  models; its effective value is whatever the CLI sends for the model by
  default and is not otherwise set. (e) The auto-mode permission
  classifier still runs on a Claude model through the gateway's Claude
  credentials, so this arm depends on Claude capacity for its permission
  decisions as well as on the Codex subscription for the model. (f) Rate
  limits are the Codex subscription's: the launcher skips its Claude
  five-hour gate for waves made only of gateway-routed runs. A session the
  gateway refuses shows in the log as an API error; the rule for it is the
  wave 2 precedent: a session that ends on an API error before its first
  tool call is **void** (nothing was produced), archived under a `_void_`
  directory and launched again once under the same rep number, and a
  session that made at least one tool call before the error is **counted
  as it stands**, scored on what is on disk and marked incomplete, never
  re-run (L1, L2) or resumable once (L3), exactly as in section 3. (g) The
  keep-alive is on for every level here,
  whereas the Claude arm's L1 and L2 ran without it, so "session finished"
  is not comparable across arms and is reported per arm. (h) Claude Code's
  agent loop, tools and system prompt were built around Claude; a
  cross-vendor difference is a difference in model plus translation layer
  and is reported as descriptive only. Nothing in this design attributes a
  difference between an OpenAI cell and a Claude cell to the model alone.
- **Waves** (in `waves.json`). `ext_canary`: `astra L1 r1` alone, to validate
  the route: tool calls execute, the classifier answers, a result event
  arrives, the harvest works. A canary that fails on route grounds is voided
  and the harness fixed before any counted run, exactly as in section 9; a
  canary that completes counts. `ext_waveA`: the other seventeen L1 and L2
  runs, six at a time. `ext_waveB`: the nine L3 runs, two at a time,
  interleaved by model (`astra`, `sol`, `gpt55`, then the second reps, then
  the third) so that no model takes all the late-night slots.
- **Reading rules.** Section 6 applies per cell: range, repeatability
  labels, process proportions, audit class per run, the L1-versus-L3
  closeness rule within each model. Across models the reading is a
  side-by-side of per-level ranges and process proportions, descriptive.
  The May L3 correction of section 5.2 applies unchanged.
- **Post-processing scope.** The summariser, scorer and figure builder
  regenerate shared derived files (`summary.json`, `results.csv`, the
  figure and table) deterministically from the logs; after the extension
  is processed, `git diff` of the twelve Claude runs' derived files must be
  empty, and the independent reader is run on named extension runs only,
  never with `all`, so the Claude assessments are not rewritten.
- **Review.** One read-only review of this section and the harness diff by
  GPT-6-Astra before the canary, recorded in section 10; the launch record
  continues in section 9.

## 12. Extension B, pre-registered 2026-09-06 01:20 local: the OpenAI models told not to stop

Written before any Extension B run was launched. Sections 1 to 11 are not
changed.

- **Question Q5 (descriptive).** Extension A ended with 20 of 27 OpenAI
  sessions stopped on a question or a plan (RESULTS.md section 8). If the
  prompt tells the model that nobody will answer and that it must finish
  in one session, what do those 20 cells produce, and how do they read
  under section 6? Tags `astraos` (`gpt-6-astra`), `solos` (`gpt-5.6-sol`),
  `gpt55os` (`gpt-5.5`), at exactly the level and rep of the Extension A
  run that stopped: Astra at every level and rep (nine), Sol at L2 and L3
  (six), GPT-5.5 at L2 (three) and L3 reps 1 and 3 (two). The seven
  Extension A runs that wrote a forecast are not re-run, on the owner's
  instruction of 2026-09-06 ("it is only for the ones that failed to give
  forecast"), and no Fable 5.1 session is run for any reason (owner
  instruction, same day). A GPT-5.5 L3 cell read across the two
  extensions therefore mixes one Extension A run with two Extension B runs
  and is reported that way, never as three of one kind; that cell cannot
  become a three-run B cell, and its retained A run is agent-reported and
  leaked, so it cannot meet section 6's recoverability requirement for
  accuracy repeatability. B is a set of fresh attempts on the cells that
  produced no forecast in A, not an isolated estimate of what the note
  does: the cells were selected on their A outcome.
- **The one change.** The runner appends `prompts/one_shot_suffix.md`
  (pinned in `pins.sha256`; a blank line, then one paragraph: "Note from
  the operator: this session is unattended. Nobody can answer a question
  or approve a plan, so do not stop to ask or to propose. Make the most
  reasonable choice for anything unclear, say what you assumed in your
  final message, and finish the whole job in this session.") to the
  frozen prompt on stdin, for these three tags only. The frozen prompt
  files, `AGENTS.md` files and the L3 prompt copy inside the sandbox are
  byte-identical to May and to every other arm; `run_meta.json` records
  the suffix hash and the delivery. The note says nothing about time
  zones, forecast origins or methods: it removes the option of stopping,
  not the ambiguity.
- **Everything else** is Extension A: the gateway route and its eight
  disclosed differences (section 11), the void and counted-as-it-stands
  rule, the attempt policy, the harvest, the summariser, the scorer, the
  reader on named runs, the audit rules of section 5.2 by the literal
  text, and the post-processing scope (the Claude runs' and Extension A's
  derived files must diff empty afterwards).
- **Waves** (`waves.json`): `ext_b_waveA`, the 12 L1 and L2 runs six at a
  time; `ext_b_waveB`, the eight L3 runs two at a time, interleaved by
  model. No canary: the route was validated by Extension A. The driver is
  `launch_extension.sh ext_b_waveA ext_b_waveB`.
- **Reading rules.** Section 6 per cell and the side-by-side reading of
  section 11. The comparison that matters is within vendor: Extension A
  cell against Extension B cell, same model, same prompt plus one
  paragraph. A run that still stops counts as it stands; a run that
  writes a forecast is scored and audited like any other. The Claude arm
  is not re-run with the note, because no Claude session stopped and the
  owner has ruled out further Fable 5.1 sessions; that asymmetry is stated
  wherever B is compared with the Claude arm.
- **Review.** One read-only review of this section together with section
  13 and the harness diff by GPT-6-Astra before launch, recorded in
  section 10.
- **Amendment, 2026-09-06 01:15 local (after launch, before any B run was
  read).** Between about 00:52 and 00:58 local the machine lost its
  network for a few minutes: the orchestrator's own permission classifier
  reported "connection failed", the four Extension C sessions then running
  logged ten API retries each with no status code and recovered, and the
  gateway answered the six Extension B sessions then running with
  `503 auth_unavailable: no auth available (providers=codex,
  model=gpt-6-astra)` (its Codex credential could not be reached). Five of
  the six exhausted Claude Code's ten retries and ended on that error
  after five to twelve minutes and five to twelve tool calls; the sixth
  (`astraos L2 r2`) rode it out. The void rule of section 11 (f) covers an
  API error before the first tool call, and the wave 2 precedent of
  section 3 treats a harness or infrastructure cut-off as not the model's
  outcome. This amendment applies the latter: a session ended by exhausted
  API retries on 5xx or connection errors is an infrastructure failure,
  archived under `_void_ext_b_outage_20260906/` (committed, contributing
  no number) and relaunched once under the same rep, in wave
  `ext_b_relaunch` after `ext_b_waveA` ends. The same rule applies to any
  Extension C session that ends the same way; none had at the time of
  writing. A session that ends on such an error a second time counts as
  it stands.

## 13. Extension C, pre-registered 2026-09-06 01:20 local: the Opus family, three runs per level

Written before any Extension C run was launched. Sections 1 to 12 are not
changed.

- **Question Q6.** The May 2026 reference is one Opus 4.7 session per
  level, and the September Opus 5 cell is one session per level. With
  three sessions per level for Opus 4.7, Opus 4.8 and Opus 5 on the frozen
  prompts, (a) does the May 2026 result repeat on the same model four
  months later on this harness (Opus 4.7, the only within-model
  comparison across the two dates), (b) how do the three Opus generations
  read side by side under section 6, and (c) does the Opus 5 cell become
  repeatable in the section 6 sense? Tags `opus47` (`claude-opus-4-7`),
  `opus48` (`claude-opus-4-8`), reps 1 to 3 at every level, and `opus5`
  reps 2 and 3 at every level beside the counted rep 1 of section 2. The
  ids were checked with a one-turn probe on 2026-09-06 (both answered).
  Twenty-four runs.
- **Held fixed.** Everything in section 3, on the direct route, exactly as
  the Claude arm: frozen prompts with no suffix, no gateway environment,
  the keep-alive on for every level (as the Claude L3 runs had it; the
  Claude L1 and L2 runs did not, and "session finished" is reported per
  arm as section 11 (g) already says), the attempt policy, the harvest and
  the audit rules. Opus 5 rep 1 is not re-run and not resumed again; its
  cell is read as three runs with its rep 1 flagged as the resumed one.
- **Quota.** These runs bill the operator's Anthropic plan, unlike
  Extensions A and B. Every wave passes the usage gate of section 3, and
  the driver waits fifteen minutes and retries when the gate is closed
  (`launch_extension.sh`, GATE_WAIT): a bounded pre-wave retry (the gate
  is read once before each wave, not between batches), so a window
  exhausted during a wave still refuses sessions, which are then void or
  counted as they stand by the wave 2 rule of section 3 and section 11
  (f). Waves: `ext_c_wave1` and `ext_c_wave2`, eight L1 and L2 runs each,
  four at a time; `ext_c_wave3`, the eight L3 runs two at a time,
  interleaved by model. Extension B runs concurrently on the gateway for
  the light waves (up to ten light sessions on the machine at once, more
  than Extension A's six); the two L3 waves are not run together, so no
  more than two L3 sessions train at once (section 7). The two extensions
  are not independent in every resource: B's permission classifier runs on
  a Claude model through the gateway's Claude credentials, and a B session
  can spawn a Claude subagent the same way (RESULTS.md 8.4); whether that
  pool is the same five-hour window C bills is not established here and is
  disclosed rather than assumed.
- **Reading rules.** Section 6 per cell (range, repeatability, process,
  audit class), and three side-by-side readings, all descriptive: Opus
  4.7 September against Opus 4.7 May (same model, same prompts, different
  harness and date; the May run's audit classes of section 5.2 stand);
  the three Opus generations against each other; and the Opus 5 cell of
  three against the Fable 5.1 cell of three. The May-versus-September
  Opus 4.7 comparison is the one place where a difference cannot be a
  model difference; it can be harness, date, configuration or sampling,
  none of which this design separates, and May's L3 keeps its different
  forecast construction (section 5.2). The three September reps show the
  observed September spread; they do not bound stochastic variation.
- **Post-processing scope.** As sections 11 and 12; in addition the
  `opus5` entries of `scoring.json` for rep 1 and its derived rows must
  diff empty. The figure builder gains the three tags; the MAPE figure is
  redrawn with the extension series and the Claude-arm points unchanged.
- **Review.** With section 12, one read-only review by GPT-6-Astra before
  launch, recorded in section 10.
