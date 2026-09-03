You are reviewing an experiment design and its launch scripts BEFORE the expensive runs start. Read-only. Do not modify any file.

Repository: the current working directory (a small Python project: a three-level prompt-engineering experiment that forecasts German hourly electricity load with Claude Code at three prompt/workspace specificity levels; original May-2026 runs are under runs/L1, runs/L2, runs/L3; prompts under prompts/).

Files to review, in this order:
1. runs_2026_09/DESIGN.md        (the pre-registered design for the September re-run)
2. scripts/run_headless.sh       (launches one isolated headless Claude Code session per run)
3. scripts/launch_wave.sh        (runs several of those in parallel)
4. scripts/summarize_runs.py     (parses the stream-json session logs afterwards)
Context you may also read: README.md, RUNBOOK.md, prompts/L1.md, prompts/L2.md, prompts/L3.md, runs/L2/AGENTS.md, runs/L3/AGENTS.md, runs/L1/forecast.py, runs/L2/forecast.py, runs/L3/metrics.json.

Facts established by the operator, take them as given:
- Claude Code 2.1.259 headless `-p` works with `--dangerously-skip-permissions` on subscription auth; `--setting-sources project,local` removes the user-level CLAUDE.md (probe verified); an AGENTS.md in the cwd is NOT auto-loaded into context (probe verified); the stream-json result event carries usage, cost and num_turns; a `rate_limit_event` line reports five-hour-window utilisation.
- The machine has 15 CPU cores; L3 runs fit N-BEATS and TSMixer on CPU and took ~20 minutes of pure compute in May.
- macOS, zsh is the interactive shell but the scripts run under bash.

Questions, answer each with a numbered finding or "no finding":
1. Design validity: any confound or unfairness in comparing (a) Fable 5.1 x3 per level, (b) Opus 5 x1 per level, (c) the May 2026 Opus 4.7 single runs, given the stated deviations? Is anything in the design under-specified so that two reasonable operators would run it differently?
2. Reading rules: are the pre-stated reading rules in DESIGN.md section 6 sound, or do they let the author conclude something the data cannot support? Propose tighter wording where needed.
3. run_headless.sh: any bug that would make a run silently differ from the design (env leakage, prompt mangling by `$(cat ...)` or argv, the perl alarm not killing the child tree, the exit code being lost, quoting)? Note `set -- $spec` word-splitting in launch_wave.sh under bash.
4. summarize_runs.py: any bug that would mis-count tokens, tool calls, files, or mis-detect the AGENTS.md read (both Read tool and shell reads such as `cat AGENTS.md`, `head`, `sed -n`, `less`, or a glob like `cat *.md`)?
5. Anything the design should measure that it does not, given the three questions Q1-Q3 in DESIGN.md section 1? Keep to measurements obtainable from the run directory and the session log without re-prompting the agent.

Output contract (plain text, no code fences around the whole answer):
FINDINGS
  F1. [BLOCKER|MAJOR|MINOR] <one-line claim>
      Where: <file:line or DESIGN.md section>
      Why it matters: <concrete failure scenario>
      Fix: <specific change>
  ...
NO-FINDING AREAS
  <list the question numbers where you found nothing, with one line each saying what you checked>
SUMMARY
  <three lines max>
