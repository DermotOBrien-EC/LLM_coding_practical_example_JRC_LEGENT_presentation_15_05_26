# RUNBOOK — Manual Phase 1 Runs

This document describes how a human operator runs each of L1, L2, L3 manually,
once each, producing the artifacts that feed the Phase-3 slides.

## Prerequisites

- Phase 0 is complete: `git status` is clean, `uv sync` has produced a working
  venv, `data/opsd_de_load.csv` loads without error.
- A few free hours of operator time, ideally on the same day so model versions
  do not drift between runs.

## Per-level procedure

For each level X ∈ {L1, L2, L3}:

### 1. Copy the prompt to your clipboard

Open a new terminal window. **Do not reuse a terminal where you previously ran
another level** — the working directory state and the `.claude/projects/`
session log can leak. Quit any Claude Code app instance already open on a
working directory adjacent to this one.

```sh
cd ~/dev/LLM_coding_example_jrc_andres/runs/LX
cat ../../prompts/LX.md | pbcopy
```

`pbcopy` is macOS-specific; on Linux use `xclip -selection clipboard` or
`wl-copy`.

### 2. Start a fresh Claude Code session

In the same terminal, now launch Claude Code:

```sh
claude
```

Confirm at the prompt that:
- The CLI shows the cwd as `runs/LX`.
- This is a fresh session (no `--continue` flag).
- The model is Opus 4.7.

### 3. Paste the prompt as the first message

At the Claude Code prompt, paste your clipboard contents (Cmd+V on macOS).
This is the first user message in the session — do not type anything before
it.

### 4. Let the agent work

Do not interject unless the agent is genuinely stuck. The agent's context
depends on the level:

- **L1**: only the prompt and `opsd_de_load.csv` in the working directory. No
  `AGENTS.md`, no schema, no constraints. The agent may produce anything (a
  single script, ad-hoc files, no metrics, no transcript). Capture whatever
  comes out.
- **L2**: as L1 plus a minimal `AGENTS.md` mentioning the data file and a
  couple of library hints. Still no schema or constraints. Likely produces a
  script and a plot.
- **L3**: a detailed `AGENTS.md` with the univariate constraint, the
  `metrics.json` schema, the expected `code/`/`figures/`/`transcript.md`
  layout, and explicit do-nots. Should produce all four artifacts.

Expected runtime: L1 may finish in under a minute (or hang asking for
clarification); L2 typically 5–15 minutes; L3 typically 20–60 minutes
including model fitting.

### 5. Save the chat export

When the agent reports completion, **before quitting the session**, run:

```
/export session_export.md
```

This saves the full Claude Code chat to `session_export.md` in the current
working directory. Note: this is **not** the same file as `transcript.md` —
`transcript.md` is the agent-written decisions log; `session_export.md` is
the verbatim chat record. Both are kept.

### 6. Verify the artifacts

```sh
ls
cat metrics.json 2>/dev/null | python -m json.tool 2>/dev/null || echo "(no valid metrics.json — that's fine, record it)"
```

L3 should have produced `code/`, `figures/`, `metrics.json`, `transcript.md`,
and `session_export.md`. L1 and L2 may have produced an ad-hoc mix of files.
**Do not ask the agent to fix or restructure anything after the fact** —
record whatever is missing (or unexpected) in a `transcript.md` you write
manually if the agent didn't produce one. The honesty of the comparison
depends on capturing what the agent did on its own, not what it could be
coaxed to produce.

### 7. Close the session and commit

```sh
exit       # or /quit inside the Claude Code TUI
cd ~/dev/LLM_coding_example_jrc_andres
git add runs/LX
git commit -m "phase 1: LX run complete"
```

---

## If the agent stops mid-task

Two options:

- **Resume (`claude --continue`)** — Pros: less wasted work, faster to a final
  artifact. Cons: muddies the demonstration — the L1/L2/L3 comparison is
  partially about how the agent recovers from a stuck state, and a resume
  hides that information.
- **Restart cleanly** — Pros: each level's outcome is a single, honest first
  attempt, which is the comparison we want for the slides. Cons: you lose
  whatever partial work the first attempt produced.

**Default: restart cleanly** for L1 and L2. Their interest *is* the first-try
behaviour. For L3, **resume is acceptable** if the model is making progress
and the stop was due to an infrastructure issue (e.g., bash timeout, not a
genuine logical block). Record the decision in `runs/LX/transcript.md`.

---

## Reproducibility note for the slide deck

The figures and metrics produced by Phase 1 are non-deterministic — different
seeds, different model class choices, different walltimes will produce
different numbers. The slide deck cites *what we observed in our specific
runs*, not "what Claude always does." Keep the original artifacts; do not
re-run levels to "polish" results.
