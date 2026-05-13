# LLM_coding_example_jrc_andres

Three-level prompt-engineering demo for a JRC C4 presentation. The same
forecasting task (Germany hourly electricity load for the first week of
January 2020) is run through Claude Code (Opus 4.7) at three levels of
*user sophistication* — prompt and workspace context scale together to
model a beginner (L1), an average user (L2), and a pro (L3).

## Layout

- `prompts/{L1,L2,L3}.md` — the three frozen prompts. Inputs to the
  experiment.
- `data/opsd_de_load.csv` — canonical copy of the German hourly load data
  (Open Power System Data, 2015–2020, ~50k rows).
- `scripts/fetch_data.py` — re-runnable producer of `data/opsd_de_load.csv`.
- `runs/L1/` — beginner workspace: just the data file (`opsd_de_load.csv`).
  No `AGENTS.md`. The agent has no project context beyond the prompt.
- `runs/L2/` — average-user workspace: the data file + a short, generic
  `AGENTS.md` mentioning the data and a couple of library hints.
- `runs/L3/` — pro workspace: the data file + a detailed `AGENTS.md` with
  the univariate constraint, output schema (`metrics.json`), do-nots, and
  style guidance.
- `slides/` — Phase 3 output.
- `RUNBOOK.md` — step-by-step for the human operator running Phase 1.

The deliberate point: workspace context scales *with* the prompt. A real
beginner doesn't have `AGENTS.md` set up; a pro does. The three rows of the
comparison table aren't just three different prompts, they're three different
user personae.

## Quickstart

```sh
uv sync
uv run python scripts/fetch_data.py      # produces data/opsd_de_load.csv
```

Then follow `RUNBOOK.md` for the Phase-1 manual runs.
