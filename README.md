# LLM_coding_example_jrc_andres

Three-level prompt-engineering demo for a JRC C4 presentation: the same
univariate German electricity load forecasting task run through Claude Code
(Opus 4.7) at three specificity levels.

## Layout

- `prompts/{L1,L2,L3}.md` — the three frozen prompts. Inputs to the experiment.
- `data/opsd_de_load.csv` — German hourly load, 2015–2020, ~50k rows
  (Open Power System Data).
- `scripts/fetch_data.py` — produces `data/opsd_de_load.csv` from the OPSD
  source CSV. Re-runnable.
- `runs/{L1,L2,L3}/` — populated manually in Phase 1, one fresh Claude Code
  session per level. Each contains `code/`, `figures/`, `transcript.md`,
  `metrics.json`, and `session_export.md`.
- `slides/` — Phase 3 output.
- `AGENTS.md` — rules for every AI agent invoked in this repo.
- `RUNBOOK.md` — step-by-step for the human operator running Phase 1.

## Quickstart

```sh
uv sync
uv run python scripts/fetch_data.py      # produces data/opsd_de_load.csv
```

Then follow `RUNBOOK.md` for the Phase-1 manual runs.
