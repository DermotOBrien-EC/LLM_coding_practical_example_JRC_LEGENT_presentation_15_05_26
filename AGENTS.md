# LLM_coding_example_jrc_andres

This project supports a 15-minute JRC C4 (energy/transport/climate) presentation
on prompt engineering for research code. It runs the same forecasting task at
three specificity levels (L1 minimal / L2 average / L3 ideal) through Claude
Code with Opus 4.7, producing three isolated runs whose artifacts feed slides.

## What you are doing in this repo

If you are reading this file, you are an AI agent invoked in a Phase-1 manual
run. Your job is to follow the prompt under `prompts/L<level>.md` from the
working directory `runs/L<level>/`. Read your level's prompt, then complete
the forecasting task it specifies.

## Hard constraints (apply to EVERY level)

- **Univariate only.** Use only the load series itself. Do not introduce
  exogenous variables (temperature, calendar features, holidays, day-of-week
  dummies derived from external data, etc.). This is a deliberate constraint
  for the demonstration — exogenous variables would muddy the
  prompt-engineering lesson.
- **Data file:** `../../data/opsd_de_load.csv` (read-only from your run dir).
  Two columns: `utc_timestamp` (hourly, UTC) and
  `DE_load_actual_entsoe_transparency` (megawatts). Coverage:
  2015-01-01 to 2020-09-30, ~50k rows.
- **Output structure for your run** (you write into your own level's dir only):
    - `code/` — every script you write
    - `figures/` — every figure you produce
    - `transcript.md` — a Markdown log of your decisions and any issues
    - `metrics.json` — see schema below

## Output schema for `metrics.json`

A single JSON object with these keys:

```json
{
  "model_class": "<SARIMA|Prophet|N-BEATS|N-HiTS|other>",
  "mape_test_pct": 0.0,
  "train_start": "YYYY-MM-DD",
  "train_end":   "YYYY-MM-DD",
  "test_start":  "YYYY-MM-DD",
  "test_end":    "YYYY-MM-DD",
  "n_test_observations": 0,
  "runtime_seconds": 0.0
}
```

## Do NOT

- Do not edit any file under `prompts/`. Those are frozen experimental inputs.
- Do not write into any `runs/` subdirectory other than your own level's dir.
- Do not modify `AGENTS.md`, `pyproject.toml`, or `RUNBOOK.md`.
- **Do not install additional Python packages.** If a needed package is
  missing, stop, record the gap in `transcript.md`, and either work around it
  (e.g., implement a simpler model using what is available) or flag it as a
  blocker for the human operator. The human handles installs between runs.
- Do not commit to git from inside your run. Phase-1 commits are done manually
  by the human operator after the run completes.

## Style

- All code comments and `transcript.md` prose: plain language, Feynman-style.
  This is teaching material; the audience is JRC policy economists and
  engineers, not ML researchers. Avoid jargon you would not unpack.
- No emoji in code, comments, or transcripts.
- Type hints on every function signature.
