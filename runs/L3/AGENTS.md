# Forecasting demo — Level 3 working directory

This directory contains the data file and is the working directory for the
Level 3 prompt-engineering demo. The full project description, prompts, and
slides live in the parent repo.

## What you are doing

You are an AI agent invoked in this directory to follow the prompt under
`../../prompts/L3.md` (which the operator pasted as the first user message).
Read your prompt and complete the forecasting task it specifies. Stay inside
this directory for all writes.

## Hard constraints

- **Univariate only.** Use only the load series itself. Do not introduce
  exogenous variables (temperature, calendar features, holidays, day-of-week
  dummies derived from external data, etc.). This is a deliberate constraint
  for the demonstration.
- **Data file:** `opsd_de_load.csv` (read-only from your run dir, in the
  current directory). Two columns: `utc_timestamp` (hourly, UTC) and
  `DE_load_actual_entsoe_transparency` (megawatts). Coverage:
  2015-01-01 to 2020-09-30, ~50k rows.
- **Output structure for your run** (you write into the current directory):
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

- Do not write outside the current working directory.
- Do not install additional Python packages. If a needed package is missing,
  stop, record the gap in `transcript.md`, and either work around it or flag
  it as a blocker for the human operator.
- Do not commit to git from inside your run. Phase-1 commits are done
  manually by the human operator after the run completes.

## Style

- All code comments and `transcript.md` prose: plain language, Feynman-style.
  Audience is JRC policy economists and engineers, not ML researchers.
  Avoid jargon you would not unpack.
- No emoji in code, comments, or transcripts.
- Type hints on every function signature.
