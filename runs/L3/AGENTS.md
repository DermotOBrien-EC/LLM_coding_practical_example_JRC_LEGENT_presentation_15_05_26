# Forecasting bake-off — Level 3 working directory

This directory is the working directory for the Level 3 prompt of a JRC C4
prompt-engineering demo. The full study description and the actual prompt
text live in `../../prompts/L3.md`.

## What you are doing

You are an AI agent invoked here to follow the prompt the operator has
pasted as the first user message. That prompt describes a six-model
univariate forecasting bake-off for German hourly electricity load on a
held-out 168-hour test window (2020-01-01 to 2020-01-07).

Stay inside this directory for all writes. Read freely from
`./opsd_de_load.csv`. Do not modify files outside this directory.

## Allowed and forbidden inputs

- **Allowed**: the load series; features derivable from the `utc_timestamp`
  column (hour, day-of-week, month, is_weekend, German public holidays via
  the `holidays` package); lag features and rolling-window features from
  the load itself.
- **Forbidden (external data)**: temperature, weather, gas prices, market
  prices, anything not in `opsd_de_load.csv`.

Time-derived features and German public holidays are *not* exogenous in the
forecasting sense — they are knowable a priori without consulting any other
dataset, so they are allowed.

## Output structure (you write into the current directory)

```
code/
    common.py
    naive.py
    sarima.py
    prophet.py
    lightgbm_features.py
    nbeats.py
    patchtst.py
    forecast.py            — orchestrator
figures/
    01_overview.png        — full series + test window highlighted
    02_forecast_comparison.png  — 2 × 3 small-multiples grid
    03_metric_comparison.png    — MAPE / RMSE / MAE bars
    04_per_day_mape.png         — per-day MAPE per model
    05_winner_with_intervals.png— winner with 80/95% PI bands
    06_residuals.png             — winner residuals by hour and day
    07_feature_importance.png    — only if LightGBM is in top 2
    08_decomposition.png         — only if Prophet is in top 2
metrics.json
metrics.csv
transcript.md
```

All figures: 300 dpi, sans-serif, units in axis labels, non-overlapping
legends, consistent qualitative palette across all panels so each model
keeps its colour throughout.

## `metrics.json` schema

```json
{
  "test_start": "2020-01-01",
  "test_end":   "2020-01-07",
  "n_test_observations": 168,
  "models": [
    {
      "name": "<naive|sarima|prophet|lightgbm|nbeats|patchtst>",
      "mape_test_pct": 0.0,
      "rmse_test_mw":  0.0,
      "mae_test_mw":   0.0,
      "mape_jan1_pct": 0.0,
      "mape_jan2_to_jan7_pct": 0.0,
      "runtime_seconds": 0.0,
      "hyperparameters": {}
    }
  ],
  "winner": "<model name>",
  "winner_coverage_80pct": 0.0,
  "winner_coverage_95pct": 0.0,
  "winner_pinball_loss_q10": 0.0,
  "winner_pinball_loss_q50": 0.0,
  "winner_pinball_loss_q90": 0.0,
  "total_runtime_seconds": 0.0
}
```

## Do NOT

- Do not write outside this directory.
- Do not pull in external data sources (weather, prices, etc.).
- Do not use the test window (2020-01-01 to 2020-01-07) for any model
  fitting, hyperparameter selection, or model-class choice.
- Do not skip the refit-on-Train+Validation step before forecasting test.
- Do not silently drop or impute missing hours (the data is clean — verify).
- Do not install additional Python packages. The venv at `../../.venv` has
  everything: `pandas`, `numpy`, `matplotlib`, `statsmodels`, `pmdarima`,
  `darts[all]` (which bundles Prophet, torch, xgboost, etc.), `lightgbm`,
  and `holidays`. If your shell's `python` resolves elsewhere, use
  `../../.venv/bin/python` directly.
- Do not commit to git from inside your run. The human operator commits
  the directory after the run completes.

## Style

- All code comments and `transcript.md` prose: plain language, Feynman-
  style. The audience is JRC policy economists and engineers, not ML
  researchers. Avoid jargon you would not unpack.
- No emoji in code, comments, or transcripts.
- Type hints on every function signature.
- One module per model (see `code/` layout above), with a thin orchestrator
  (`code/forecast.py`) that imports them and writes the outputs.
