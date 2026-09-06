Completed the study: **LightGBM wins**, and the full sequential run took **9.7 minutes**.

### Test results: 1–7 January 2020

| Model | MAPE (%) | RMSE (MW) | MAE (MW) |
|---|---:|---:|---:|
| **LightGBM** | **5.11** | **3,639** | **2,774** |
| Prophet + German holidays | 8.18 | 4,900 | 4,125 |
| N-BEATS | 10.23 | 6,711 | 5,406 |
| Transformer substitute | 10.99 | 7,328 | 5,636 |
| Seasonal naive | 12.78 | 8,809 | 7,238 |
| SARIMA | 14.66 | 9,466 | 8,165 |

LightGBM reduced MAPE by **60.0%** relative to naive. Its interval coverage was **79.2% for the nominal 80% band** and **97.6% for the nominal 95% band**. Pinball losses at q10/q50/q90 were **470.6 / 1,213.3 / 501.3 MW**.

### Deliverables

- [Study write-up](transcript.md), including validation results, holiday breakdown and discussion.
- [Metrics CSV](metrics.csv) and [metrics JSON](metrics.json).
- **Eight 300-dpi figures**, including [forecast comparison](figures/02_forecast_comparison.png) and [winner with intervals](figures/05_winner_with_intervals.png).
- All requested scripts under `code/`, plus saved forecasts, validation traces and provenance under `artifacts/`.
- [Review and verification record](artifacts/review_record.md).

**Recorded assumptions:** one fixed-origin 168-hour forecast, UTC calendar features, and a 364-day annual lag. Darts lacked PatchTST, so the sixth model is explicitly labeled **TransformerModel**, not PatchTST.

LightGBM’s hardest day was Jan 6, not Jan 1. SARIMA converged but failed to beat naive. This single holiday week does not establish year-round superiority.

### Reproduce

```sh
../../.venv/bin/python -u code/forecast.py
../../.venv/bin/python code/verify_outputs.py
```

Five tests, artifact checks and independent peer verification passed. The documented mypy internal error prevented strict static-type verification.