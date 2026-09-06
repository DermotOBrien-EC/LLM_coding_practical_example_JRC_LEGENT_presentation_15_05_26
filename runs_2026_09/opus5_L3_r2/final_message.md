Done. All deliverables are in place and verified.

## Results

| Model | Test MAPE | RMSE (MW) | Jan 1 (holiday) | Jan 2–7 | Validation MAPE |
|---|---:|---:|---:|---:|---:|
| **LightGBM** | **5.47%** | 3586 | 4.56% | 5.62% | 3.13% |
| Prophet | 7.31% | 4721 | 13.35% | 6.30% | 6.41% |
| PatchTST (TSMixer) | 8.62% | 5500 | 4.62% | 9.28% | 6.74% |
| N-BEATS | 11.13% | 7806 | 3.00% | 12.49% | 5.42% |
| Seasonal naive | 12.78% | 8809 | 6.52% | 13.82% | n/a |
| SARIMA | 14.33% | 9482 | 6.24% | 15.68% | 4.93% |

Winner LightGBM: 80% coverage 77.4%, 95% coverage 98.8%; pinball 401.8 / 1177.4 / 372.9.

## The finding that drives everything

The test week's preceding 168 hours are 25–31 December 2019 — **the most anomalous week in the entire dataset** (0.7th percentile of all seven-day means, 12.9% below the test week). Every model leaning on "last week" inherits that distortion. SARIMA is the only entrant with *negative* skill (−12.1% vs naive). The deep models get Jan 1 best (3.00%, 4.62%) for the wrong reason — a holiday-shaped input context suits a holiday — then collapse on the ordinary weekend (N-BEATS 28.2% Sunday).

Prophet fails inversely and instructively: second overall, but **worst of all six on the holiday** despite being the only model with a fitted New Year's Day term. Its holiday effect is one multiplier on the day, so it can lower the level but cannot flatten the morning industrial ramp — it over-predicts 06:00–09:00 by ~11,000 MW.

## Three bugs found and fixed

1. **SARIMA was silently reproducing the naive baseline.** `simple_differencing=True` makes statsmodels return forecasts *of the differenced series*. It scored a plausible 7.12% — identical to naive's 7.12% to two decimals, which is what exposed it. A dead model with a healthy-looking metric.
2. **`code/prophet.py` shadowed the real `prophet` package** (Python puts a script's dir on `sys.path[0]`), making darts report Prophet as uninstalled.
3. **Early stopping over-counted refit epochs** by the patience window.

## Judgement calls worth your attention

- **PatchTST is absent from darts 0.41.0.** Substituted `TSMixerModel` (named acceptable by the prompt), recorded in module docstring and transcript.
- **LightGBM forecasts recursively**, not teacher-forced. A `lag_24h` for 7 January sits inside the test week; using it would have given LightGBM a 24-hour task while the others did 168. The teacher-forced variant (3.66% vs 5.47%) is reported separately as a diagnostic.
- **Validation rank barely transferred to test** (Spearman 0.10, p=0.87). SARIMA was best non-winner on validation and worst on test. I've flagged this as the study's main methodological caveat rather than burying it — no validation origin required forecasting *out of* a holiday period, which is exactly the test task.

I did not retro-fit any grid after seeing test scores; grids were fixed from validation-origin probes only.

`transcript.md` carries a `verify_narrative` guard that recomputes all 22 numbers quoted in the prose and refuses to write the file if any drifts — I proved it goes red before trusting it. Runtimes in the table are inflated by three-way CPU contention (~59 min wall clock, 155 min summed), which the transcript states.