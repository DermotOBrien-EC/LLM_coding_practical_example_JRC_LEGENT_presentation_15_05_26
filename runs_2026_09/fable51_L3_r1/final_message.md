The bake-off is complete and every required output is in place.

**Result:** LightGBM on engineered features wins the held-out week with a test MAPE of 5.03 %. It also won validation, so the ranking is not a test-week fluke for the winner.

| Model | Test MAPE (%) | 1 Jan holiday (%) | 2 to 7 Jan (%) |
|---|---:|---:|---:|
| LightGBM | 5.03 | 2.08 | 5.52 |
| Prophet (DE holidays) | 7.61 | 13.28 | 6.67 |
| TSMixer (PatchTST slot) | 7.62 | 3.89 | 8.24 |
| N-BEATS | 10.25 | 4.09 | 11.28 |
| SARIMA | 12.34 | 4.87 | 13.59 |
| Seasonal naive | 12.78 | 6.52 | 13.82 |

**What the week exposed.** The naive and SARIMA models copy the Christmas week forward and under-predict working days by about 10 GW. The deep networks see only that distorted week as input and mis-shape the weekend and the first working days. Prophet knows the holiday but applies the wrong daily profile on 1 January. LightGBM's own worst day is Monday 6 January, Epiphany, a regional holiday absent from the federal holiday flag. Its 95 % interval covers 94 % of hours, but its 80 % interval covers only 57 %.

**Decisions worth knowing about**, all recorded in `transcript.md`:
- The installed darts has no PatchTST, so TSMixer fills that slot, as the task allowed.
- SARIMA compares two families, and validation chose a weekly-differenced ARMA with a 24-hour season, which is the task's "let weekly structure emerge from differencing" reading.
- LightGBM forecasts recursively so that no test observation feeds its 24-hour lag; calendar features are read in Berlin local time.
- All six models are selected on one protocol: parameters from Train only, scored at seven rolling origins across the validation window, then refit on Train plus Validation.
- The full pipeline reran end to end in 1354 s and reproduced every test MAPE exactly.

**Outputs:** `code/` (eight modules), `figures/` (all eight PNGs at 300 dpi; 07 and 08 apply because LightGBM and Prophet finished first and second), `metrics.json`, `metrics.csv`, `transcript.md`, plus a `forecasts/` cache holding each model's test forecast, quantiles, and full validation table.

**Not done:** no Codex cross-review ran in this session, since it was non-interactive, so the study has had a single-author review only. I also could not verify git status from the sandbox, and nothing was committed, per the working-directory rules.