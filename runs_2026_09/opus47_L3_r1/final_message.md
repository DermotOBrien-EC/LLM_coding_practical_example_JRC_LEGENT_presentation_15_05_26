All deliverables are intact:

- **8 code modules** in `code/` (7 required + `write_transcript.py`)
- **8 figures** at 300 dpi in `figures/`
- **metrics.json** / **metrics.csv** / **transcript.md** at the root

**Final ranking (test MAPE, 2020-01-01 to 2020-01-07, 168 hours):**

| Rank | Model | MAPE | RMSE | Runtime |
|---|---|---:|---:|---:|
| 1 | LightGBM (features) | 3.24% | 2167 MW | 162.6 s |
| 2 | Prophet + DE holidays | 7.61% | 4874 MW | 17.9 s |
| 3 | TSMixer (PatchTST substitute) | 8.28% | 5513 MW | 231.1 s |
| 4 | SARIMA(1,0,1)(1,1,1,24) | 9.79% | 6632 MW | 588.4 s |
| 5 | N-BEATS | 11.00% | 7118 MW | 26.3 s |
| 6 | Seasonal-naive (168h) | 12.78% | 8809 MW | 0.0 s |

LightGBM wins by a wide margin and is the only model uniformly accurate across the whole week (Jan 1 holiday: 2.24% MAPE; Jan 2-7: 3.41%). Winner interval coverage is 74% / 92% against nominal 80% / 95%. Total fit-time cost was ~1026 s wall-clock.