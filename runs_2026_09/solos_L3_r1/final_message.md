## Completed

| Rank | Model | Test MAPE |
|---:|---|---:|
| 1 | LightGBM | **5.31%** |
| 2 | Prophet | 7.58% |
| 3 | SARIMA | 9.78% |
| 4 | N-BEATS | 10.02% |
| 5 | TSMixer substitute | 10.07% |
| 6 | Seasonal naive | 12.78% |

LightGBM achieved 3,756 MW RMSE and 2,891 MW MAE. Its empirical interval coverage was 84.5% for the nominal 80% interval and 100.0% for the nominal 95% interval. Full results are in `metrics.json:1`, `metrics.csv:1`, and `transcript.md:15`.

Produced:

- All eight figures in `figures/`, including LightGBM feature importance and Prophet decomposition
- All eight requested Python modules in `code/`
- `metrics.json`
- `metrics.csv`
- `transcript.md`

Assumptions and deviations are documented:

- Used the prompt-approved `TSMixerModel` substitute because Darts 0.41.0 does not expose PatchTST, `transcript.md:9`
- Used a 364-day, 8,736-hour annual lag and recursive LightGBM forecasting to prevent validation or test leakage
- Assigned holidays by UTC calendar date to match the frozen evaluation days
- Used CPU for both deep models after the MPS backend exited with code 139, `transcript.md:42`

Successful model fitting and output assembly took 957.9 seconds, approximately 16 minutes. Verification included independent metric recomputation, schema checks, quantile-order checks, 300 dpi image inspection, compilation, Ruff linting, and confirmation that the Discussion is 378 words.

Independent Fable peer review could not run because the gateway probe found `/Users/doob/cliproxyapi/config.yaml` unreadable. I did not silently substitute another review lane.