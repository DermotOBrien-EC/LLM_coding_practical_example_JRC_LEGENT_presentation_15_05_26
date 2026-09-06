Done.

Created the full reproducible bake-off in the current directory:

- `code/common.py`
- `code/naive.py`
- `code/sarima.py`
- `code/prophet.py`
- `code/lightgbm_features.py`
- `code/nbeats.py`
- `code/patchtst.py`
- `code/forecast.py`
- helper artifact runner: `code/run_one.py`

Generated outputs:

- `metrics.json`
- `metrics.csv`
- `transcript.md`
- `figures/01_overview.png`
- `figures/02_forecast_comparison.png`
- `figures/03_metric_comparison.png`
- `figures/04_per_day_mape.png`
- `figures/05_winner_with_intervals.png`
- `figures/06_residuals.png`
- `figures/07_feature_importance.png`
- `figures/08_decomposition.png`

Final ranking by test MAPE:

1. LightGBM: 5.46%
2. Prophet: 7.61%
3. TSMixer substitute for PatchTST: 9.58%
4. N-BEATS: 9.81%
5. Seasonal naive: 12.78%
6. SARIMA: 13.81%

Winner: `lightgbm`

Winner probabilistic diagnostics:

- 80% interval coverage: 89.29%
- 95% interval coverage: 96.43%
- Pinball loss q10: 479.31 MW
- Pinball loss q50: 1358.13 MW
- Pinball loss q90: 684.78 MW

Verification run completed:

- `../../.venv/bin/python -m py_compile code/*.py`
- full bake-off run completed with `ALL_DONE winner=lightgbm`
- all required files present
- no em-dashes found in generated prose or code
- palette validated using the dataviz skill's reference six-colour subset

Assumption recorded in `transcript.md`: Darts 0.41.0 does not expose `PatchTSTModel`, so `TSMixerModel` was used as the requested transformer-family univariate substitute.