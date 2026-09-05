Done.

Wrote the full self-contained study in:

- `code/common.py`
- `code/naive.py`
- `code/sarima.py`
- `code/prophet.py`
- `code/lightgbm_features.py`
- `code/nbeats.py`
- `code/patchtst.py`
- `code/forecast.py`
- `figures/01_overview.png`
- `figures/02_forecast_comparison.png`
- `figures/03_metric_comparison.png`
- `figures/04_per_day_mape.png`
- `figures/05_winner_with_intervals.png`
- `figures/06_residuals.png`
- `figures/07_feature_importance.png`
- `figures/08_decomposition.png`
- `metrics.json`
- `metrics.csv`
- `transcript.md`

Winner: **LightGBM**, test MAPE **3.77%**.

Key verification completed:

- Dataset shape and splits checked: 50,400 rows, no NaNs, hourly spacing, test `n=168`.
- All six models ran end-to-end.
- Validation selection was done before train-plus-validation refit.
- `PatchTSTModel` was not exposed by installed Darts 0.41.0, so `TSMixerModel` was used and recorded in `transcript.md`.
- All required figures are present at 300 dpi.
- LightGBM and Prophet optional figures were produced because both finished in the top two.
- `metrics.json` schema checks passed.
- Checked added prose and code for em-dashes, none found.