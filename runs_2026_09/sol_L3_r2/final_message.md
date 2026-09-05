Verified before planning:

- CSV has exactly 50,400 ordered hourly rows, no missing values, duplicates, or gaps.
- Exact split sizes are 41,616 train, 2,208 validation, 43,824 final-refit, and 168 test rows.
- Darts 0.41.0 does not expose `PatchTSTModel`; `TSMixerModel` is installed, so I will use it as the required transformer-style substitute and record this in `transcript.md`.
- Apple MPS acceleration is available.
- The six-color tab10 ordering has been validated for color-vision separation.

## Plan

1. **Build and test shared contracts**
   - Create `code/common.py` with typed dataclasses, strict data validation, split constants, MAPE/RMSE/MAE, pinball loss, interval coverage, per-day metrics, German holiday logic, fixed colors, and plotting style.
   - Create `code/test_pipeline.py` covering exact split isolation, metric calculations, interval ordering, and LightGBM feature leakage.
   - Keep test observations out of every model function. Models receive the test timestamps only; scoring happens after all forecasts return.

2. **Implement the six model modules**
   - `code/naive.py`: 168-hour seasonal lag.
   - `code/sarima.py`: compare three compact SARIMAX daily-seasonal orders on validation MAPE, refit the winner through 2019-12-31, and obtain Gaussian analytic quantiles.
   - `code/prophet.py`: German holidays plus daily, weekly, and yearly seasonality; compare three changepoint prior scales, refit, and draw 1,000 predictive samples.
   - `code/lightgbm_features.py`: timestamp features, 24-hour and 168-hour lags, a day-of-week-aligned 8,736-hour annual lag, and shifted rolling statistics. Validation and test forecasts will be recursive, so no validation or test actual enters a future feature row. Fit point and 0.025/0.1/0.5/0.9/0.975 quantile models.
   - `code/nbeats.py`: default N-BEATS stack configuration, 168-hour input/output chunks, quantile likelihood, at most 30 epochs, and early stopping on logged validation MAPE.
   - `code/patchtst.py`: TSMixer substitution with the same horizon, quantile likelihood, epoch limit, and validation-MAPE stopping.
   - Neural validation selects the training epoch. The selected epoch count is then used for a fresh refit on train plus validation.

3. **Create the orchestrator**
   - `code/forecast.py` will run all models with fixed seeds, record runtimes and selected hyperparameters, score only after forecasting, rank by test MAPE, calculate winner coverage and pinball losses, and write `metrics.json` and `metrics.csv`.
   - It will fail closed on wrong timestamps, lengths, non-finite predictions, interval inversions, or accidental access to post-2019 data during selection.

4. **Generate publication figures**
   - Produce the six required figures at 300 dpi with fixed model colors.
   - For `03_metric_comparison.png`, use three aligned bar panels rather than one mixed-unit axis. MAPE is percent while RMSE and MAE are MW, so putting all three on one numerical axis would be misleading.
   - Produce LightGBM importance or Prophet decomposition only when the relevant model ranks in the top two.

5. **Write `transcript.md`**
   - Include the requested methods sections, sorted results table, 200–400 word discussion, production recommendation, seeds, substitution disclosure, total runtime, and exact `uname -a` output.
   - Explicitly document recursive LightGBM prediction and any quantile rearrangement needed to prevent crossing.

6. **Verify and review**
   - Run pytest, the full pipeline from the project venv, JSON-schema checks, CSV/JSON agreement checks, split and leakage assertions, and a second complete reproducibility run if runtime permits.
   - Open every generated PNG and inspect labels, legends, clipping, shared scales, and interval bands.
   - Obtain one independent Claude peer review of the finished study, evaluate each finding, apply accepted fixes, and rerun affected checks.

Approve this plan and I will implement it.