Environment checks passed:

- 50,400 hourly rows, no missing values, duplicates, or gaps.
- Splits contain 41,616 train, 2,208 validation, and 168 test observations.
- All required packages import from `../../.venv`.
- Darts 0.41.0 has no `PatchTSTModel`, so the documented fallback will be `TSMixerModel`.

## Implementation plan

1. **Build shared infrastructure in `code/common.py`**
   - Define split constants, model names, fixed colours, random seed, and typed result dataclasses.
   - Load and validate the CSV fail-closed against the stated row count, timestamps, hourly frequency, and missing-value contract.
   - Implement MAPE, RMSE, MAE, per-day MAPE, holiday breakdown, coverage, and pinball loss.
   - Add utilities for probabilistic sample quantiles, JSON-safe values, and 300 dpi figure output.

2. **Implement the six model modules**
   - `code/naive.py`: exact 168-hour seasonal lag.
   - `code/sarima.py`: a small explicit SARIMAX order grid with daily seasonality, selected by validation MAPE, then refitted on train plus validation. Prediction intervals will use the fitted state-space distribution.
   - `code/prophet.py`: Darts Prophet with German holidays and explicit daily, weekly, and yearly seasonality. Validation will select seasonality mode and prior scale; final uncertainty will come from Prophet samples.
   - `code/lightgbm_features.py`: timestamp, holiday, lag, and strictly past-only rolling features. Candidate tree configurations will be scored with blocked validation forecasts. Final quantile models will cover 0.025, 0.1, 0.5, 0.9, and 0.975 so both requested interval levels are available if LightGBM wins.
   - `code/nbeats.py`: default N-BEATS stack, 168-hour input/output, no more than 30 epochs, validation-MAPE early stopping, then refit for the selected epoch count. Monte Carlo dropout will provide probabilistic samples.
   - `code/patchtst.py`: TSMixer substitution under the requested `patchtst` result name, with the same horizon, validation, refit, and probabilistic procedure. The substitution will be disclosed in `transcript.md`.

3. **Prevent temporal leakage**
   - Candidate models will fit only on data through 2019-09-30.
   - Validation will use blocked rolling-origin forecasts in 168-hour chunks, plus the final 24 hours. Earlier validation observations may become history only after their forecast block has been scored.
   - Test forecasting will be a single 168-hour forecast from the 2019-12-31 cutoff.
   - LightGBM test lag and rolling features will be updated recursively with forecasts, never with held-out test observations.

4. **Create `code/forecast.py`**
   - Run all six models in a deterministic order and record individual runtimes.
   - Recompute all test metrics centrally rather than trusting model-local summaries.
   - Select the winner strictly by held-out test MAPE.
   - Write `metrics.json`, flattened `metrics.csv`, and the methods-style `transcript.md`.

5. **Generate and inspect the figures**
   - Produce all six required figures and conditional LightGBM or Prophet figures.
   - Keep a fixed model-to-colour mapping throughout.
   - For `03_metric_comparison.png`, use three aligned bar panels rather than one mixed-unit axis. MAPE and MW cannot share a meaningful numeric scale. It will retain the requested MAPE sorting and labels above every bar.
   - Render at the requested physical dimensions and 300 dpi, then inspect the PNGs for clipping, overlapping legends, labels, and consistent ranges.

6. **Verify the complete study**
   - Run syntax and import checks before the expensive fit.
   - Execute with `../../.venv/bin/python code/forecast.py`.
   - Independently recompute metrics from saved forecasts, validate the JSON schema and CSV rows, inspect image dimensions and metadata, and scan added prose for forbidden em-dashes.
   - Attempt the required independent review transport. If it is unavailable, report that rather than substituting a self-review.

Approve this plan and I will implement and run it.