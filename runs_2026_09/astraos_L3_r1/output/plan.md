# Forecasting bake-off plan

1. Verify `opsd_de_load.csv`, the prescribed interpreter, and installed model source. Use a fixed origin for every 168-hour forecast. Data-derived future features must use predictions, not future observations.
2. Write consumer tests in `code/test_study.py`, shared contracts in `code/common.py`, then the six named model modules. Fit initial coefficients and scalers only on data before 2019-10-01. Score the 13 complete validation weeks and final 24 hours without updating coefficients. Earlier validation observations may update forecast contexts at later origins.
3. Select SARIMA orders, Prophet seasonality mode (prior strength fixed), LightGBM tree complexity, and neural epoch count by validation MAPE. Refit fresh models and scalers on all data before 2020-01-01. No test values are arguments to model-fitting functions.
4. Produce test forecasts and five quantiles where supported. Use analytic SARIMA intervals, Prophet posterior-predictive samples, and neural quantile regression. LightGBM has five quantile fits, including the extra 0.025 and 0.975 needed for a 95% interval. Its recursive intervals are conditional on the median path and omit propagation of predictor uncertainty.
5. `code/forecast.py` orchestrates training and writes `metrics.json`, `metrics.csv`, forecast and validation evidence, all required figures, and `transcript.md`. Preserve raw data and write only under this directory. Use `../../.venv/bin/python`, no installs.
6. Verify every metric from the saved forecasts, examine the figure renders, and request independent review. Record failures and departures honestly. No commits or publication.

## Assumptions fixed before seeing test scores

- One forecast issued at 2020-01-01 00:00 UTC covers the entire week. Observed load during that week cannot enter predictors.
- Validation covers all 2,208 hours, weighted equally, rather than choosing favorable weeks.
- Daily SARIMA does not automatically learn weekly seasonality. An explicit 168-hour difference supplies weekly structure; daily residual orders are selected on validation. For horizons up to 168 hours the inverse difference uses only known load.
- PatchTST is not present in Darts 0.41.0. Use its actual `TransformerModel`, not TSMixer, and label the substitution everywhere visible while retaining the requested `patchtst` machine key.
- Neural training uses full default N-BEATS stacks, 168-hour input/output lengths, a recorded 168-hour training stride, at most 30 epochs, and validation-MAPE early stopping. Sparse windows bound compute without shortening the historical span.
- The Jan 1 and Jan 2–7 evaluation groups use UTC dates. Jan 2–7 includes Saturday and Sunday. Calendar predictors use German local time where explicitly supported; any library UTC calendar limitation is disclosed.
- Metric bars are normalized to the naive model (dimensionless), with raw values and units on the bars and in tables.
- If the naive model wins, its native intervals remain unavailable. A separate validation-only signed-error calibration supplies auxiliary intervals for the required winner diagnostics, explicitly labeled as such.

The operator authorized unattended end-to-end delivery. No interactive approval gate is used. An independent Claude reviewer was asked to challenge this protocol before implementation.
