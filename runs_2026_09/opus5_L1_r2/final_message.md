The two LightGBM fits on the full 43,824-hour history are the remaining work. I have a watcher armed on the output file that will notify me the moment the run completes, so I'll stop polling and pick this up when it fires.

Here's where things stand:

**Verified so far**
- Data is complete and clean: 50,400 hours, no gaps, no NaNs, validated on load rather than silently interpolated.
- 7/7 tests pass, including a forecast-invariance test that scrambles every observation from the forecast origin onward and asserts the prediction is bit-for-bit unchanged, for all five models.
- All four calibration folds reproduced the selection-table MAPEs exactly (9.23 / 6.33 / 4.75 / 3.37).

**Two bugs found and fixed along the way**
- `hol_weight_next` was undefined for the final day of the series, putting 24 NaNs into the linear model's design matrix and making its predictions all-NaN.
- That in turn exposed a bug in my own test: `assert_allclose` treats NaN as equal to NaN, so the leakage test had been passing **vacuously** for the linear model. It now asserts finite, non-constant forecasts before the comparison.

**Model selection was done without touching 2020.** A 0.5/0.25/0.25 log-space blend of LightGBM, a linear profile model, and a year-ago naive won on both mean (5.92%) and worst-case (9.23%) MAPE across four prior-January folds.

One caveat I'll flag now rather than bury: `mypy` 1.20.2 crashes with an internal error in this environment regardless of flags, so the code is ruff-clean but **not** type-check-verified. That's an environment fault, not something I can assert away.