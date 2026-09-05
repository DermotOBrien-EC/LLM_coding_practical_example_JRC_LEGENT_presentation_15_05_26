I found `opsd_de_load.csv`, covering January 2015 through September 2020. Its timestamps are UTC, so I will interpret the requested period as `2020-01-01 00:00` through `2020-01-07 23:00 UTC`.

Plan:

1. Add `test_forecast_load.py` first, covering the exact 168-hour holdout, training cutoff, leakage-safe features, metrics, and output files.
2. Add `forecast_load.py` using pandas, matplotlib, and scikit-learn. It will train a gradient-boosted model on data before 2020 using calendar features and only lags of at least 168 hours.
3. Save `german_load_forecast_2020_week01.csv` and `german_load_forecast_2020_week01.png`. Report MAE, RMSE, MAPE, R², and comparison with a weekly seasonal baseline.
4. Add dependency metadata, create a local uv environment, run pytest, Ruff, mypy, execute the forecast, and inspect the rendered plot.
5. Run an independent review, evaluate its findings, apply valid fixes, then rerun verification.

Approve this plan and I will implement it.