Done. Trained a LightGBM model on 2015 through 2019 hourly load (43,824 rows), forecast the 168 hours 2020-01-01 00:00 through 2020-01-07 23:00 UTC, and evaluated against the actuals present in the same file.

Features: hour of day, day of week, month, day of year (with Fourier annual terms), German public holidays (with prev/next-day flags), and a Dec 24 through Jan 6 bridge-period flag. `holidays` and `lightgbm` are already in the project's `pyproject.toml`, so I ran through the project's uv venv.

Week-level metrics:

- MAE 1922 MW
- RMSE 2465 MW
- MAPE 3.60 %
- Peak forecast 72,009 MW at 2020-01-07 17:00 UTC vs actual peak 70,914 MW same day 16:00 UTC (+1.5 %, one hour late)

Daily MAPE (Europe/Berlin dates, last row is the trailing hour of Jan 7 that falls into Jan 8 local):

| Date       | Actual mean | Forecast mean | MAPE |
| ---------- | ----------: | ------------: | ---: |
| 2020-01-01 |      44,060 |        46,187 |   5% |
| 2020-01-02 |      53,865 |        56,905 |   6% |
| 2020-01-03 |      55,655 |        59,333 |   7% |
| 2020-01-04 |      52,640 |        50,980 |   4% |
| 2020-01-05 |      47,989 |        48,082 |   1% |
| 2020-01-06 |      56,631 |        56,277 |   2% |
| 2020-01-07 |      62,598 |        62,645 |   1% |

Weekend and Epiphany Monday come in at 1 to 2 % MAPE. The remaining error is concentrated on Thu 2020-01-02 and Fri 2020-01-03 (return-to-work days after New Year), where the model still over-forecasts by roughly 3 GW. The bridge flag helps but does not fully capture the post-holiday ramp-in — a lag feature (24h and 168h load) would be the next step if you want that below 2 %.

Artifacts:
- `runs/37e6d0/forecast.py` — the model and eval driver
- `runs/37e6d0/forecast_2020w1.csv` — 168-hour forecast with actuals side by side