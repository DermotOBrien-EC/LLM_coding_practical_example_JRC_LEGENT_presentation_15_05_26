## Result

The 168-hour forecast achieved:

| Metric | Result |
|---|---:|
| MAE | **2,233 MW** |
| RMSE | **2,714 MW** |
| MAPE | **4.026%** |
| sMAPE | **4.020%** |
| R² | **0.902** |
| Mean bias | **+209 MW** |

In practical terms, the hourly forecast was off by about **4% on average** and captured roughly **90% of the observed hourly variation**. The small positive bias means it slightly overpredicted overall.

Historical New Year backtests for 2017–2019 produced a mean MAPE of **4.183%**, close to the 2020 result, so the final accuracy is consistent with prior unseen weeks.

## Deliverables

- Script: [`forecast_load.py`](forecast_load.py)
- Tests: [`test_forecast_load.py`](test_forecast_load.py)
- Forecast and actual values: [`german_load_forecast_2020_week1.csv`](german_load_forecast_2020_week1.csv)
- Plot: [`german_load_forecast_2020_week1.png`](german_load_forecast_2020_week1.png)
- Metrics and backtests: [`german_load_forecast_2020_week1_metrics.json`](german_load_forecast_2020_week1_metrics.json)

The script uses LightGBM with German local-time calendar, seasonal, trend, and federal holiday features. It trains strictly on 2018-01-01 through 2019-12-31, then forecasts the requested week without using target-period load values (`forecast_load.py:214`, `forecast_load.py:232`, `forecast_load.py:247`).

Run it with:

```bash
/Users/doob/dev/energy_forecast_ws/dbc151/project/.venv/bin/python forecast_load.py
```

## Verification

- 10 pytest tests passed.
- `ruff format` and `ruff check` passed.
- `mypy --strict` passed for both source and tests.
- Independently recomputed metrics match the report.
- Confirmed exactly 168 continuous UTC timestamps.
- Confirmed the training period ends before the first target hour.
- CSV and PNG SHA-256 digests match those recorded in the metrics report.
- A repeated run produced byte-identical outputs.

The independent review found and prompted fixes for timestamp misalignment, malformed forecast indexes that could cause leakage, incomplete backtest history, constant-series R² handling, UTC normalization, and output provenance. A final reread found no remaining correctness or leakage defects. The required Fable peer lane was unavailable because its gateway configuration could not be read, so the review was a separate same-model clean reread instead.

Assumptions: the requested dates refer to the dataset’s UTC timestamps, specifically 2020-01-01 00:00 UTC through 2020-01-07 23:00 UTC. No weather data was available, and the holiday feature covers federal rather than state-specific German holidays.