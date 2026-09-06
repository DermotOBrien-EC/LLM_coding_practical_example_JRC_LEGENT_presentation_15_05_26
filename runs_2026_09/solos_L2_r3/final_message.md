## Forecast complete

The script is at `forecast_load.py`. It:

- Reads the OPSD data with pandas and validates hourly continuity and coverage (`forecast_load.py:190`).
- Builds Berlin calendar, holiday, Fourier, trend, rolling, and weekly-lag features (`forecast_load.py:264`, `forecast_load.py:315`).
- Trains a LightGBM model using only observations before each forecast boundary (`forecast_load.py:353`).
- Produces exactly 168 hourly forecasts from 2020-01-01 00:00 UTC through 2020-01-07 23:00 UTC.
- Writes a coherent, locked artifact bundle with SHA-256 verification (`forecast_load.py:626`).

### Accuracy

| Metric | Result |
|---|---:|
| MAPE | **3.244%** |
| MAE | **1,760.6 MW** |
| RMSE | **2,222.3 MW** |
| R² | **0.9343** |
| Mean error | **+865.8 MW** |

The model's typical absolute error was about **1.76 GW**, or **3.24%** of actual load. The positive mean error indicates a modest average overforecast.

A one-week seasonal-naive forecast had **12.781% MAPE**, so the model reduced MAPE by **74.62%** relative to that baseline. Historical validation over the first January weeks of 2017, 2018, and 2019 averaged **3.562% MAPE**.

### Output files

- Forecast and actual values: `german_load_forecast_2020-01-01_to_2020-01-07.csv`
- Accuracy, provenance, and artifact hashes: `german_load_forecast_2020-01-01_to_2020-01-07_metrics.json`
- Actual versus forecast plot with residual panel: `german_load_forecast_2020-01-01_to_2020-01-07.png`

The CSV contains 168 rows, no null cells, and timestamps covering the requested period exactly. The JSON hashes match the published CSV and PNG.

### Run it

From the run directory:

```bash
../../.venv/bin/python forecast_load.py
```

### Verification

- pytest: **25 passed**
- Ruff formatting: passed
- Ruff checks: passed
- strict mypy: passed
- PNG rendered and visually inspected
- CSV timestamps, schema, row count, null count, and artifact digests verified

Assumptions: the requested hours are interpreted in UTC; calendar and German public-holiday features use Europe/Berlin local time; the OPSD ENTSO-E transparency load column is treated as MW. No commit or push was performed.