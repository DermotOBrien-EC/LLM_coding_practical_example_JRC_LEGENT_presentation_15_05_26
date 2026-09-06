Created and ran [forecast_load.py](forecast_load.py), producing all **168 hourly forecasts** for January 1–7, 2020.

### Accuracy

| Metric | Forecast | Previous-week baseline |
|---|---:|---:|
| Mean absolute error | **1,566 MW** | 7,238 MW |
| Root mean squared error | **2,051 MW** | 8,809 MW |
| Mean absolute percentage error | **2.80%** | 12.78% |

The forecast reduced MAE by **78.4%** versus repeating the previous week, although Christmas makes that baseline unusually weak. It captured the daily pattern but underpredicted January 7.

### Method and assumptions

- Used **UTC**, matching the source timestamps; calendar features use German local time.
- Selected calendar-based gradient boosting using five pre-2020 validation weeks.
- Fitted the final model on **2016–2019**, reserving 2015 as common lag warm-up during model comparison.
- Forecast the entire week without using any January actuals for training or selection.

### Files

- [Actual-versus-forecast plot](outputs/forecast.png)
- [All 168 predictions and actual values](outputs/forecast.csv)
- [Full metrics and validation results](outputs/metrics.json)
- [Instructions and methodology](README.md)

Run again from this directory:
```bash
../../.venv/bin/python forecast_load.py
```

**Verification:** 17 tests passed, and a repeat run reproduced the numerical outputs byte-for-byte. Independent implementation review was unavailable; details are recorded in [REVIEW.md](REVIEW.md). This is a single-week retrospective backtest, not a guarantee of year-round accuracy.