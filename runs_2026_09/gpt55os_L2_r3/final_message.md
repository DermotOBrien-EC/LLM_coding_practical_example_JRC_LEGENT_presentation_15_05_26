Done.

Created:

- `forecast_german_load.py`
- `german_load_forecast_jan2020.csv`
- `german_load_forecast_jan2020.png`

Run with:

```bash
.venv/bin/python forecast_german_load.py
```

Accuracy for the 168 hours from `2020-01-01 00:00 UTC` through `2020-01-07 23:00 UTC`:

| Metric | Value |
|---|---:|
| MAE | 2,175.8 MW |
| RMSE | 2,703.8 MW |
| MAPE | 4.14% |
| Bias | +665.9 MW |

Model used: ridge regression with calendar features, German national holiday flags, and recursive lag features at 24, 48, 72, 168, and 336 hours. It trained only on rows before `2020-01-01`.

I selected the ridge regularization using first-week-of-January backtests for 2017, 2018, and 2019. Best validation setting was `alpha=100`.

Assumptions made:

- Forecast target is UTC hours, matching the dataset timestamps.
- “Historical data” means all available rows before `2020-01-01 00:00 UTC`.
- German holidays are national holidays only, not state-specific holidays.
- Since the project note asked for pandas and matplotlib but they were not available globally, I created a local `.venv` and installed the needed packages there.