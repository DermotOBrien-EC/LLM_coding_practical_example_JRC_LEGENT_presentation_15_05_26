Created and ran [forecast_load.py](forecast_load.py). It forecasts all **168 hours from 2020-01-01 00:00 through 2020-01-07 23:00 UTC**, matching the supplied dataset’s timezone.

### Accuracy

| Method | Mean absolute error | RMSE | Mean absolute percentage error |
|---|---:|---:|---:|
| **Selected model** | **2,810 MW** | **3,448 MW** | **5.08%** |
| Repeat previous week | 7,238 MW | 8,809 MW | 12.78% |
| Repeat 52 weeks earlier | 5,483 MW | 7,253 MW | 10.76% |
| Repeat 365 days earlier | 4,900 MW | 5,911 MW | 9.18% |

The model reduced average absolute error by **42.6% versus the best baseline**. It captured the daily pattern but generally overpredicted demand, with an average bias of **+2,720 MW**.

The selected model uses gradient-boosted trees with hourly, weekday, seasonal, and German holiday features. Model selection used six pre-2020 validation weeks; **no target-week or later load values were used for training or selection**. These results describe one holiday-heavy week, not guaranteed future performance.

### Deliverables

- [Actual versus forecast plot](outputs/forecast.png)
- [All 168 predictions, actuals, and errors](outputs/forecast.csv)
- [Detailed report](outputs/report.md)
- [Exact metrics and run details](outputs/metrics.json)
- [Setup and usage instructions](README.md)

Rerun from this directory:
```bash
../../.venv/bin/python forecast_load.py
```

**Verification:** 12 tests passed, lint and type checks passed, and a second run produced identical forecast values. Independent peer review was unavailable because its gateway configuration could not be read.