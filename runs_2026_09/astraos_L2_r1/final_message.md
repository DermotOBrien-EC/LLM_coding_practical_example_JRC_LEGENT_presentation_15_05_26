Created and ran [forecast_load.py](forecast_load.py), producing all **168 hourly forecasts for January 1–7, 2020**.

### Accuracy

| Method | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| **Forecast model** | **2,592 MW** | **3,191 MW** | **4.79%** |
| Repeat previous week | 7,238 MW | 8,809 MW | 12.78% |
| Repeat same weekday/hour 52 weeks earlier | 5,483 MW | 7,253 MW | 10.76% |

The forecast missed by **2.59 GW per hour on average**. It captured the daily pattern but generally predicted too much demand, with an average upward bias of **2,535 MW**, particularly overpredicting January 2–3.

The model used 2015–2019 history, with its settings selected using five historical validation weeks. **No January 2020 loads entered training, model selection or forecast features.**

### Deliverables

- [Forecast versus actual plot](outputs/forecast_vs_actual.png)
- [All 168 predictions, actuals and errors](outputs/forecast.csv)
- [Detailed accuracy metrics](outputs/metrics.json)
- [Method, assumptions and reproduction instructions](README.md)

Run again with:
```bash
.venv312/bin/python forecast_load.py
```

**Assumptions:** The requested week is interpreted in **UTC**, matching the CSV; calendar features use German local time. These are retrospective results for one holiday week, not guaranteed year-round accuracy. The previous-week baseline is unusually weak because it copies Christmas week.

**Verification:** 17 tests pass, and an independent review found no blockers. Full findings and responses are in [REVIEW.md](REVIEW.md).