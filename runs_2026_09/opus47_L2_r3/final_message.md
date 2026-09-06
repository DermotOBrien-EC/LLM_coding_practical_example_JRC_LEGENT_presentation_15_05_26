The forecast tracks the actual load closely across the full week. Both the New Year's Day dip (2020-01-01 was a Wednesday holiday) and the weekend drop (Jan 4-5) come out right; the model slightly over-shoots the mid-week evening peaks by 1000-3000 MW.

**Accuracy on the 168 test hours** (mean actual load 53.4 GW):

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| LightGBM (calendar + 168h/336h/8760h lags) | 1,716 MW | 2,124 MW | **3.13 %** |
| Naive: same hour one week earlier | 7,238 MW | 8,809 MW | 12.78 % |

The naive baseline collapses because "one week earlier" is Christmas week, so it under-predicts every hour after Jan 1; the model does 4x better precisely by seeing that the year-ago lag (Jan 2019) is a much better analogue than the week-ago lag.

**Files written:**
- `runs/85b8bf/forecast_week.py` (the script)
- `runs/85b8bf/forecast.png` (the plot above)

Run with `.venv/bin/python runs/85b8bf/forecast_week.py` from the project root.