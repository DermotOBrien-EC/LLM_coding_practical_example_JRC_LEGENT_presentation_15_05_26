Done. The script runs, forecasts the 168 hours, plots against actuals, and scores it.

## What it does

`forecast_load.py`:
1. Reads `opsd_de_load.csv`, puts it on a gap-free hourly UTC grid (interpolating the few OPSD gaps so lags line up).
2. Trains a gradient-boosting model (`HistGradientBoostingRegressor`) on **43,488 hours, 2015-01-15 to 2019-12-31**, using only features knowable at the forecast origin: hour, weekday, month, annual position (sin/cos of day-of-year), weekend flag, German public-holiday flag, and lags of 168h and 336h (same hour one and two weeks earlier).
3. Predicts all 168 hours of 2020-01-01 to 2020-01-07 in one shot, then compares to the held-out actuals.

**No look-ahead leakage:** the model never sees any load value from inside the forecast week, and it only uses lags of ≥168h, which for this week map to Dec 18–31 (all known before the week begins). A true ex-ante forecast, not a fit-to-the-answer.

## Accuracy

| | MAPE | MAE | RMSE |
|---|---|---|---|
| **Model (GBM)** | **2.68%** | 1,447 MW | 1,826 MW |
| Naive (same hour, prev. week) | 12.78% | 7,238 MW | 8,809 MW |

**2.68% MAPE is a strong result** for day-of hourly load forecasting (utility operational forecasts typically land in the 1.5–3% range, and those use weather inputs, which this model does not have).

The naive baseline does badly here for a specific reason worth knowing: its "previous week" is Dec 25–31, the Christmas/New Year holiday stretch, when load is far below a normal week. The model beats it by 79% mainly because the holiday flag lets it treat Jan 1 (a public holiday, clearly the low day in the plot) correctly while still forecasting normal weekday levels for Jan 2–7.

## Where it's weakest

Looking at the plot, the forecast slightly **over-predicts the daytime peaks on Jan 2–3** (by ~1–1.5 GW) and is marginally low on the strong Jan 7 evening. Without temperature data it can't catch weather-driven swings, so those peak misses are the expected residual.

## Outputs (in the working directory)
- `forecast_jan2020.png` — the chart above
- `forecast_jan2020.csv` — hourly actual, forecast, and baseline values

One environment note: your Python 3.14 couldn't bootstrap pip (`ensurepip` failed), so I built the dependencies into a 3.13 venv at `~/.cache/energy_forecast_venv`. Run it with:

```
~/.cache/energy_forecast_venv/bin/python forecast_load.py
```