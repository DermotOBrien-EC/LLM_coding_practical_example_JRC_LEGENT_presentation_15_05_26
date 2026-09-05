Forecast generated for **2020-01-01 00:00 through 2020-01-07 23:00 UTC**, 168 hourly observations.

**Files**

- [Hourly forecast CSV](forecast_jan_2020.csv)
- [Reproducible forecasting script](build_forecast.py)
- [Backtest diagnostics](forecast_diagnostics.json)

| UTC date | Mean forecast |
|---|---:|
| 2020-01-01 | 45,923 MW |
| 2020-01-02 | 56,812 MW |
| 2020-01-03 | 58,949 MW |
| 2020-01-04 | 51,977 MW |
| 2020-01-05 | 48,161 MW |
| 2020-01-06 | 58,826 MW |
| 2020-01-07 | 62,174 MW |

The forecast uses an ensemble of LightGBM models with hourly, weekday, holiday, annual-seasonality, recent-level, weekly-lag, and prior-year features. Rolling-origin backtests on the equivalent weeks of 2017–2019 produced **5.19% MAPE**.

The supplied data contains the January 2020 actuals. They were excluded from model fitting and feature leakage, then used only to evaluate the finished retrospective forecast: **1,889 MW MAE, 2,375 MW RMSE, and 3.56% MAPE**.