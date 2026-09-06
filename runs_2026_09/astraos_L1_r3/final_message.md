Saved **168 hourly forecasts** in [forecast.csv](forecast.csv), covering **1–7 January 2020, UTC**, with load in **MW**.

### Daily average forecast

| Date (UTC) | Average load (GW) |
|---|---:|
| 1 January | 45.62 |
| 2 January | 57.18 |
| 3 January | 60.29 |
| 4 January | 53.69 |
| 5 January | 50.00 |
| 6 January | 59.69 |
| 7 January | 64.09 |

- **Weekly average:** 55.79 GW
- **Highest hourly load:** 72.66 GW, 7 January at 16:00 UTC
- **Total energy:** 9.373 TWh

The forecast blends two holiday-aware LightGBM models, using only observations before 2020. On the first weeks of January 2017–2019, the blend achieved **2.47 GW mean absolute error** and **4.64% mean absolute percentage error**. These are model-selection backtests, not guaranteed accuracy for 2020.

**Assumptions:** UTC follows the supplied file; “first week” means January 1–7 inclusive. No weather forecasts were supplied or used.

[Methods and backtest report](forecast_report.md) · [Reproducible code](forecast_load.py) · [Verification and independent review](verification.md)