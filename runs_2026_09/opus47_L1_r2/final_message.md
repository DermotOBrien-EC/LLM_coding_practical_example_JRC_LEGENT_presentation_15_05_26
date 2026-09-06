Final skill: MAPE 3.32%, MAE 1.72 GW, RMSE 2.04 GW, bias ~100 MW on 168 hourly steps. That's ballpark of what a decent day-ahead forecast achieves on this horizon.

## Result

Forecast for Germany, 2020-01-01 00:00 through 2020-01-07 23:00 UTC (168 hourly points) is in `forecast_2020_w01.csv` (columns: `utc_timestamp, forecast_mw, actual_mw`).

**Method** (in `forecast.py`, three multiplicative components fit on 2015-2019):
1. `dow_profile[dow, hour]` — normal-week hourly shape, learned from days 8-31 of January so New-Year-week anomalies don't contaminate the "typical" pattern.
2. `calendar_factor[day, hour]` — how each Jan-1..7 date deviates from its own DOW-hour baseline, pooled across years. This captures Jan 1 (holiday), Jan 2-3 (bridge days), Jan 6 (Epiphany), and week-of-January recovery as calendar effects rather than DOW effects.
3. `level_factor` — scalar, `mean(Dec 2019) / mean(all Decembers)`, to re-level to the current operating point (German load has drifted down over 2015-2019).

**Held-out skill vs. actuals in the file:**

| Day (CET) | Forecast mean MW | Actual mean MW | MAPE |
|---|---|---|---|
| Wed 2020-01-01 | 46,245 | 44,060 | 4.96% |
| Thu 2020-01-02 | 54,664 | 53,865 | 3.11% |
| Fri 2020-01-03 | 57,395 | 55,655 | 3.90% |
| Sat 2020-01-04 | 50,198 | 52,640 | 5.01% |
| Sun 2020-01-05 | 46,457 | 47,989 | 3.21% |
| Mon 2020-01-06 | 57,396 | 56,631 | 1.72% |
| Tue 2020-01-07 | 61,889 | 62,598 | 1.45% |
| **All 168 h** | | | **3.32%** |

**Caveats.** No weather covariates (temperature would tighten Jan 4 further, that Saturday was mild). Point forecast only, no uncertainty band. Baseline model; on this dataset an ENTSO-E-style GBM or a temperature-aware SARIMAX typically lands nearer 2% MAPE.