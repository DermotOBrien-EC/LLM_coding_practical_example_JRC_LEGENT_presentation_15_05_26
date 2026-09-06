Created the **[168-hour forecast CSV](outputs/forecast.csv)**, with hourly load predictions in MW, UTC and German local timestamps, and approximate uncertainty bands.

**Assumed horizon:** 1 January 2020, 00:00 through 7 January, 23:00 **UTC**, matching the input data. Training and model selection use only observations before 2020.

### Forecast summary

| Date (UTC) | Average load (GW) | Peak hourly load (GW) |
|---|---:|---:|
| 1 January | 44.98 | 52.43 |
| 2 January | 56.58 | 66.94 |
| 3 January | 59.39 | 67.82 |
| 4 January | 54.02 | 61.17 |
| 5 January | 49.17 | 57.57 |
| 6 January | 59.61 | 68.23 |
| 7 January | 63.76 | 71.85 |

- **Weekly average:** 55.36 GW
- **Total energy:** 9.300 TWh
- **Peak:** 71.85 GW on 7 January at 16:00 UTC, or 17:00 CET

The selected model blends calendar-aware regression and boosted trees. Its mean absolute error on three earlier New Year weeks was **2.17 GW**, versus **7.82 GW** for repeating the previous week.

**Uncertainty caveat:** the bands are empirical, not guaranteed. The nominal 80% band covered only about 68% of hours in those earlier New Year backtests.

[Forecast chart](outputs/forecast.png) · [Methodology and reproduction instructions](README.md) · [Independent review](outputs/review.md)

All seven tests passed; independent review found no blocking leakage, calendar, or arithmetic errors.