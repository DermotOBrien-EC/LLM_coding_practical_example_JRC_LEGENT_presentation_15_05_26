Created **[forecast.csv](forecast.csv)** with all **168 hourly forecasts**, in MW, plus approximate 90% prediction bands.

**Period assumed:** 1 January 2020 00:00 through 7 January 2020 23:00 **UTC**, matching the input timestamps. German local timestamps are also included.

| Date (UTC) | Average load | Peak hourly load |
|---|---:|---:|
| 1 January | 44.75 GW | 51.54 GW |
| 2 January | 56.08 GW | 64.37 GW |
| 3 January | 57.35 GW | 64.99 GW |
| 4 January | 51.12 GW | 58.07 GW |
| 5 January | 48.12 GW | 55.48 GW |
| 6 January | 57.68 GW | 64.94 GW |
| 7 January | 60.26 GW | 67.91 GW |

**Weekly average:** 53.62 GW. **Total energy:** 9.009 TWh.

The selected holiday-aware LightGBM model achieved **2.16 GW mean absolute error** across January 2017–2019 backtests, versus 7.82 GW for repeating the previous week. No 2020 actual loads were used in fitting or selection.

The prediction bands are approximate, not guaranteed coverage. No weather inputs were used. Eight runtime tests pass, including a full rerun proving that changing future load values leaves the forecasts unchanged.

See **[methodology and reproduction instructions](README.md)** and **[independent review](review.md)**. Strict static typing remains incomplete because the supplied environment lacks pandas/scikit-learn type stubs.