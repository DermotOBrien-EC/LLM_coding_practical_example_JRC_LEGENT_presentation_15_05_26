# Week-ahead load forecast: 1 to 7 January 2020

Model: LightGBM on calendar, holiday and weekly-lag features, trained on data up
to and including 2019-12-31 23:00 UTC (the last observed hour). All 168 hours
are predicted directly from that data (no recursion).

## Target week (2020)

| Week | MAE (MW) | RMSE (MW) | MAPE (%) | Max abs error (MW) |
|---|---|---|---|---|
| Model | 1,247 | 1,543 | 2.30 | 3,775 |
| Naive: same hour last week | 7,238 | 8,809 | 12.78 | 22,403 |
| Naive: same weekday and hour last year | 5,483 | 7,253 | 10.76 | 22,279 |

## Per-day model error, 2020

| Day (UTC) | MAE (MW) | Mean bias (MW) | Mean actual (MW) |
|---|---|---|---|
| 2020-01-01 | 806 | +99 | 43,997 |
| 2020-01-02 | 1,139 | +997 | 54,077 |
| 2020-01-03 | 1,525 | +1,525 | 55,618 |
| 2020-01-04 | 1,645 | -1,555 | 52,561 |
| 2020-01-05 | 653 | -165 | 48,003 |
| 2020-01-06 | 1,720 | +832 | 56,859 |
| 2020-01-07 | 1,239 | -1,098 | 62,681 |

## Backtests: first week of earlier years, same procedure

| Week | MAE (MW) | RMSE (MW) | MAPE (%) | Max abs error (MW) |
|---|---|---|---|---|
| Model 2017 | 3,344 | 3,911 | 6.10 | 8,306 |
| Naive last week 2017 | 9,521 | 10,274 | 15.97 | 18,041 |
| Naive last year 2017 | 3,819 | 4,538 | 7.03 | 9,667 |
| Model 2018 | 2,767 | 3,220 | 5.43 | 7,582 |
| Naive last week 2018 | 6,896 | 8,229 | 11.92 | 17,488 |
| Naive last year 2018 | 4,629 | 6,514 | 9.42 | 21,964 |
| Model 2019 | 1,076 | 1,299 | 1.98 | 3,271 |
| Naive last week 2019 | 7,054 | 8,913 | 11.71 | 22,007 |
| Naive last year 2019 | 3,247 | 5,462 | 6.44 | 21,514 |
