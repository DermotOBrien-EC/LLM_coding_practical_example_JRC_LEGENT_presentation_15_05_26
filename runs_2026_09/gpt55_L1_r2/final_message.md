Created `forecast_jan2020_week1.csv` in:

`/Users/doob/dev/energy_forecast_ws/760623/project/runs/760623/forecast_jan2020_week1.csv`

Method: trained only on data before `2020-01-01`, using a regularized log-load regression with trend, hour, weekday, weekday-hour, annual seasonality, and first-week-of-January effects.

Backtest on first week of January 2017-2019:

```text
MAPE: 4.75%
RMSE: 3,325 MW
```

For the held-out actual first week of January 2020:

```text
MAPE: 4.58%
RMSE: 3,097 MW
```

Daily summary:

```text
date        forecast avg MW  actual avg MW  MAPE
2020-01-01  47,930.8         43,997.2       9.00%
2020-01-02  57,379.2         54,077.5       5.74%
2020-01-03  59,912.5         55,618.5       7.48%
2020-01-04  52,294.3         52,561.3       1.80%
2020-01-05  48,193.0         48,003.4       1.27%
2020-01-06  58,917.9         56,859.3       3.49%
2020-01-07  64,496.1         62,680.8       3.25%
```

First rows of the forecast:

```csv
utc_timestamp,forecast_de_load_mw
2020-01-01 00:00:00+00:00,44630.8
2020-01-01 01:00:00+00:00,43297.2
2020-01-01 02:00:00+00:00,42184.2
2020-01-01 03:00:00+00:00,42008.2
2020-01-01 04:00:00+00:00,42106.9
2020-01-01 05:00:00+00:00,42248.7
2020-01-01 06:00:00+00:00,43159.5
2020-01-01 07:00:00+00:00,43652.4
```