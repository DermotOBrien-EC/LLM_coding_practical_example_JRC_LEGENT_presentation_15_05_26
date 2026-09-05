Created the forecast here:

`/Users/doob/dev/energy_forecast_ws/c702f9/project/runs/c702f9/forecast_de_load_2020-01-01_to_2020-01-07.csv`

It contains 168 hourly UTC forecasts for German load, first week of January 2020, in MW.

I used `opsd_de_load.csv`, trained only on data through `2019-12-31 23:00 UTC`, with lag, calendar, weekly, annual, and holiday features. Since the file also contains the actual 2020 values, I checked the forecast afterward for context:

- MAE: `2268.3 MW`
- MAPE: `4.20%`

Daily forecast averages:

| Date | Forecast avg MW | Forecast peak MW | Forecast min MW |
|---|---:|---:|---:|
| 2020-01-01 | 45,416.6 | 51,364.6 | 37,723.1 |
| 2020-01-02 | 53,086.9 | 60,204.7 | 43,455.1 |
| 2020-01-03 | 55,724.0 | 63,371.0 | 44,935.7 |
| 2020-01-04 | 50,927.3 | 58,704.6 | 42,612.9 |
| 2020-01-05 | 49,243.1 | 57,436.9 | 41,122.6 |
| 2020-01-06 | 57,836.5 | 67,090.0 | 44,597.5 |
| 2020-01-07 | 56,370.3 | 64,103.1 | 46,607.3 |