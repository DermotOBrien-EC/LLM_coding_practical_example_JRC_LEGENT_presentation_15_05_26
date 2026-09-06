Done. Script at `runs/25a019/forecast.py`, plot at `runs/25a019/forecast_vs_actual.png`.

**Approach.** Train LightGBM on every hour from 2015-01-01 through 2019-12-31 (43,320 rows) using calendar features (hour, day-of-week, month, day-of-year with sin/cos encodings, weekend and German-holiday flags) plus three weekly lags (168 h, 336 h, 504 h) — all lags safe for a 168-hour-ahead forecast because each target hour's lag lands in the training window.

**Accuracy on 2020-01-01..2020-01-07 (168 h).**

| Metric | Value |
|---|---|
| MAE  | 1,534 MW |
| RMSE | 1,959 MW |
| MAPE | 2.87 % |
| Peak actual   | 70,914 MW |
| Peak forecast | 72,448 MW |

The plot tracks the daily double-peak and weekly shape closely; the largest miss is the New Year's Day evening (forecast dips lower than actual around Wed 01 Jan) — a holiday the model has only five Jan 1s to learn from.

Run it again with `~/dev/energy_forecast_ws/25a019/project/.venv/bin/python3 runs/25a019/forecast.py` (or `uv run` from the project root once `.venv` permissions clear).