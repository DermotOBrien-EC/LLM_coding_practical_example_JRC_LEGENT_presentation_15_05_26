The forecast is complete and validated. Here's the result.

## German hourly load forecast — 1–7 January 2020

**Approach.** German electricity demand is driven almost entirely by *local-time* calendar structure (daily double-peak, weekly cycle, winter level) plus holidays. I trained a gradient-boosted tree model (`HistGradientBoostingRegressor`) on **calendar + holiday features only** — hour, day-of-week, weekend flag, yearly harmonics, and explicit German holiday flags (national, regional/Epiphany, and pre/post-holiday bridge days) — using **only 2015–2019 data**, then forecast the 168 hours of Jan 1–7 2020 and scored against the held-out actuals that are in the file.

**Why a feature model over a naive lag:** the first week of January is dominated by holidays. Jan 1 2020 is a Wednesday public holiday; a "same hour last week" copy would import an ordinary working Wednesday and miss it entirely.

**Backtest (Jan 1–7 of 2017, 2018, 2019; train only on prior data):**

| Model | Mean MAPE | Mean MAE |
|---|---|---|
| **Calendar + holiday GBM** | **≈5.0 %** | 2,745 MW |
| Same-hour-last-year | 7.6 % | 3,887 MW |
| Same-hour-last-week | 13.1 % | 7,792 MW |

**Held-out January 2020 week:** **MAPE 4.4 %**, MAE ≈ 2,380 MW, RMSE ≈ 2,800 MW — vs 10.8 % and 12.7 % for the two naive baselines.

The plot shows the forecast tracking the load shape closely, including the depressed New Year's Day (red band) and Epiphany (purple band).

**Two honest caveats, both visible in the error panel:**
- **~+4 % high bias** (forecast mostly above actual). Tree models can't extrapolate the gentle year-on-year decline in German load past the 2019 training ceiling, so they predict roughly 2019 winter levels. I deliberately did **not** bias-correct using the 2020 actuals — that would be cheating on the hold-out.
- **Jan 3 (Friday) is the worst day (~7.6 %)**, over-forecast because many people took the "bridge day" vacation between the Wednesday holiday and the weekend; the model treats it as a normal Friday.

**Files written** (in `runs/9f7fdb/`):
- `forecast_jan2020.csv` — 168 hourly rows: UTC + local timestamp, forecast MW, actual MW, absolute error
- `forecast_jan2020.png` — the chart above
- `forecast.py`, `plot.py` — reproducible pipeline (backtest + forecast, then plot)

A natural improvement, if you want it: replace the raw `year` feature with an explicit linear trend component (GBM on the detrended residual) so the model can extrapolate the declining level and remove most of that +4 % bias. Say the word and I'll add it.