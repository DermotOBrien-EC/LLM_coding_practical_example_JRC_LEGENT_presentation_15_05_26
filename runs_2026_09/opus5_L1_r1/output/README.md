# German hourly load forecast, 1-7 January 2020

Forecast of DE hourly electricity load for the first week of January 2020, made as a
single 168-hour-ahead forecast from an origin of 2020-01-01 00:00 Europe/Berlin.
Training data is the OPSD ENTSO-E transparency series, 2015-01-01 through
2019-12-31 23:00 local. No 2020 observation enters the model.

## Model

Ridge regression on log load over calendar features only, so the whole week is
predictable from an origin a week earlier with no recursive lag propagation:

- day type (Mon-Sun, plus national public holiday as an 8th type) x hour of day
- annual Fourier terms (k=1..3), interacted with 4-hour time-of-day blocks
- turn-of-year day offset (Dec 18 - Jan 12) x time-of-day block, which is what
  carries the New Year trough and the gradual industrial recovery through the week
- Epiphany as a population-weighted regional holiday (BY, BW, ST = 0.317)
- pre/post-holiday bridge-day dummies, linear trend
- a level correction: the mean log residual over 3 Nov - 17 Dec 2019, a recent
  window that ends before the Christmas period so the anchor is not holiday-contaminated

Ridge strength (alpha = 30) and the level correction were selected on 449 rolling
winter origins with all Jan 1-8 windows excluded, so the reported January results
are not part of model selection.

## Results (1-7 Jan 2020, 168h ahead)

MAPE 3.07% · MAE 1,727 MW · RMSE 2,249 MW · peak error +2.44% · 80% interval coverage 76.2%

Model comparison, mean MAPE over the Jan 1-7 windows of 2017-2020:

| model | mean MAPE |
|---|---|
| ridge (this model) | 3.45% |
| ridge, no level correction | 3.76% |
| ridge + LightGBM ensemble | 3.79% |
| LightGBM on the same features | 4.57% |
| mean of prior years, same date-hour, rescaled | 9.44% |
| seasonal naive (lag 168h) | 13.03% |

## Files

| file | purpose |
|---|---|
| `loadfc.py` | data loading, calendar features, design matrix, reference ridge, metrics |
| `fastridge.py` | incremental ridge over nested training prefixes (backtests refit in O(1) per origin) |
| `forecast.py` | the deliverable: fits, forecasts, empirical intervals, writes the CSV |
| `tune.py` | ridge-strength sweep on held-out winter origins |
| `backtest.py` | model comparison against baselines and LightGBM |
| `models.py` | baselines and the LightGBM/reference-ridge forecasters used by `backtest.py` |
| `plot.py` | chart |
| `explore.py` | data coverage and gap checks |

Outputs: `forecast_2020_jan_w1.csv` (168 rows: local and UTC timestamps, point
forecast, p10/p90, actual), `forecast_2020_jan_w1.png`, `tuning_results.csv`.

Run with the project venv: `../../.venv/bin/python forecast.py`

## Limitation

The model has no weather input. Temperature is the dominant driver of the residual,
and the largest errors in the backtest are weather years the calendar cannot see
(Jan 2017 was unusually cold: 5.33% MAPE against 2.2-3.1% in the other years).
For 1-7 Jan 2020 the model over-forecasts Thu 2 and Fri 3 January by 4.6-5.9%,
expecting a faster post-holiday industrial recovery than actually occurred.
