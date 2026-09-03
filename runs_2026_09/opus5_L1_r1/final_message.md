Done. Forecast for 1-7 January 2020, made from an origin of 1 Jan 00:00 local using only data through 31 Dec 2019.

**Accuracy against the actuals in the file: MAPE 3.07%, MAE 1,727 MW, RMSE 2,249 MW, weekly peak +2.44%.**

| day | forecast mean | actual mean | err | forecast peak | actual peak | MAPE |
|---|---|---|---|---|---|---|
| Wed 1 Jan | 44,194 | 43,988 | +0.5% | 51,769 | 51,755 | 1.5% |
| Thu 2 Jan | 56,345 | 53,865 | +4.6% | 66,648 | 62,923 | 4.6% |
| Fri 3 Jan | 58,951 | 55,655 | +5.9% | 67,489 | 63,230 | 5.7% |
| Sat 4 Jan | 51,528 | 52,640 | -2.1% | 58,069 | 60,436 | 2.9% |
| Sun 5 Jan | 47,315 | 47,989 | -1.4% | 54,855 | 56,412 | 1.8% |
| Mon 6 Jan | 58,107 | 56,631 | +2.6% | 67,227 | 66,215 | 2.8% |
| Tue 7 Jan | 63,322 | 62,598 | +1.2% | 72,645 | 70,914 | 2.2% |

**Model.** Ridge on log load over calendar features only, so all 168 hours come from one origin with no recursive lags. The features that earn their place are day-type × hour, annual Fourier terms interacted with time of day, and a turn-of-year day-offset × time-of-day block that carries the New Year trough and the industrial recovery through the week. Epiphany enters as a population-weighted regional holiday (BY/BW/ST, 0.317). A level correction uses the mean log residual over 3 Nov to 17 Dec 2019, deliberately ending before Christmas so the anchor is not holiday-contaminated.

α=30 and the level correction were selected on 449 rolling winter origins with every Jan 1-8 window excluded, so the January numbers above are not part of selection. Mean MAPE over the Jan 1-7 windows of 2017-2020: this model 3.45%, ridge without the level correction 3.76%, LightGBM on the same features 4.57%, prior-years-same-date rescaled 9.44%, seasonal naive 13.03%.

**Two things worth flagging.**

The model has no weather input, and temperature is what the residual is made of. The worst backtest year is Jan 2017 (5.33% MAPE) purely because it was cold. The 80% intervals in the CSV are empirical, from 391 past winter origins, and they achieved 76.2% coverage here.

The visible failure is Thu 2 and Fri 3 January, over-forecast by 4.6-5.9%. The calendar says "holiday over, back to work"; German industry actually came back more slowly. That is the single largest structural error in the week and it is not fixable without either weather or an industrial-activity proxy.

Files in the run directory: `forecast_2020_jan_w1.csv` (168 rows, local + UTC timestamps, forecast, p10/p90, actual), `forecast_2020_jan_w1.png`, `README.md`, and the scripts. One implementation note: backtests refit via incrementally accumulated Gram matrices over nested training prefixes (`fastridge.py`), verified identical to the direct fit to 4e-13, which took the 449-origin sweep from ~8 minutes to 24 seconds.