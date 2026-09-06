# German hourly electricity load: forecast for 1-7 January 2020

## Task and protocol

Forecast hourly German system load for the first week of January 2020, defined as
168 hours of local time, 2020-01-01 00:00 to 2020-01-07 23:00 Europe/Berlin
(2019-12-31 23:00 to 2020-01-07 22:00 UTC).

The model sees load only up to 2019-12-31 22:00 UTC. Actuals for the target week
exist in the source file and are used solely to score the forecast afterwards.

Model configuration was chosen on backtest origins in 2017-2019 under a rule
declared before scoring: **lowest mean MAPE across all 18 winter origins** (three
New Year weeks plus fifteen ordinary winter weeks). The 2020 week was not read
during selection. See `selection_summary.csv`.

## Data

- `opsd_de_load.csv` (given): OPSD / ENTSO-E transparency German load,
  2015-01-01 to 2020-09-30, hourly UTC. 50,400 rows, no gaps, no missing values,
  no duplicate timestamps.
- Temperature (fetched): population-weighted 2m temperature over 14 German cities
  from the Open-Meteo ERA5 archive, cached in `../work/temp_de.csv`. Temperature
  over the target week is treated as known, a perfect-weather-forecast
  assumption; the sensitivity run below quantifies what that is worth.

## Model

Ridge regression on log load with:

- day-type x hour-of-day profile, a national holiday taking its own day type
- month x hour-of-day terms plus annual Fourier harmonics
- turn-of-year day offset (-11 to +10 days from 1 January) interacted with
  6-hour blocks, carrying the Christmas-New Year depression and recovery
- German holiday calendar with computed Easter, population-weighted regional
  holidays, and bridge-day flags
- piecewise-linear temperature response (hinges from -5 to 25 C) plus smoothed
  temperature at 24 h and 72 h half-lives, interacted with time of day
- training restricted to within 30 calendar days of the target date across all
  years, which is what the selection rule picked

## Results on the held-out week

| Model | MAPE | MAE | RMSE | Bias | Weekly energy |
|---|---|---|---|---|---|
| **Selected (blind to this week)** | **5.35%** | 2,873 MW | 3,456 MW | +2,829 MW | 9,436 GWh |
| Runner-up: full history, no season window | 2.95% | 1,608 MW | 2,081 MW | +445 MW | 9,036 GWh |
| Post hoc: runner-up plus working-run feature | 2.25% | 1,210 MW | 1,456 MW | -103 MW | 8,944 GWh |
| Seasonal naive (load 168 h earlier) | 12.69% | 7,187 MW | 8,773 MW | -6,857 MW | 7,809 GWh |
| Analog: mean of previous 3 years, rescaled | 10.16% | 5,518 MW | 6,740 MW | +1,681 MW | 9,243 GWh |

Actual weekly energy 8,961 GWh; actual peak 70,914 MW on Tue 7 Jan 17:00.

The post-hoc row is **not** independent test evidence: that feature was added
after inspecting this week's errors. It is reported to explain the failure, not
to claim accuracy.

## Why the selected model missed

1 January 2020 fell on a Wednesday, leaving only two working days (Thu 2 and
Fri 3 January) between the holiday and the weekend. Germans took that bridge as
leave, and load on 3 January reached only 89% of the November weekday baseline
against 96-99% in the three preceding years. No year in 2015-2019 has a
post-New-Year working run shorter than three days, so the pattern is absent from
the training data and every calendar model over-forecast those two days.

Adding an explicit working-run-length feature cuts the 2 January daily error
from +5,259 MW to +961 MW. On the 2017-2019 backtests that feature is neutral to
marginally harmful (3.87% against 3.83% mean MAPE on the New Year origins,
2.093% against 2.078% across all 18), because those years contain no short run
for it to discriminate. It could not have been validated in advance on this
dataset, and a forecaster in December 2019 had no evidence for adding it.

## Known limitations

- The 80% prediction interval covers only 28% of hours. It is calibrated on
  three New Year weeks, which cannot express a regime the training years do not
  contain. Treat the interval as an underestimate of turn-of-year risk.
- Selection across 18 correlated winter origins separated the top candidates by
  less than 0.1 pp of MAPE, which is inside the noise band. The rule picked the
  season-windowed model; the full-history runner-up was better on this week.
- Weather enters as observed reanalysis, not a forecast. Degrading temperature
  with autocorrelated error growing from about 0.9 K at day 1 to 2.4 K at day 7
  moves MAPE only from 5.35% to 5.46% on average over 60 draws, so the perfect
  weather assumption is not what drives the result.

## Files

- `forecast_2020_week1.csv` - 168 hourly rows: forecast, 80% interval, both
  alternative forecasts, actual, temperature, error.
- `forecast_2020_week1.png` / `_dark.png` - forecast against actual, and errors.
- `selection_summary.csv`, `selection_2017_2019.csv` - selection evidence.
- `summary.json` - scored results for every model, machine readable.
- `../work/codex_review.md` - independent code review (leakage, timezone,
  holiday calendar), and the fixes it prompted.

Reproduce with `work/select.py`, then `work/deliver.py`, then `work/plot.py`.
