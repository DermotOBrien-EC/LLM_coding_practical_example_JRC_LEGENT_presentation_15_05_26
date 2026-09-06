Done. Artifacts are in `out/`.

## What I built

A ridge model on log load, trained on 2015–2019 only, forecasting 168 hours (local 1–7 Jan 2020). Features: day-type × hour profile, month × hour seasonality, a turn-of-year day-offset term for the Christmas/New Year depression, a German holiday calendar with computed Easter and population-weighted regional holidays, and a piecewise-linear temperature response. The file has no weather, so I fetched population-weighted ERA5 temperature for 14 German cities.

Configuration was picked under a rule fixed before scoring (lowest mean MAPE over 18 winter backtest origins in 2017–2019), with 2020 never read during selection.

## Result

| Model | MAPE | MAE |
|---|---|---|
| **Selected, blind to the week** | **5.35%** | 2,873 MW |
| Runner-up (full history, no season window) | 2.95% | 1,608 MW |
| Seasonal naive (lag 168 h) | 12.69% | 7,187 MW |
| 3-year analog | 10.16% | 5,518 MW |

The honest headline is 5.35%. The runner-up was better on this week, but the pre-declared rule chose the other one, and reporting its number as the result would be selection after the fact.

## Why it missed

1 January 2020 fell on a Wednesday, leaving only two working days before the weekend. Germans took that bridge as leave: 3 January reached 89% of the November weekday baseline against 96–99% in the three prior years. No year in 2015–2019 has a post-New-Year working run shorter than three days, so every calendar model over-forecast Thu/Fri.

Adding a working-run-length feature cuts the 2 January error from +5,259 MW to +961 MW (2.25% MAPE overall). I've labelled that row as **not** test evidence in every artifact: I added it after seeing the failure, and it's neutral-to-slightly-worse on 2017–2019, so no forecaster in December 2019 had grounds to include it.

## Two things worth flagging

- **The 80% interval covers 28% of hours.** It's calibrated on three New Year weeks, which cannot represent a regime the training years don't contain. It understates turn-of-year risk badly.
- **The perfect-weather assumption isn't load-bearing.** Degrading temperature with autocorrelated error (0.9 K at day 1 → 2.4 K at day 7, 60 draws) moves MAPE only 5.35% → 5.46%.

## Codex review

Codex couldn't initialize its sandbox in this directory, so it read nothing in three attempts; I got it working by inlining the sources. It confirmed no target-week load reaches the model, and found four real defects, which I fixed: a leap-year wrap in the seasonal window (2016-12-04 measured 30 days from 4 January instead of 31), missing regional holidays (Buß- und Bettag every year, inside the level-correction window; Women's Day Berlin; wrong Reformation Day population shares), bridge flags misread at training-index edges, and silent NaN coverage in the metrics.

Re-running selection on the corrected features changed the winner, which also surfaced an error of mine: an earlier run had labelled `wx_full` as the rule's winner when `wx_w30_a0.3` beat it. The numbers above are from the corrected code.