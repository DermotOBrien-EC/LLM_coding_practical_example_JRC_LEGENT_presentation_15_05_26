Read-only review of a time-series forecasting analysis. Do not modify files.

Context: forecast German hourly electricity load for 1-7 January 2020 (168 hours,
Europe/Berlin local days) from opsd_de_load.csv, which contains hourly load from
2015-01-01 to 2020-09-30. The actuals for the target week are present in the file
and are used only to score the forecast afterwards.

Files to read (all under this directory):
  work/loadfc.py       features, holiday calendar, ridge/GBM models, metrics
  work/backtest.py     walk-forward backtest harness
  work/deliver.py      final artifact generation for the 2020 target week
  work/tune3.py        the model-selection sweep over winter origins

Please check, in order of importance:

1. Look-ahead / leakage. The intended contract is that nothing at or after the
   forecast origin (2019-12-31 23:00 UTC) influences the fitted model or any
   post-fit adjustment. Trace every place training data is sliced, every lag,
   the `_level_correction` window, the `sample_weights` helper, and the
   prediction-interval calibration. Report the exact expression and line if any
   path admits target-period load. Note separately that temperature over the
   target week IS used deliberately (a perfect-weather-forecast assumption);
   that is disclosed, not a defect, but confirm it is the ONLY such path and
   that no LOAD value from the target week reaches the model.

2. Timezone and DST correctness. Data is UTC; features are built in
   Europe/Berlin. The target is defined as 168 local hours from local midnight.
   Check hour-of-day, day-of-week, day-of-year, the turn-of-year offset, and the
   year-ago analog index for off-by-one or DST errors.

3. The German holiday calendar in `german_holidays` and `easter_sunday`:
   correctness of the Easter computation and of the fixed and movable dates.

4. `work_run`, `build_features` bridge flags, and `turn_of_year_offset`: do they
   compute what the docstrings claim, including at the edges of a
   non-contiguous (season-windowed) training index?

5. Metric definitions in `metrics` (MAPE, MAE, RMSE, bias, peak error) and
   whether the reported numbers mean what a reader would assume.

For each finding give: file:line, what is wrong, the concrete input or state that
triggers it, and the impact on the reported numbers. Distinguish real defects
from style preferences. If a section is correct, say so briefly rather than
inventing issues.
