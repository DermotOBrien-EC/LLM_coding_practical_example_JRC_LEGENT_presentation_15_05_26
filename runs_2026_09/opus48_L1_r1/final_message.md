Done. Forecast and backtest complete.

## What I did

The CSV (OPSD German load, hourly, 2015-01 to 2020-09) already contains the actuals for the target week, so this is a proper held-out backtest: I trained on everything strictly before `2020-01-01 00:00 UTC` and forecast the 168 hours `2020-01-01 00:00`..`2020-01-07 23:00 UTC`, then scored against truth.

**Model:** LightGBM (direct, no recursion) on calendar features (hour, day-of-week, month, cyclical encodings), German federal holiday flags plus explicit `epiphany` and a `xmas_window` flag for the Dec-24→Jan-6 load depression, and three level anchors that are all known over the whole horizon because they point into the past: load 7 days, 14 days, and **364 days** (exactly 52 weeks, so same weekday and hour one year prior) earlier.

## Results (Jan 1-7 2020, 168 h)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Seasonal-naive (last week) | 7238 MW | 8809 MW | 12.78% |
| **LightGBM** | **1602 MW** | **1949 MW** | **2.92%** |

The seasonal-naive baseline is poor precisely because "last week" is Dec 25-31, the holiday trough, an unrepresentative analog for early January. The LightGBM model learns the holiday-vs-normal pattern from five prior Januaries and tracks the actual curve closely (see plot).

Worst days are Jan 3 (4.2%) and Jan 6 / Epiphany (4.3%) — the model slightly over-predicts the daytime peaks mid-week, visible as the red line running above black on the 2nd-3rd and 6th. Best is Jan 1 (New Year holiday, 1.8%), which the holiday features capture well.

## Outputs (in `runs/d2b79c/`)
- `forecast_jan2020.csv` — hourly forecast, actual, and per-hour abs % error
- `forecast_jan2020.png` — forecast vs actual vs baseline
- `forecast_jan2020.py` — reproducible script

Two caveats worth stating: (1) this is a clean backtest, not a live forecast, but it uses only information available before the target week, so the error is a fair estimate; (2) I have not run a Codex review on this per your workflow rules, since it's a single self-contained analysis script — say the word if you want that, or if you'd rather I add a proper walk-forward CV or a second model (e.g. a Darts/statsmodels seasonal model) for an ensemble.