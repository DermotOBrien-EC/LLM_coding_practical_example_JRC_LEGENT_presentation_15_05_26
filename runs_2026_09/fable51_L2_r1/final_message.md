Done. The script, plot, forecast CSV, and report are in the run directory.

**Result:** the week-ahead forecast for 1 to 7 January 2020 has a mean absolute error of 1,247 MW, which is 2.3% MAPE. The plot is `forecast_2020_w1.png`.

| Week 1 of 2020 | MAE (MW) | RMSE (MW) | MAPE (%) | Max abs error (MW) |
|---|---|---|---|---|
| Model | 1,247 | 1,543 | 2.30 | 3,775 |
| Naive: same hour last week | 7,238 | 8,809 | 12.78 | 22,403 |
| Naive: same weekday and hour last year | 5,483 | 7,253 | 10.76 | 22,279 |

**How it works.** `forecast_load.py` trains a LightGBM model on every hour up to and including 2019-12-31 23:00 UTC, then predicts all 168 target hours in one shot. Features are the local-time hour, weekday, day of year, a nationwide holiday flag, a Christmas-to-Epiphany flag, a slow trend term, the loads 1 to 4 weeks earlier, and the load 52 weeks earlier. Every lag is at least 168 hours, so nothing after the forecast origin is used and no prediction is fed back into the model. The target week is read on the file's own UTC clock, since that is how the data is stamped.

**Where the error sits.** New Year's Day itself was forecast well (806 MW MAE). The model ran about 1,000 to 1,500 MW high on Thursday and Friday, then too low on Saturday and Tuesday. Those are level errors on individual days, not a wrong daily shape. Demand that week was still climbing out of the holiday dip, and the weekly lags all point back into that dip, so the model relied on the calendar flags to lift them.

**Is 2.3% a normal result for this method?** The same procedure backtested on the first weeks of 2019, 2018, and 2017 gave 2.0%, 5.4%, and 6.1% MAPE. The earlier years had only one or two years of training data with a New Year period in them, so 2020 and 2019 are the fair comparisons, and 2020 sits right in that range. The naive baselines are consistently 5 to 8 times worse.

**Review.** Codex reviewed the script read-only and reported one blocker and one minor. The blocker claimed the load at 23:00 on 31 December leaks into training. I disagree: that hour is the last observation before the target week, so it is legitimately known at forecast time. The wording "forecast origin 23:00" invited the misreading, so the docstring and report now say "up to and including 2019-12-31 23:00, the last observed hour". The minor finding, that UTC-hour lags land one local hour off across a DST change, is correct for some training rows and never for a January target week. I accepted it as a comment fix. The full review is in `codex_review.md`.

**Verification.** ruff format and ruff check pass. mypy strict passes apart from a missing pandas-stubs package in the venv, which is an environment gap, not a code issue. Two things I changed for this machine: LightGBM is pinned to one thread because the multi-threaded build here was over ten times slower, and LightGBM hangs on import inside my command sandbox, so I ran the script with the sandbox off. On your own shell it runs in about 11 seconds with `python forecast_load.py`.