**No target-week load enters any numerical fit, level correction, or interval calibration in the supplied code.** However, the explicitly post-hoc feature choice is target-informed, so the broader frozen-origin contract does not hold for that variant.

This was a static review of the inlined files. No files were modified or tests run. Numerical effects cannot be quantified without rerunning; selection and preregistration chronology cannot be established from comments alone.

The leakage trace is:

| Path | Assessment |
|---|---|
| Training slices | `s.loc[: origin - pd.Timedelta(hours=1)]` in `backtest.py:60` and `tune3.py:27`, and the equivalent expression in `deliver.py:38`, stop at **2019-12-31 22:00 UTC** for the final forecast. Seasonal filtering only removes rows. |
| Weekly lag | `s.reindex(tgt - pd.Timedelta(hours=168))`, `backtest.py:33`, accesses **24 December 23:00 through 31 December 22:00 UTC**. All precede the origin. |
| Year analogs | The local-calendar `DateOffset(years=k)` expressions at `backtest.py:40–41,47–49` access previous years only. |
| Level adjustments | `_level_correction`, `loadfc.py:238–245`, ends at `origin - 1 hour`. Its callers supply training-only actuals and fitted values. The analog adjustment at `backtest.py:43–51` likewise uses only pre-origin loads. |
| Sample weights | `(index[-1] - index).days / 365.25`, `loadfc.py:251`, depends only on training timestamps. Sorting is preserved by the shown slices. All supplied model calls leave the half-life unset, so weights are actually `None`. |
| Interval calibration | `deliver.py:55–69,103–108` uses residuals from 2017–2019 weeks. Each calibration forecast has its own pre-origin training cutoff. No 2020 load enters the bounds. |
| Weather | `loadfc.py:123–133` uses trailing EWM/rolling operations. Future temperatures cannot change earlier training features. Target temperatures deliberately affect target predictions; sensitivity perturbations do likewise. Target actual loads in `deliver.py:76,86` are used only to score those predictions. |

Thus, **target temperature is the only future observed numerical input to predictions in the shown implementation**. The post-hoc design decision is a separate information path, described below.

Timezone handling is correct for the requested week. `backtest.py:26–29` and `deliver.py:26–28` produce **2019-12-31 23:00 through 2020-01-07 22:00 UTC**, representing seven complete Berlin days. Local hour, weekday and day-of-year at `loadfc.py:141–161` are correct. January has no DST transition, so elapsed 24-hour horizon buckets also match local days. The year-ago analog preserves local date and hour correctly for these January targets.

`turn_of_year_offset`, `loadfc.py:100–108`, is correct: 21 December maps to −11, 31 December to −1, 1 January to 0, and 11 January to 10. `work_run`, lines 74–97, correctly reconstructs a contiguous calendar across seasonal gaps; its ten-day padding exceeds the maximum working run between weekends. For 2020, 2–3 January form a two-day run. Regional Epiphany on 6 January does not interrupt a run, consistently with the documented **national-holiday** rule.

The Easter algorithm is correctly transcribed. The included movable offsets and nationwide fixed dates are correct; for example, Easter 2020 is 12 April. [PTB’s Easter calculation and date table](https://www.ptb.de/cms/ptb/fachabteilungen/abt4/fb-44/ag-441/darstellung-der-gesetzlichen-zeit/wann-ist-ostern.html)

For complete, positive load data, `metrics` correctly computes hourly MAPE, MAE, RMSE and signed bias, with **positive bias meaning overprediction**. `peak_MAPE_%` compares each local day’s predicted maximum against its actual maximum; it does **not** measure peak timing or error at the actual peak hour. Equal maxima occurring at different hours therefore produce zero peak error. That is a valid metric, but should be described explicitly.

The interval arithmetic is also correct: residuals are `(actual − prediction) / prediction`, so multiplying by `1 + quantile` reconstructs the corresponding bounds. “80%” remains nominal: these are pooled empirical residual intervals, with correlated observations and calibration weeks also used for configuration selection, rather than independently validated 80% coverage.

## Verdict

Status: needs-attention

**1. The post-hoc variant does not satisfy the frozen-origin contract.**

- File: `work/deliver.py:20–21,98,145–146`
- Category: data
- Severity: critical
- Description: The comment explicitly says `run_feat=True` was motivated by the 2020 failure. Consequently, the target outcome influences model design, although no target load row enters fitting. Including this variant under “Accuracy on the held-out week” does not make its scores independent test evidence. This affects interpretation of every post-hoc metric; it does not establish leakage into the primary forecast.
- Recommendation: Treat these scores as exploratory and use an untouched period for confirmatory evaluation.

**2. Seasonal distance incorrectly wraps leap-year day numbers on a 365-day circle.**

- File: `work/deliver.py:34,40`; `work/tune3.py:23,30`
- Category: correctness
- Severity: important
- Description: `np.minimum(r, 365 - r)` mishandles day 366 and shifts leap-year seasonal boundaries. For the final January forecast, the centre is day 4. **2016-12-04**, day 339, gets distance `365 - abs(339 - 4) = 30`, although it is 31 calendar days before 4 January. It is wrongly admitted to the 30-day window. This changes the preregistered model’s training rows and seasonal candidates’ tuning scores; the full-history primary model is not directly filtered by this expression.
- Recommendation: Calculate seasonal distance on a consistent month/day calendar with an explicit leap-day policy.

**3. The holiday calendar omits regional holidays present in the training period.**

- File: `work/loadfc.py:39–56,64`
- Category: data
- Severity: important
- Description: Missing dates receive share zero through `table.get(d, 0.0)`. Examples include **2019-03-08 in Berlin**, **2019-09-20 in Thüringen**, and **Buß- und Bettag in Saxony**, including 2019-11-20. These omissions encode affected training observations as ordinary days and can change fitted calendar effects and forecasts. The November omission also lies inside the primary level-correction window. These are missing events, distinct from the expressly approximate population weights. [Berlin’s 2019 announcement](https://www.berlin.de/ba-marzahn-hellersdorf/aktuelles/pressemitteilungen/2019/pressemitteilung.789824.php), [Thüringen’s 2019 announcement](https://www.gruene-thl.de/familie/weltkindertag-ist-feiertag-thueringen), [Saxon holiday law](https://www.revosax.sachsen.de/vorschrift/3997-SaechsSFG)
- Recommendation: Add historically applicable regional dates and appropriate approximate shares. Also correct line 54’s geography: four states joined five existing states in 2018; that comment correction alone has no numerical effect. [Hannover’s historical account](https://www.hannover.de/Service/Presse-Medien/Hannover.de/Aktuelles/Gut-zu-wissen/Glossar-Feste-und-Feiertage/Warum-und-wie-Hannover-den-Reformationstag-feiert)

**4. Bridge flags depend on the supplied index’s boundaries and gaps.**

- File: `work/loadfc.py:169–171`
- Category: edge-case
- Severity: important
- Description: Both shifted lookups use `.reindex(index_utc).fillna(False)`, treating absent neighbouring dates as working days. At the **2019 New Year backtest origin**, training ends on Monday 2018-12-31. That day lies between Sunday and New Year’s Day, but the next-day rows are absent, so all 24 hours incorrectly receive `bridge=0`. The same problem occurs at seasonal-window boundaries. It changes training features and can change backtest scores, model selection and calibrated bounds. `work_run` itself avoids this defect.
- Recommendation: Determine neighbouring local dates’ off-day status from the calendar independently of which rows are retained.

**5. The 2016 analog’s missing first hour produces inconsistent metric coverage.**

- File: `work/backtest.py:40–42`; `work/loadfc.py:307–316`
- Category: edge-case
- Severity: important
- Description: For local **2016-01-01 00:00**, the nearest analog requests **2014-12-31 23:00 UTC**, before the stated CSV start; older analogs are also unavailable. `np.nanmean` therefore yields NaN. Hourly MAPE, MAE, RMSE and bias become NaN, while daily `.max()` skips the missing prediction and can still produce a finite peak MAPE. The metrics consequently describe different observation coverage. This affects the 2016 backtest, not the complete 2020 analog or the explicitly restricted 2017–2019 average.
- Recommendation: Require complete scoring coverage, or explicitly report a common valid-observation mask and its sample count.
