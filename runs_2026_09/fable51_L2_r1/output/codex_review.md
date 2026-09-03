1. **blocker — forecast_load.py:146, 132, 182**  
   Mechanism: the stipulated origin is `2019-12-31 23:00 UTC`, but training uses `index < start`, where `start` is `2020-01-01 00:00`. Thus the origin-hour target enters training. Moreover, the final test row’s `lag_168`—and its naive baseline—equals the load at the origin. The same leakage occurs in every backtest. Other lags, calendar/holiday features, `trend_days`, and `dropna()` introduce no additional leakage.  
   Fix: define an explicit origin and require both training targets and every lag source to be strictly earlier than it. Under the stated contract, exclude the origin row and replace or remove `lag_168` wherever its source is not strictly pre-origin. Alternatively, if the origin observation is intended to be available, redefine the forecast origin as `2020-01-01 00:00 UTC` throughout.

2. **minor — forecast_load.py:42-45, 131-134**  
   Mechanism: row shifts of 168 hours and 52×168 hours preserve UTC hour, not Berlin local-clock hour. Around DST transitions they can select a local hour one hour earlier or later than the feature comments imply. This affects some training rows, although none of the January target or baseline weeks crosses a DST change. Holiday lookup and local calendar conversion themselves are correct.  
   Fix: either document these explicitly as UTC-hour lags or construct local-clock-aligned lags with defined handling for duplicated and nonexistent DST hours.

The metrics use exactly the requested 168 rows. Model, actual, and both naive series share the same ordered index, so there is no separate positional misalignment. The material claim mismatch is the report/docstring’s “forecast origin” wording versus the strict pre-origin requirement described in finding 1.

## Verdict

Status: needs-attention

- File: forecast_load.py:146
- Category: data
- Severity: critical
- Description: Origin-hour information enters training and the final `lag_168` feature.
- Recommendation: Enforce a strictly pre-origin cutoff for targets and lag sources.