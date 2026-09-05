One protocol decision before I write the implementation plan: **should all 168 test predictions be issued together at 2020-01-01 00:00 UTC, with no updates from observed test loads?**

This matters for LightGBM: after the first 24 forecast hours, `lag_24h` reaches into the test week. Rolling features also reach into it. Having those observations in the CSV does not make them available at forecast time.

- **Fixed-origin, week-ahead:** recursively feed predictions into the unavailable lags and rolling features. All six models face the same 168-hour forecasting task.
- **Rolling, updated forecasts:** reveal actual loads as time advances and update forecasts. This is a different task and requires the same update schedule for all six models.

Your stated horizon suggests the first interpretation. Please confirm that is intended.