# German electricity-load forecasting bake-off

A short methods write-up. Six forecasting approaches are fit on the same data,
scored on the same held-out week, and compared on the same metrics.

## 1. Data

The single input is `opsd_de_load.csv` from Open Power System Data, derived from
the ENTSO-E Transparency Platform: hourly German national electricity load in
megawatts. It runs from 2015-01-01 00:00:00 to 2020-09-30 23:00:00 UTC, which is
exactly 50,400 hourly rows. We checked and the series has no
missing values (0 NaN) and no missing hours
(0 gaps between the first and last timestamp), so no imputation
was needed and none was done. All timestamps are UTC; we drop the (constant)
timezone label after loading only because some of the libraries dislike
timezone-aware indices.

## 2. Why these six models

The six form a deliberate complexity gradient, from "reuse last week" to a deep
transformer, so we can see how much each added layer of machinery actually buys.

* **Seasonal-naive** reuses the load 168 hours (one week) earlier. It is the
  anchor: it costs nothing and everything else should beat it.
* **SARIMA** is the classical statistical model: autoregression, moving-average
  errors, and an explicit daily (period-24) season, with weekly structure left
  to emerge through differencing.
* **Prophet** decomposes the series into trend plus daily, weekly and yearly
  shapes and, crucially, knows the German federal holiday calendar.
* **LightGBM** is gradient-boosted trees on hand-built features: calendar flags
  (including a public-holiday indicator), lags at 24 h, 168 h and 8760 h, and
  rolling means and standard deviations.
* **N-BEATS** is a deep fully-connected forecaster that learns structure from
  the raw numbers, reading one week and predicting the next.
* **PatchTST (Transformer)** is a self-attention deep model in the same role.
  The installed darts (0.41.0) does not expose `PatchTSTModel`, so we substitute
  `darts.models.TransformerModel`, a genuine encoder-decoder transformer on the
  univariate series. The substitution is noted here as required.

Everything is univariate on purpose: no weather, prices, or other external data.
The study isolates how much skill lives in the load series' own temporal
structure, not in its weather-dependence.

## 3. Validation strategy

* **Train**: 2015-01-01 to 2019-09-30 (41,616 hours).
* **Validation**: 2019-10-01 to 2019-12-31 (2,208 hours), used only to pick
  hyperparameters / model orders / epoch counts.
* **Test**: 2020-01-01 to 2020-01-07 (168 hours), held out and never touched
  during any fitting or selection.

Each model with knobs is fit on Train, its candidates are scored on Validation,
and the chosen configuration is then refit on Train+Validation combined
(2015-01-01 to 2019-12-31) before forecasting the test week. Refitting on all
pre-test data is standard practice: once the configuration is fixed, throwing
away three months of the most recent data would only hurt. The naive baseline
has nothing to tune, so it skips selection. LightGBM forecasts the test week
recursively (feeding its own predictions back in as lags) so it never sees a
test-window actual as an input feature.

## 4. Results

| Rank | Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2-7 MAPE (%) |
|------|-------|----------|-----------|----------|----------------|------------------|
| 1 | LightGBM | 4.83 | 3079 | 2547 | 4.06 | 4.96 |
| 2 | Prophet | 7.61 | 4877 | 3795 | 13.25 | 6.67 |
| 3 | N-BEATS | 9.68 | 6464 | 5061 | 5.63 | 10.36 |
| 4 | SARIMA | 11.00 | 7592 | 6193 | 11.32 | 10.94 |
| 5 | PatchTST (Transformer) | 12.30 | 7893 | 6346 | 6.19 | 13.32 |
| 6 | Seasonal-naive | 12.78 | 8809 | 7238 | 6.52 | 13.82 |

Jan 1 2020 is a Wednesday and New Year's Day (a German federal holiday).

## 5. Discussion

**LightGBM won**, with a test MAPE of 4.83%
against the seasonal-naive baseline's 12.78%. The full ranking by test MAPE was: LightGBM > Prophet > N-BEATS > SARIMA > PatchTST (Transformer) > Seasonal-naive. All five non-trivial models beat the naive baseline (SARIMA, Prophet, LightGBM, N-BEATS, PatchTST (Transformer)). That the feature-engineered gradient-boosting model came first is what theory would predict for a one-week horizon: the load's own recent lags and the calendar carry almost all the signal, and trees exploit them directly. Its most important features were the one-week and one-day lags together with the public-holiday flag. The holiday is more subtle than 'calendar-blind models fail'. The biggest holiday penalty belongs to Prophet (13.3% MAPE on Jan 1 against 6.7% on the ordinary days Jan 2-7), and it is not because Prophet ignores the holiday but because its holiday-and-seasonal correction overshoots, pulling the New Year's-morning load too far down. LightGBM did the opposite: its public-holiday flag helped it to its best day of the week on Jan 1 (4.1%). The seasonal-naive baseline is a trap for reading the Jan 1 column: it has no calendar at all yet scores a low 6.5% there, purely by luck, because the load it copies from 168 hours earlier falls on Dec 25, itself a holiday with similarly low demand. The deep models split. The transformer landed near the bottom, and even the stronger N-BEATS only reached mid-table (rank 3); both make their largest daily errors on the low-load weekend (both on Sun 05). This is the expected outcome at this budget: with one univariate series of about four and a half years and only about 30 training passes, a large neural network cannot out-learn a tree that was simply handed the right features. They would need far more data, tuning, or covariates to compete. On the probabilistic side, the winner's 80% interval covered 76% of the test hours and its 95% interval covered 98%. Coverage close to nominal means the uncertainty estimate is roughly trustworthy, though the residuals (figure 6) show it also tends to over-predict by one to a few percent. Caveats: the test window is a single week, and a holiday week at that, so the ranking is indicative rather than definitive; a fuller study would test several windows across seasons. All models are univariate, so none can see the weather that drives a real cold snap.

## 6. Recommendation

For a JRC short-term load-forecasting pipeline I would put **LightGBM** into production. It was the most accurate here, it trains and predicts in seconds, it is easy to inspect (feature importances tell an operator what drives a forecast), and it already handles the holiday calendar. Before deploying I would do three things first: (1) add weather features (temperature above all), which are the biggest missing driver of load and were deliberately excluded here; (2) validate on a rolling set of test weeks across all seasons, not one January week, and add proper backtested prediction intervals; and (3) build the recursive multi-step forecast into a monitored job with fallback to the seasonal-naive baseline whenever inputs are late or missing.

## 7. Reproducibility

* Random seed: 42 (numpy, LightGBM, and the darts/torch models). The two
  torch models (N-BEATS and the transformer) train in float32 on the Apple
  MPS backend when available; MPS floating-point reductions are not bit-for-bit
  deterministic, so their numbers can move by a small amount between runs even
  with a fixed seed. All other models are deterministic.
* Total wall-clock runtime for the six fits: 1816 seconds
  (30.3 minutes).
* Hardware / OS: `Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64`
* Every per-model result is cached under `code/_cache/`; delete it to force a
  clean refit. Run `../../.venv/bin/python code/forecast.py` from the run
  directory to reproduce.
