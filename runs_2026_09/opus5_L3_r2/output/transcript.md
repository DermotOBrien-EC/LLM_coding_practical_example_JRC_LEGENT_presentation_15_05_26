# Forecasting bake-off: German hourly electricity load

Six models, one held-out week, one set of rules. Everything below is produced
by `code/forecast.py`. Numbers in the results tables are injected from the
computed metrics, and every figure quoted in the discussion is re-derived from
those same results and checked against the prose before this file is written
(`verify_narrative` in `code/forecast.py`), so the text and the tables cannot
silently disagree.

## 1. Data

The study uses a single file, `opsd_de_load.csv`, published by
[Open Power System Data](https://open-power-system-data.org/) and derived from
the ENTSO-E Transparency Platform. It holds one column of interest,
`DE_load_actual_entsoe_transparency`, the actual German national electricity
load in megawatts, recorded every hour in UTC. Coverage runs from
2015-01-01 00:00:00+00:00 to 2020-09-30 23:00:00+00:00.

We verified rather than assumed the claim that the data is clean. The file
holds 50,400 rows against the 50,400 hours that date range
contains, so no hour is missing and none is duplicated. The counts are
0 missing timestamps, 0 duplicates, 0 NaN values and
0 non-positive readings. That is why there is no imputation step
anywhere in this study: there was nothing to impute. Load ranges from
31,307 MW to 77,549 MW with a mean of 55,492 MW. Because no
value is anywhere near zero, MAPE is well defined at every hour.

The study is deliberately univariate. No temperature, no gas or power prices,
no weather forecast. The only things the models are allowed beyond the load
series itself are facts you can read off a calendar years in advance: the hour,
the day of the week, the month, whether it is a weekend, and whether it is a
German federal public holiday (via the `holidays` package). Those are not
exogenous data in the forecasting sense, because using them requires consulting
no other dataset. The point of the constraint is to isolate how much forecast
skill lives in the temporal structure of load alone, before any
weather-dependence is added.

## 2. Why these six models

The six form a deliberate complexity gradient, and each is there to answer a
specific question.

The **seasonal-naive** rule (forecast hour *t* with the load at *t* minus 168
hours) is the anchor. It costs nothing and encodes the single strongest fact
about electricity load: this week looks like last week. Any model that cannot
beat it is not paying for itself. **SARIMA** asks what a purely statistical
account of the series' own autocorrelation can do. **Prophet** asks what an
explicitly decomposed model buys, where trend, daily, weekly and yearly shapes
and a fitted holiday effect are separate named terms. **LightGBM on engineered
features** asks what happens when a strong general-purpose learner is handed
the domain knowledge directly, as calendar flags and lagged and rolling values
of the load. **N-BEATS** and the **PatchTST slot** ask the opposite question:
given only the raw series and no hand-built features at all, how much of that
structure can a deep network rediscover for itself?

**Substitution, recorded as the prompt requires.** The installed darts version
is 0.41.0, which does not expose a `PatchTSTModel` (verified by listing
`darts.models`; the catalogue offers BlockRNNModel, DLinearModel, NBEATSModel,
NHiTSModel, NLinearModel, TCNModel, TFTModel, TSMixerModel, TiDEModel and
TransformerModel). We therefore used `darts.models.TSMixerModel`, which the
prompt names as the acceptable first option and which is the closest relative
of PatchTST available: same long-horizon forecasting family, same strategy of
mixing information along the time axis with light layers instead of one large
attention matrix. It keeps the name `patchtst` throughout the outputs so the
required schema is respected.

## 3. Validation strategy

The split is fixed in `code/common.py` and every model obeys it:

| Split | Window | Hours |
|---|---|---:|
| Train | 2015-01-01 00:00 to 2019-09-30 23:00 | 41,616 |
| Validation | 2019-10-01 00:00 to 2019-12-31 23:00 | 2,208 |
| Test (held out) | 2020-01-01 00:00 to 2020-01-07 23:00 | 168 |

Hyperparameters are chosen on the validation window and nowhere else. One
detail matters more than it might look. The task we are actually judged on is a
*single 168-hour-ahead forecast*, so scoring candidates on one-step-ahead error
would select for the wrong thing. Instead every configuration is scored by
**rolling-origin evaluation**: seven forecast origins spaced a fortnight apart
across the validation window (1, 15, 29 October; 12, 26 November; 10, 24
December 2019), each producing a full 168-hour forecast from history strictly
before that origin, each scored by MAPE, and the seven averaged. Each model
module runs that loop itself and obtains its history through one shared
primitive, `common.history_before(series, origin)`, which returns strictly
everything earlier than the origin. That single function is the anti-leakage
boundary: a model is handed the slice rather than the full series, so it cannot
see past its own origin even by accident. The 24 December origin is included
on purpose: the test week contains a public holiday, and we want the selection
step to notice which models cope with one.

Once a configuration is selected, the model is **refitted on train plus
validation combined** (2015-01-01 to 2019-12-31) and only then forecasts the
test week. This is the standard practice the prompt requires: the final model
should use all data prior to the test window. The seasonal-naive baseline skips
selection because it has nothing to select.

No test observation entered any fitting or selection decision. To be precise
about what that claim covers: the candidate grids in each model module were
fixed by probing the *validation* origins only (this is visible in the
`selection_log` each model returns), and the choice within each grid is made by
validation MAPE alone. During development a reduced-grid smoke test did print
test-window scores while checking that the code paths ran; no grid, feature
set, or model class was changed afterwards, and the SARIMA differencing bug
described in the appendix was found and fixed from validation-window evidence
before that point.

## 4. Results

All figures are 300 dpi and share one colour per model across every panel.

| Model | Test MAPE (%) | RMSE (MW) | MAE (MW) | MAPE 1 Jan, holiday (%) | MAPE 2-7 Jan, ordinary days (%) | Validation MAPE (%) | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 5.47 | 3586 | 2873 | 4.56 | 5.62 | 3.13 | 1683.4 |
| Prophet | 7.31 | 4721 | 3637 | 13.35 | 6.30 | 6.41 | 287.6 |
| PatchTST | 8.62 | 5500 | 4553 | 4.62 | 9.28 | 6.74 | 3505.3 |
| N-BEATS | 11.13 | 7806 | 5857 | 3.00 | 12.49 | 5.42 | 2812.0 |
| Seasonal naive | 12.78 | 8809 | 7238 | 6.52 | 13.82 | n/a (no hyperparameters) | 0.0 |
| SARIMA | 14.33 | 9482 | 8058 | 6.24 | 15.68 | 4.93 | 1040.7 |

Per-day MAPE across the test week:

| Model | 1 Jan (Wed, holiday) | 2 Jan (Thu) | 3 Jan (Fri) | 4 Jan (Sat) | 5 Jan (Sun) | 6 Jan (Mon) | 7 Jan (Tue) |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 4.56 | 5.03 | 3.14 | 3.92 | 5.77 | 9.34 | 6.54 |
| Prophet | 13.35 | 7.01 | 3.39 | 11.13 | 8.20 | 2.61 | 5.47 |
| PatchTST | 4.62 | 6.70 | 4.32 | 9.46 | 19.04 | 6.85 | 9.32 |
| N-BEATS | 3.00 | 4.38 | 8.53 | 19.46 | 28.24 | 2.76 | 11.55 |
| Seasonal naive | 6.52 | 19.17 | 12.35 | 9.40 | 5.54 | 11.77 | 24.71 |
| SARIMA | 6.24 | 16.27 | 12.69 | 12.62 | 10.48 | 15.44 | 26.57 |

Probabilistic scores for the winning model (LightGBM):

| Quantity | Nominal | Actual |
|---|---:|---:|
| 80% prediction-interval coverage | 80.0% | 77.4% |
| 95% prediction-interval coverage | 95.0% | 98.8% |

| Pinball loss (MW) | q=0.1 | q=0.5 | q=0.9 |
|---|---:|---:|---:|
| LightGBM | 401.8 | 1177.4 | 372.9 |

## 5. Discussion

**LightGBM won**, with a test MAPE of 5.47%, a 57% reduction against the
seasonal-naive anchor, and it was the only entrant never worse than 9.4% on
any single day. Its advantage is not sophistication. It is the only model that
was handed the calendar directly. The gain ranking needs reading carefully:
`lag_168h` alone accounts for 80.3% of total gain, so LightGBM is mostly doing
what the naive baseline does, looking back one week. What separates it is the
next 8% of gain, held by `is_public_holiday_de` (4.7%) and `day_of_week`
(3.3%). Those two flags are small in aggregate because they matter on only a
handful of hours per year, and they are exactly the hours this test week is
made of. They are what let the model override its own weekly anchor when the
anchor is Christmas.

The week is a hard case by construction, which is what makes the ranking
interesting. The 168 hours immediately before it are 25-31 December 2019, the
most unusual week in the entire 2015-2020 record: its mean load sits at the
0.7th percentile of all seven-day means, 12.9% below the test week. Every
model that leans on "last week" inherits that distortion. The seasonal naive
is 12.9% too low by construction. SARIMA, whose selected configuration was a
weekly-difference variant, is anchored on the same week and adds an ARMA
correction estimated on 26 weeks that end inside the anomaly; it is the only
entrant with negative skill (14.33%, worse than doing nothing). N-BEATS and
TSMixer take that same contaminated week as their 168-hour input context, and
their errors track it exactly: both are the best two models on 1 January
(3.00% and 4.62%) because a holiday-shaped context happens to suit a holiday,
and both collapse on the ordinary weekend that follows, N-BEATS reaching 28.2%
on the Sunday.

**Prophet fails in the opposite direction**, which is the most instructive
result here. It is second overall (7.31%) and second on ordinary days (6.30%),
yet worst of all six on the holiday itself (13.35%), despite being the only
model carrying an explicitly fitted New Year's Day term. The hourly residuals
say why: it under-predicts midnight by about 6,600 MW and over-predicts the
06:00-09:00 window by about 11,000 MW. Prophet's holiday effect is one
multiplier applied to the whole day. It can lower the level, but it cannot
flatten the morning industrial ramp, and flattening that ramp is what actually
happens on 1 January.

**Is the ordering what theory predicts?** Only partly. The complexity gradient
did not hold: the two deep models placed third and fourth, behind a 2017
decomposition model and a gradient-booster with good features. Roughly 41,000
training hours is simply not enough for a univariate network to rediscover
calendar structure it could have been told. What theory does predict, and what
did happen, is that models keyed to *absolute* calendar position beat models
keyed to *relative* recency whenever the recent past is unrepresentative.

**The most important caveat is methodological.** Rank order on the validation
window barely survived to the test window: Spearman correlation between
validation and test MAPE across the five tuned models is 0.10 (p=0.87). SARIMA
was the best non-LightGBM model on validation (4.93%) and the worst on test.
Only LightGBM's win transferred. The cause is that none of the seven validation
origins required forecasting *out of* a holiday period into a normal week,
which is the exact task the test week sets. A single 168-hour window cannot
separate models that differ by a percentage point or two, and this one is not a
typical week.

## 6. Recommendation

**Put LightGBM into the pipeline.** It won on both the validation and the test
window, it retrains cheaply (about 28 minutes here including the entire
hyperparameter sweep, against 47 minutes for N-BEATS and 58 for TSMixer;
Prophet is cheaper still at 5 minutes, and that is a genuine point in Prophet's
favour), it is inspectable in a way that matters to a regulator
(the gain ranking in `07_feature_importance.png` is an auditable statement
about what drives the forecast), and its 80% intervals were the best calibrated
of anything we fitted (77.4% actual against 80% nominal). Before trusting it in
production, though, four things would need doing, in this order. First, replace
this single test week with a proper rolling backtest of 52 or more origins
spread across the year, because the validation-to-test rank collapse documented
above means one week cannot support a model-selection decision. Second, fix the
one systematic bias the residuals expose: the model over-forecasts almost
everywhere on this week (median residual about -2,000 MW, worst at 04:00-07:00
UTC and on the Monday), which is recursive drift accumulating across the
horizon, and the standard remedy is to train separate direct models per horizon
block rather than feeding predictions back in. Third, widen the intervals
honestly, since the 95% band is over-wide (98.8%) while the 80% is slightly
narrow, and the current construction conditions the quantiles on a single
recursive path rather than propagating path uncertainty. Fourth, and by far the
largest expected gain, **relax the univariate constraint**: this study
deliberately excluded temperature, and German load is strongly
temperature-driven in January. The honest reading of the 5.47% headline is that
it measures how far calendar structure alone can go, and a production
short-term load forecaster should be expected to beat it comfortably once
weather forecasts are allowed in.

## 7. Reproducibility

**Seeds.** A single seed, `SEED = 42`, is set in `code/common.py` and
passed to every component that takes one: LightGBM's `random_state` (point and
all five quantile fits), darts' `random_state` for N-BEATS and TSMixer (which
seeds torch), and `numpy.random.seed` before Prophet's sampling. SARIMA is
deterministic given the data, as is the naive baseline.

**Residual non-determinism, and what we did not check.** Two sources remain.
Prophet's L-BFGS fit through cmdstanpy is not guaranteed bit-identical across
platforms, and multi-threaded floating-point reduction in LightGBM and torch
can reorder additions. We did not rerun the full study end to end to measure
how large that wobble actually is, so no reproducibility tolerance is claimed
here. Anyone repeating this should expect the seeded configuration selections
to be stable and the third decimal place of a MAPE not to be.

**Hardware and wall clock.** The per-model runtimes in the results table sum
to 9329 seconds (155.5 minutes). That total is
not the wall-clock time, because the six models were fitted as three concurrent
processes (naive, SARIMA, Prophet and LightGBM in sequence in one; N-BEATS in a
second; TSMixer in a third), which finished in about 59 minutes of wall clock.
The concurrency cuts both ways and should be read as a caveat on the runtime
column: three processes competing for the same cores inflated every individual
timing, so these numbers rank the models' relative cost only roughly and would
all be smaller if run alone. The neural models were trained on CPU on purpose:
Apple's MPS backend rejects the float64 arrays darts supplies, and on models
this size the CPU is fast enough (10-20 seconds per epoch on an idle machine)
while being repeatable.

```
Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64
```

Python 3.12.13, pandas 2.3.3, numpy
2.4.4. Key modelling libraries: statsmodels 0.14.6, pmdarima 2.1.1,
darts 0.41.0, prophet 1.3.0, lightgbm 4.6.0, torch 2.10.0, holidays 0.96.

**How to reproduce.** From this directory:

```
../../.venv/bin/python code/forecast.py            # fit everything, write all outputs
../../.venv/bin/python code/forecast.py --report-only   # redraw from cached fits
```

Per-model fits are cached under `code/_cache/`; delete that folder to force a
cold run.

## Appendix: notes and caveats

**LightGBM forecasts recursively.** The test task is one 168-hour-ahead
forecast made on 31 December 2019, so a `lag_24h` feature for 7 January would
be a value from inside the test week and is not knowable at forecast time. The
model therefore predicts hour 1, feeds that prediction back in as history,
predicts hour 2, and so on for 168 steps. No test observation is ever used as
an input, which is what makes the comparison against the other five models
fair.

As a diagnostic we also ran the easier teacher-forced variant, where the model
is handed the real recent load at every step: it scores 3.66% MAPE against
5.47% for the honest recursive forecast. The gap is the price of not knowing
the near past, and it is the reason a published MAPE for a load model is
meaningless without the forecast horizon attached.

**Interval width for LightGBM is conditional on one path.** The five quantile
models are evaluated along the single recursive trajectory produced by the
point model. They therefore describe the spread around that path and do not
accumulate the uncertainty of the path itself, so they are expected to be too
narrow at long horizons. The measured coverage reported above is the honest
check on this, and should be read with it in mind.

**A bug worth reporting, because the symptom was a plausible number.** The
first working version of the SARIMA module passed `simple_differencing=True` to
statsmodels. Under that setting statsmodels fits the differenced series and
`get_forecast` returns forecasts *of the differenced series*, not of load. In
the weekly-difference variant those near-zero values were then added back to
last week's load, so the model silently degenerated into the seasonal-naive
baseline. It did not crash and it did not look wrong: it scored 7.12% on the
first validation origin, a perfectly reasonable-looking number. What exposed it
was that the seasonal naive scored 7.12% on the same origin, to two decimal
places. The lesson generalises past this one flag. A forecast that is quietly
equal to a baseline is indistinguishable from a working model by its error
metric alone, so it is worth checking explicitly that a model's forecast
actually differs from the baseline's (SARIMA's final forecast differs from the
naive one by 1,682 MW on average, which is how we know the fix took). The
`simple_differencing=False` setting now carries a comment in `code/sarima.py`
explaining why it must stay.

**Skill against the baseline.** The winner reduces MAPE by 57% relative
to the seasonal-naive anchor (5.47% against 12.78%).
Note that the naive baseline is unusually handicapped here: the week it copies
is 25-31 December 2019, which is Christmas, when German load is far below a
normal week.

**One week is one week.** Every conclusion here rests on 168 hours containing
one public holiday. Differences of a few tenths of a percentage point between
adjacent models are not resolvable at this sample size, and the ranking should
be read as indicative rather than settled. A production assessment would repeat
this over many rolling test weeks spread across the year.
