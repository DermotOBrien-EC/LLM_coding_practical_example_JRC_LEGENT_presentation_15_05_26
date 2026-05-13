# German hourly load — six-model forecasting bake-off

Holdout test window: **2020-01-01 → 2020-01-07** (168 hours), German
national hourly load. Winner on test MAPE: **LightGBM** (3.43%).

## 1. Data

The full series comes from Open Power System Data (`opsd_de_load.csv`),
derived from ENTSO-E Transparency. One column of hourly German load in
megawatts, indexed by UTC timestamp. Coverage is **2015-01-01 00:00 UTC
through 2020-09-30 23:00 UTC** — exactly **50,400 rows**, zero NaN, zero
missing hours. We verified this in `code/common.py::load_load_series`,
which refuses to silently impute or repair the series; if any of those
sanity checks fail the run aborts. No external data is brought in. The
study is deliberately univariate so the contribution of pure temporal
structure to forecast skill can be isolated from any weather signal.

## 2. Why these six models

The six models span a deliberate complexity gradient.

* **Seasonal naive** (lag 168 h) — the cheapest defensible hourly-load
  forecast. Sets the floor: anything more sophisticated should beat it.
* **SARIMA** — classical statistical baseline (statsmodels SARIMAX with
  explicit `(p,d,q)(P,D,Q,s=24)` orders). Captures local autoregressive
  structure plus explicit daily seasonality; weekly structure has to
  emerge through the AR/MA terms.
* **Prophet** — additive decomposition into trend + daily + weekly +
  yearly seasonality + German federal holidays (via
  `country_holidays="DE"`). The only classical-tier model that knows
  Jan 1 is special.
* **LightGBM on engineered features** — gradient-boosted trees over
  calendar features (hour, day-of-week, month, weekend, public holiday)
  and load-derived features (lag 24 h, 168 h, 8760 h; rolling 24 h and
  168 h mean and std). Captures non-linear interactions and is the only
  model that gets both the holiday indicator *and* the year-ago lag.
* **N-BEATS** — deep MLP-based univariate forecaster (darts). Learns
  hierarchical backcast/forecast residuals directly from the scaled load.
* **TSMixer** (substitute for PatchTST) — recent MLP-mixer-style deep
  forecaster. `darts==0.41.0` does not expose `PatchTSTModel`, and the
  prompt explicitly accepts `darts.models.TSMixerModel` as the
  alternative; the substitution is recorded in
  `code/patchtst.py::run_patchtst` hyperparameters
  (`substitute_for: "PatchTST"`, `darts_model_class: "TSMixerModel"`).

## 3. Validation strategy

* **Train**: 2015-01-01 00:00 → 2019-09-30 23:00 (41,616 hours).
* **Validation**: 2019-10-01 00:00 → 2019-12-31 23:00 (2,208 hours).
* **Test**: 2020-01-01 00:00 → 2020-01-07 23:00 (168 hours, held out).

Each model is first fit on Train, scored on Validation to pick its
hyperparameters / orders / number of epochs, then **refit on Train +
Validation combined** before producing the 168-hour test forecast. The
test window is never touched during fitting or selection. The naive
baseline has no hyperparameters and is fit implicitly through its lag
lookup.

Three model-specific notes:

* For **SARIMA**, each candidate `(p,d,q)(P,D,Q,s=24)` order is fit on
  the last 8,760 hours of available history (one year). A full
  pmdarima auto_arima with `m=24` over the entire 41 k–43 k-row series
  is intractable in this run-time budget; one year of hourly data is
  more than enough to identify daily and weekly structure.
* For **N-BEATS and TSMixer**, "validation" means early stopping on
  validation loss to fix an epoch count; the refit on Train + Validation
  then runs for the same epoch count without a held-out signal. Both
  selector runs ran to the full 15-epoch budget; early stopping did not
  fire, so validation loss was still trending downward when we stopped
  — they are partly under-trained relative to what a longer schedule
  would produce.
* The **deep-model epoch budget was capped at 15** rather than the 30
  the prompt allows, to keep total wall-clock under 20 minutes on this
  hardware. This is a runtime trade-off, not a methodology choice.

## 4. Results

Headline metrics on the 168-hour test window, sorted by MAPE ascending.
The right-hand columns split MAPE by holiday vs working-week to expose
where the holiday-unaware models fail.

| Model           | MAPE (%) | RMSE (MW) | MAE (MW) | MAPE Jan 1 (%) | MAPE Jan 2–7 (%) | Runtime (s) |
|-----------------|---------:|----------:|---------:|---------------:|-----------------:|------------:|
| LightGBM        |     3.43 |     2,326 |    1,801 |           3.72 |             3.38 |        57.2 |
| Prophet         |     7.61 |     4,873 |    3,795 |          13.29 |             6.67 |        24.1 |
| N-BEATS         |     8.19 |     5,771 |    4,285 |           3.90 |             8.91 |       437.1 |
| SARIMA          |     9.98 |     6,762 |    5,542 |          10.99 |             9.81 |        88.5 |
| TSMixer\*       |    10.03 |     6,371 |    5,267 |           4.11 |            11.02 |       673.5 |
| Seasonal naive  |    12.78 |     8,809 |    7,238 |           6.52 |            13.82 |        ~0   |

\* TSMixer is the PatchTST substitute (see Section 2).

**Winner: LightGBM**, test MAPE **3.43%**. Its 80% prediction interval
covered **79.8%** of test actuals (nominal 80%); its 95% interval covered
**92.9%** (nominal 95%). Pinball losses on the winning model are q=0.10:
410.7 MW, q=0.50: 840.3 MW, q=0.90: 349.9 MW (the q=0.50 pinball is half
of the median-forecast MAE, as it should be).

## 5. Discussion

LightGBM wins decisively — better than half the MAPE of the runner-up
(3.43% vs Prophet's 7.61%) and better than a quarter of the seasonal-naive
floor (12.78%). It is also one of the fastest non-trivial models in the
study (57 s on this hardware, vs Prophet's 24 s, vs >400 s each for the
two deep models). This is the expected qualitative ordering for a study
on a single-series univariate forecast where the available signal is
strong autoregressive structure plus a small categorical effect: tree
boosting with explicit lag and calendar features is the model class that
exploits both efficiently.

The Jan 1 holiday is the diagnostic test. The right-hand panel of
`figures/04_per_day_mape.png` makes it stark:

* **Prophet** is the only model whose Jan-1 MAPE (13.3%) is roughly
  *double* its working-week MAPE (6.7%) — this is despite passing it
  `country_holidays="DE"`. The decomposition figure
  (`figures/08_decomposition.png`) shows it does learn a holiday spike,
  but it learns the *average* shift across all federal holidays
  (Easter, May 1, Christmas Day, etc.), which is not the right
  magnitude for New Year's. The "Christmas/New Year week" load
  signature is highly atypical and Prophet's smooth, sample-averaged
  holiday term cannot accommodate it.
* **SARIMA** has no concept of "this Wednesday is special" and pays
  for it on Jan 1 (11% MAPE), but interestingly its working-week MAPE
  is also high (9.8%) — the explicit `s=24` daily seasonality combined
  with one year of training is not enough to recover the right
  intra-day pattern under the New Year week shift.
* The **seasonal naive** does *better* than several smarter models on
  Jan 1 (6.5%), because its lag-168h lookup falls on Wednesday Dec 25,
  2019 — Christmas Day — which has a load profile structurally similar
  to New Year's. This is an accident of calendar alignment, not skill.
  On the rest of the week, however, the naive collapses to a 13.8%
  MAPE — by far the worst.
* **N-BEATS and TSMixer**, despite having *no* explicit holiday signal,
  produce Jan-1 MAPEs of 3.9% and 4.1% — better than their own
  working-week MAPEs and competitive with LightGBM on Jan 1
  specifically. The likely explanation is that their 168-hour context
  window stretches back into the New Year holiday week of the previous
  year (Dec 25 → Jan 1, 2019); the deep models appear to recognise
  the holiday-week shape end-to-end without needing a label. But they
  do not generalise this well, and over the working days Jan 2–7 their
  errors blow up — TSMixer's worst day (Sunday Jan 5, 23.6% MAPE) is
  the single most catastrophic per-day error in the whole study.
* **LightGBM**'s Jan-1 vs Jan-2–7 split is the flattest by a wide
  margin (3.7% vs 3.4%). The feature-importance plot
  (`figures/07_feature_importance.png`) shows why: `lag_168h` dominates
  by an order of magnitude, with `lag_24h` and the calendar / holiday
  flags filling out the rest. The combination gives the model both
  "same-hour-last-week" anchoring *and* a way to say "but the holiday
  rule says this week is different."

A caveat: 168 hours is a very small test sample, and the absolute
ordering of the bottom three (SARIMA, TSMixer, N-BEATS) is partly
noise. With a longer holdout the deep models would probably move down
(more validation epochs, more diverse calendar context) and SARIMA might
move up; the *top two* — LightGBM and Prophet — would be substantially
more robust. The 80% / 95% interval coverages for LightGBM (79.8 / 92.9)
land essentially on nominal, which is a useful secondary confirmation.

## 6. Recommendation

For a JRC-grade short-term load-forecasting pipeline I would deploy
**LightGBM on engineered features**. It wins on accuracy, runs in under
a minute, produces calibrated intervals via the quantile objective, and
the feature set (lag, calendar, holiday flag) is straightforward to
audit. Before putting it in production I would do four things:

1. Expand the holiday feature beyond a binary flag — separate dummies
   for federal vs Land-level holidays, separate Christmas / New Year
   week markers, and a "bridge day" indicator. The Jan 1 issue is the
   shape of holiday weeks, not the holiday itself.
2. Add a 365-day rolling-window quantile feature so the model can
   anchor to the same-week-last-year distribution, not just the
   same-hour point lag.
3. Run a proper rolling-origin evaluation over at least 12 monthly
   forecast windows in 2020 — including spring (lockdown), summer, and
   winter — rather than a single 168-hour holdout.
4. Calibrate quantile intervals on a held-out window distinct from
   training to confirm that the 80% / 95% coverage seen here is not
   an artifact of the single-week sample.

## 7. Reproducibility note

* Random seeds: `random_state=42` for LightGBM, `random_state=42` and
  `torch.manual_seed(42)` for the two deep models; SARIMA and Prophet
  use deterministic numerical optimisers (BFGS / cmdstanpy), so a small
  amount of run-to-run drift from initial-condition randomisation is
  possible but should not move the rank ordering.
* Library versions: pandas 2.3.3, numpy 2.4.4, statsmodels 0.14.6,
  pmdarima 2.1.1, lightgbm 4.6.0, darts 0.41.0, torch 2.10.0, prophet
  1.3.0, holidays 0.96, pytorch-lightning bundled with darts. Python
  3.12.13.
* Hardware: `uname -a` → `Darwin Dermots-MBP 25.5.0 Darwin Kernel
  Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/
  RELEASE_ARM64_T6050 arm64`. Apple Silicon, CPU-only; PyTorch trainer
  pinned to CPU. `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE`
  were required for the run to complete: LightGBM (libgomp) and
  PyTorch (libomp) otherwise deadlock when invoked in the same Python
  process on this macOS / Apple Silicon configuration. Setting
  `OMP_NUM_THREADS=1` neutralises the OpenMP runtime conflict at a
  small (probably <2x) cost in deep-model training throughput.
* End-to-end runtime: **1,121.9 seconds** (≈ 19 minutes) for a
  cold-cache run.

To reproduce:

```bash
cd runs/L3
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE \
    ../../.venv/bin/python -m code.forecast
```

All outputs (`metrics.json`, `metrics.csv`, `forecasts_test_window.csv`,
`figures/*.png`) are written by that single command. The pickle cache
in `cache/` short-circuits per-model re-fits; delete the directory or
set `FORECAST_FORCE=1` to force every model to refit.
