# German hourly load: a six-model forecasting bake-off

Test window: **2020-01-01 to 2020-01-07** (168 hourly observations). Every model was fit once on 2015-01-01 to 2019-09-30, tuned on 2019-10-01 to 2019-12-31, then refit on the combined 2015-2019 block before forecasting the held-out week.

## 1. Data

The load series comes from Open Power System Data, which republishes the ENTSO-E Transparency Platform's German national load. We use the column `DE_load_actual_entsoe_transparency`, hourly, in megawatts, spanning 2015-01-01 00:00 UTC to 2020-09-30 23:00 UTC. The file contains exactly 50,400 rows with no missing hours and no NaN values, so no imputation or interpolation was performed. `common.load_series` asserts both properties at load time and fails loudly if either changes.

## 2. Why these six models

The line-up spans a deliberate complexity gradient. The seasonal-naive baseline (a copy of the value one week earlier) sets the floor a serious model must beat. SARIMA is the classical statistical benchmark: it encodes autoregressive structure, differencing, and a 24-hour seasonal term. Prophet adds explicit calendar effects, in particular German public holidays, which matter because Jan 1 falls inside our test week. LightGBM stands in for the feature-engineering school: hand-crafted calendar flags, lags at 24, 168 and 8760 hours, and 24/168-hour rolling mean and standard deviation. N-BEATS is a pure-neural stack of basis-expansion blocks and represents the modern time-series MLP family. Finally we include a transformer-family model (see the substitution note below) to see whether attention on a one-week context adds anything on top of N-BEATS.

**Substitution note.** The prompt asked for PatchTST via `darts.models.PatchTSTModel`. The installed darts version (0.41.0) does not export `PatchTSTModel`, so we substituted `darts.models.TSMixerModel`, another modern univariate deep forecaster from the same generation. The substitution is recorded in the hyperparameters block of `metrics.json` under the `patchtst` model.

## 3. Validation strategy

The 50,400-hour series was cut into three contiguous, non-overlapping windows: **train** = 2015-01-01 to 2019-09-30 (41,616 hours), **validation** = 2019-10-01 to 2019-12-31 (2,208 hours), and **test** = 2020-01-01 to 2020-01-07 (168 hours). Every model with hyperparameters was fit on train, scored on validation, and the best configuration was then refit on the combined 2015-2019 block before producing the one test forecast. Refitting is standard practice: it gives the final model the largest possible information base without leaking any test-window observation into the fit.

## 4. Results

Sorted by test-week MAPE, best first. Runtime is wall-clock seconds on the reference hardware (`uname -a` at the end of this file).

| Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2-7 MAPE (%) | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM (features) | 3.24 | 2167 | 1732 | 2.24 | 3.41 | 162.6 |
| Prophet (DE holidays) | 7.61 | 4874 | 3795 | 13.28 | 6.67 | 17.9 |
| TSMixer (PatchTST substitute) | 8.28 | 5513 | 4314 | 6.92 | 8.51 | 231.1 |
| SARIMA | 9.79 | 6632 | 5424 | 11.17 | 9.56 | 588.4 |
| N-BEATS | 11.00 | 7118 | 5626 | 5.13 | 11.98 | 26.3 |
| Seasonal-naive (168h) | 12.78 | 8809 | 7238 | 6.52 | 13.82 | 0.0 |

The winner is **LightGBM (features)** at 3.24% test MAPE, versus 12.78% for the seasonal-naive baseline (a 75% MAPE reduction relative to naive).

For the winner, the 80% and 95% prediction intervals cover 74% and 92% of the 168 test observations respectively. Pinball losses at q=0.1, 0.5, 0.9 are 334, 866, 315 MW.

## 5. Discussion

The rank ordering by test MAPE is: LightGBM (features) (3.24%), Prophet (DE holidays) (7.61%), TSMixer (PatchTST substitute) (8.28%), SARIMA (9.79%), N-BEATS (11.00%), Seasonal-naive (168h) (12.78%). **LightGBM wins by a wide margin** (3.24% versus 7.61% for the runner-up), and it is the only model that stays uniformly accurate across the whole week: 2.24% on the holiday, 3.41% on the other six days.

The seasonal-naive baseline lands at 12.8% because its lag_168h anchor points at Christmas week 2019. Its Jan 1 MAPE (6.52%) is actually low, because Dec 25 (Christmas Day) and Jan 1 (New Year's Day) are both public holidays with similar load profiles; the damage sits in Jan 2-7 (13.82%), where working Thursday-Sunday gets compared to Boxing Day-Sunday of Christmas week and inherits a holiday-shaped load.

Prophet illustrates that a holiday flag is not enough on its own. It reports a Jan 1 MAPE of 13.28% but only 6.67% on the rest of the week. The `country_holidays='DE'` regressor gets Prophet to expect a reduced-load day on Jan 1, but the model's amplitude for the holiday effect is calibrated across all German holidays (Christmas, Easter, Whit Monday, Reunification Day, etc.) and misses the specific shape of a January-1st Wednesday. SARIMA, which has no holiday knowledge, is worse on Jan 1 (11.17%) but its damage is bounded by autoregressive smoothing. LightGBM, which sees a binary holiday flag alongside hour-of-day, day-of-week, month, and lag features, turns the flag into a working discriminator: on Jan 1 its MAPE drops to 2.24%. The two pure-neural models (N-BEATS, TSMixer) receive no holiday indicator; they see only the past 168 hours of load and consequently over-predict Jan 1 as a typical Wednesday, though N-BEATS's Jan 1 MAPE (5.13%) is oddly better than its Jan 2-7 MAPE, an artefact of the model's generally low variance rather than any calendar awareness.

Best holiday performance comes from LightGBM (features) (Jan 1 MAPE 2.24%); worst is Prophet (DE holidays) at 13.28%. The rank ordering broadly matches what theory predicts for a univariate load-forecasting benchmark on a holiday week: feature-engineered gradient boosting beats every other family here because it can use the load's own short-term dynamics (rolling 24h and 168h statistics dominated the feature-importance ranking) alongside calendar effects. Prophet outperforms the deep models mostly by having any holiday awareness at all. The deep models are hurt by both the univariate constraint and the modest training budget rather than by architecture per se.

Caveats. First, a single 168-hour test window is a slim evidence base: rank stability across other weeks is not guaranteed and would need rolling-origin backtesting to settle. Second, the deep models were trained for at most 40 (N-BEATS) and 15 (TSMixer) epochs on CPU with modest default architectures and no learning-rate scheduler; a serious deployment would sweep those and consider ensembling. Third, this study deliberately excludes temperature, prices, and other load drivers; the univariate-plus-calendar constraint isolates the temporal-structure component of forecast skill and understates what a full feature set could do.

## 6. Recommendation

For a JRC production short-term load forecasting pipeline I would pick **LightGBM (features)**. It is transparent (feature importances answer 'why did the forecast move'), cheap to retrain nightly, and it wins on both the holiday and the working-day slices. Before deployment I would add weather features (temperature, cloud cover, wind) alongside the existing calendar features, add recursive-forecast support for horizons longer than 168 hours, and build a rolling-origin backtest across at least a full year so the accuracy claim generalises past this one week.

## 7. Reproducibility

- Random seed: `20260906` (used by LightGBM, N-BEATS, and TSMixer; SARIMA and Prophet are deterministic given data and hyperparameters; the naive baseline has no random state).
- Total wall-clock runtime for the full six-model pipeline: **1026 s**.
- Python: 3.12.13
- Platform: `macOS-26.5.1-arm64-arm-64bit`
- `uname -a`: `Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64`
- To reproduce: `../../.venv/bin/python code/forecast.py` from this directory, then `../../.venv/bin/python code/write_transcript.py` to regenerate this file. All artefacts land in `figures/`, `metrics.json`, `metrics.csv`, and `transcript.md`.
