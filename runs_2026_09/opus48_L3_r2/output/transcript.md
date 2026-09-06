# German hourly load: a six-model forecasting bake-off

## 1. Data

The single input is `opsd_de_load.csv` from Open Power System Data, itself
derived from the ENTSO-E Transparency Platform. It holds one column of interest,
the actual German national load in megawatts, sampled every hour from
2015-01-01 00:00 UTC to 2020-09-30 23:00 UTC. That is exactly 50,400 hourly
rows. On loading we check the row count, confirm there are no missing values,
and confirm the hourly timestamps form an unbroken sequence with no gaps or
duplicates. All three checks pass, so no imputation or gap-filling is needed and
none is done: the study runs on the raw series exactly as delivered.

## 2. Why these six models

The six models are a deliberate complexity gradient, from "almost no model" to
"deep neural network", so we can see how much each extra layer of machinery
actually buys.

- **Seasonal-naive** repeats the load from the same hour one week earlier. It
  has no parameters and exists as the bar every other model must clear.
- **SARIMA** is the classical statistical workhorse: autoregression, moving
  averages, and a daily seasonal term after differencing.
- **Prophet** fits a smooth trend plus daily, weekly and yearly seasonal
  shapes, and is explicitly told about German public holidays.
- **LightGBM** abandons the time-series view and instead predicts each hour
  from engineered features: calendar fields, holiday flag, lags of the load
  (one day, one week, one year), and rolling means and spreads.
- **N-BEATS** is a deep fully-connected network that learns the forecast purely
  from the recent shape of the series.
- **PatchTST** (served here by TSMixer, see below) is a modern deep architecture
  aimed at longer horizons.

The gradient tests a real question: on a purely univariate problem, does
sophistication pay, or does a good feature set plus gradient boosting win?

Substitution note: PatchTST is not available in the installed darts (0.41.0), so the PatchTST slot is filled by darts TSMixerModel, the first substitute the prompt allows. Reason recorded: darts 0.41.0 does not expose PatchTSTModel.

## 3. Validation strategy

The data is cut into three non-overlapping blocks in time. Train is
2015-01-01 to 2019-09-30 (about 4.75 years, 41,616 hours). Validation is
2019-10-01 to 2019-12-31 (2,208 hours). Test is the 168 hours of
2020-01-01 to 2020-01-07, held out and scored exactly once.

Every model with settings to choose is fit on Train only, then used to forecast
the first week of the validation window (2019-10-01 to 2019-10-07); the setting
with the lowest validation MAPE is kept. This mirrors the real test task, which
is a single 168-hour-ahead forecast. Once a model's settings are locked, the
model is refit on Train and Validation combined (all data up to
2019-12-31 23:00) and only then asked to forecast the test week. Refitting on
all pre-test data is standard practice: there is no reason to withhold the
validation months from the final model once they have done their tuning job.
The test week is never seen during any fitting or selection step.

## 4. Results

| Model | MAPE (%) | RMSE (MW) | MAE (MW) | MAPE Jan 1 (%) | MAPE Jan 2-7 (%) | Runtime (s) |
|---|---|---|---|---|---|---|
| LightGBM | 4.61 | 3,094 | 2,487 | 2.77 | 4.92 | 146.8 |
| Prophet | 7.66 | 4,888 | 3,822 | 13.43 | 6.70 | 51.0 |
| N-BEATS | 9.86 | 6,691 | 5,391 | 10.94 | 9.68 | 53.1 |
| PatchTST (TSMixer) | 11.55 | 8,613 | 6,729 | 5.52 | 12.55 | 42.5 |
| Seasonal-naive | 12.78 | 8,809 | 7,238 | 6.52 | 13.82 | 0.0 |
| SARIMA | 12.87 | 8,835 | 7,292 | 8.45 | 13.61 | 37.8 |

Winner: **LightGBM**, with a test MAPE of 4.61%, which is 64%
lower than the seasonal-naive baseline (12.78%).

For the winner, the prediction intervals were checked against the actual load:

- 80% interval: nominal 80%, actual coverage 76.8%
- 95% interval: nominal 95%, actual coverage 96.4%
- Pinball loss (MW): q0.1 = 419.6, q0.5 = 1,065.2, q0.9 = 367.7

## 5. Discussion

The ranking is close to what forecasting theory would predict for a
univariate problem with a strong, learnable calendar structure. LightGBM won because
its features hand the model exactly the things that drive load: what hour and
weekday it is, whether it is a holiday, and where the level has been over the last
day, week and year. Gradient boosting then only has to learn a fairly smooth map
from those features to demand.

The New Year holiday is a more subtle test than it first looks, and the per-day
breakdown (figure 4) overturns the obvious expectation that every model should
stumble on Jan 1. For the models that lean on the one-week lag it is the reverse:
the seasonal-naive baseline predicts the Jan 1 holiday well (6.5% MAPE)
and LightGBM is best of all on it (2.8%). The reason is a calendar
coincidence. One week before Jan 1 2020 is Dec 25 2019, Christmas Day, itself a
depressed-demand holiday, so the one-week lag quietly hands these models a
holiday-shaped template for the New Year. Their real weakness is the mirror image:
the worst single day in the whole study is the post-holiday return to normal.
Seasonal-naive's largest error is Tue 7 Jan (24.7%), because the day
one week earlier (New Year's Eve, still inside the Christmas lull) is a poor
template for an ordinary working Tuesday. So the lag is a gift on the holiday and a
trap on the recovery. The one model clearly caught out on Jan 1 itself is Prophet
(13.4%): its additive holiday term over-corrects and
pushes the Jan 1 forecast well below the actual load, the deep dip visible in its
panel of figure 2.

Where each model fails is instructive. Seasonal-naive is exact whenever this week
resembles last week and badly wrong when it does not, so its error sits on the
recovery days, not the holiday. SARIMA, restricted to daily seasonality, never
really captures the weekly rhythm and lands essentially on top of the naive
baseline: the classical machinery buys almost nothing here without a weekly
seasonal term. Prophet has the right seasonal vocabulary and a holiday term, so it
is the runner-up, but it fits a smooth average shape, misses the sharper
hour-to-hour moves, and over-shoots the New Year dip. The two deep models are a
surprise only if one expected depth to win automatically: with a single week of
context, no calendar inputs, and under five years of hourly data, they have to
relearn structure the feature-based model is simply handed, and they finish
mid-pack, noticeably noisier hour to hour (figure 2). This matches the standard
finding that on tabular, feature-rich forecasting problems, gradient-boosted trees
are very hard to beat. The main caveat is that this is a single 168-hour window
over an unusual holiday week, so the exact ranking should not be over-read; a fair
operational verdict would repeat the exercise over many rolling windows across the
seasons.

## 6. Recommendation

For a JRC production short-term load forecasting pipeline, the model to put in
first is **LightGBM**. It gave the lowest error here, it is cheap to fit and fully
deterministic (a rerun gives the identical answer; the runtime in the table is
mostly the hyperparameter sweep and the hour-by-hour interval forecast, not a
single fit), and its feature-based design is the natural place to later add the
exogenous signals this study deliberately excluded. The first improvement to make would be
to add weather, above all temperature, since heating and cooling demand is the
single largest driver of load that a purely calendar-and-lag model cannot see.
After that: a proper holiday-and-bridge-day calendar (the day after New Year, the
days between Christmas and New Year), and rolling-origin backtesting across many
weeks and seasons rather than a single winter test week, so the error estimate
is not tied to one unusual holiday period. Prediction intervals would be
recalibrated on a validation set before anyone relied on them operationally.

## 7. Reproducibility

- Random seed: 42 (Python, NumPy, and PyTorch).
- The two deep models (N-BEATS, TSMixer) train on the Apple MPS GPU when
  available; MPS is not bit-for-bit deterministic across runs, so their numbers
  can move by a few tenths of a MAPE point between runs. The tree, statistical
  and baseline models are fully deterministic, and the winner is one of those,
  so the headline result is stable.
- Total model-fitting wall-clock this run: 331 seconds.
- Hardware / OS: Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64
- To reproduce: `../../.venv/bin/python code/forecast.py` from this directory.
