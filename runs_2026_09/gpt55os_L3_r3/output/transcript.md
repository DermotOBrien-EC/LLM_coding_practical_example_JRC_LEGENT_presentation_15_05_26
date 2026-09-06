# German hourly load forecasting bake-off

## Data

The study uses `opsd_de_load.csv`, an Open Power System Data extract derived from the ENTSO-E Transparency Platform. It contains German national hourly load from 2015-01-01 00:00 UTC to 2020-09-30 23:00 UTC, with 50,400 rows and one target column, `DE_load_actual_entsoe_transparency`, measured in megawatts. The timestamp index was rechecked as hourly, gap-free, duplicate-free, and without missing target values, so no imputation was needed.

## Why these six models

The six models form a complexity gradient. The seasonal-naive model asks how much skill comes from simply copying the same hour from the previous week. SARIMA adds a classical stochastic time-series structure. Prophet adds explicit daily, weekly, yearly, and German-holiday components. LightGBM uses timestamp features, public-holiday flags, lags, and rolling summaries in a supervised learning setup. N-BEATS and the transformer-family TSMixer substitute test whether modern neural sequence models extract more from the same univariate history.

## Validation strategy

The initial training window was 2015-01-01 to 2019-09-30 (41,616 hourly rows), validation was 2019-10-01 to 2019-12-31 (2,208 rows), and the held-out test was 2020-01-01 to 2020-01-07 (168 rows). Hyperparameters or model orders were selected only on the validation window. After selection, each non-naive model was refit on train plus validation (43,824 rows) before the single final forecast of the test week.

## Results table

| Rank | Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2 to Jan 7 MAPE (%) | Runtime (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | LightGBM | 5.46 | 3785 | 3018 | 2.59 | 5.93 | 27.5 |
| 2 | Prophet | 7.61 | 4878 | 3798 | 13.28 | 6.67 | 35.6 |
| 3 | TSMixer | 9.58 | 7422 | 5530 | 4.69 | 10.39 | 156.6 |
| 4 | N-BEATS | 9.81 | 6516 | 5127 | 8.63 | 10.01 | 46.6 |
| 5 | Seasonal naive | 12.78 | 8809 | 7238 | 6.52 | 13.82 | 0.0 |
| 6 | SARIMA | 13.81 | 8157 | 7052 | 16.32 | 13.39 | 30.4 |

## Discussion

LightGBM won on the primary metric, with test MAPE of 5.46%. The margin over Prophet was 2.16 percentage points, so the ranking should be read as a short-horizon test-week result rather than a universal law. The seasonal-naive baseline reached 12.78% MAPE, which is a strong reminder that German load is highly weekly. A model has to add holiday handling, local trend adjustment, or nonlinear lag interactions to justify its extra machinery. LightGBM's result (5.46%) shows the value of simple, knowable calendar variables plus direct lag features: the model can use the Jan 1 holiday flag while still leaning on the previous day, previous week, and aligned previous year. Prophet's German holidays make it interpretable and useful for explaining the holiday effect, but its smooth components can miss the exact level of an unusual week; here it scored 7.61%. The neural sequence models, TSMixer (9.58%) and N-BEATS (9.81%), had enough flexibility to learn repeated daily and weekly shape, but this small bake-off gives them only one univariate series and a short validation target. That is not the setting where deep models usually dominate. SARIMA is the most transparent stochastic benchmark, but daily seasonality is only a proxy for the true 168-hour structure, so weekly and holiday mismatches appear as residual structure. Across models, Jan 1 is the stress test: it is a Wednesday public holiday, so models that mainly copy ordinary Wednesdays or recent weekdays over-predict morning and daytime load.

## Recommendation

For a JRC production short-term load forecasting pipeline under the same univariate-input rule, I would start from LightGBM because it had the lowest held-out MAPE in this bake-off. Before production use, I would expand the evaluation to rolling weekly backtests across multiple years, keep the holiday-vs-working-day breakdown as a standing diagnostic, calibrate prediction intervals on recent residuals, and only then consider weather variables in a separate, clearly labelled multivariate study.

## Reproducibility note

Random seed 42 was used for LightGBM and the Darts neural models. The categorical figure palette is the validated six-slot light-mode subset from the data-visualization reference palette: blue, orange, aqua, yellow, magenta, and green. Total wall-clock runtime was 309.5 seconds. Hardware and OS from `uname -a`: `Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64`.

Note: PatchTSTModel is not exposed by darts 0.41.0, so TSMixerModel is used as the transformer-family univariate substitute requested in the prompt.
