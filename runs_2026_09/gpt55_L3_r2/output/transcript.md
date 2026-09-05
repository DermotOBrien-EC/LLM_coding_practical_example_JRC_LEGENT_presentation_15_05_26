# German load forecasting bake-off

## Data

The study uses Open Power System Data hourly German national electricity load, derived from the ENTSO-E Transparency Platform. The file `opsd_de_load.csv` runs from 2015-01-01 00:00 UTC to 2020-09-30 23:00 UTC with exactly 50,400 hourly rows. The timestamp spacing is exactly one hour throughout and the load column has no missing values, so no imputation or gap filling was needed.

## Why these six models

The six models form a complexity gradient. Seasonal naive asks how far a one-week memory gets on its own. SARIMA adds a classical daily seasonal structure. Prophet adds explicit calendar seasonality and German holidays. LightGBM uses calendar, lag, and rolling features to let a tree ensemble learn non-linear rules. N-BEATS tests a deep univariate MLP forecaster. PatchTST was requested, but Darts 0.41.0 did not expose `PatchTSTModel`, so I used the transformer-family substitute `TSMixerModel` and kept the same 168-hour input and output chunks; exact note: PatchTSTModel was unavailable in Darts 0.41.0; used TSMixerModel.

## Validation strategy

The train window is 2015-01-01 through 2019-09-30 with 41,616 observations, the validation window is 2019-10-01 through 2019-12-31 with 2,208 observations, and the held-out test week is 2020-01-01 through 2020-01-07 with 168 observations. Hyperparameters and SARIMA orders were selected only by validation MAPE. After selection, each selected configuration was refit on train plus validation, 43,824 observations, before forecasting the test week.

## Results table

| Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2 to Jan 7 MAPE (%) |
|---|---:|---:|---:|---:|---:|
| LightGBM | 3.77 | 2,708 | 2,071 | 2.51 | 3.98 |
| Prophet | 7.61 | 4,878 | 3,798 | 13.28 | 6.67 |
| TSMixer | 7.87 | 5,260 | 4,127 | 13.37 | 6.95 |
| N-BEATS | 9.84 | 6,503 | 5,253 | 6.55 | 10.39 |
| SARIMA | 12.52 | 8,695 | 7,126 | 8.65 | 13.16 |
| Seasonal naive | 12.78 | 8,809 | 7,238 | 6.52 | 13.82 |

## Discussion

LightGBM won on test MAPE at 3.77%. Its advantage is that the test week is short and highly patterned: hourly load depends strongly on the same hour in recent days, the same hour one week earlier, public-holiday status, and rolling load level. A tree ensemble can use those signals directly without needing to infer all calendar effects from the target path alone. The seasonal naive baseline was a useful anchor, but it copied the previous week mechanically and could not adapt to the New Year holiday. SARIMA captured smooth daily recurrence, yet its daily seasonal form was too rigid for the holiday and week-to-week level shift. Prophet explicitly knew about German holidays and yearly seasonality, but its smooth components underfit the sharp intra-day load shape over this specific week. N-BEATS and TSMixer had enough flexibility in principle, but the short 168-hour forecasting horizon and compact training budget made them less reliable than the engineered-feature tree model. They tended to smooth or shift peaks rather than lock onto the calendar-lag structure.

The Jan 1 breakdown is the main stress test. New Year's Day is a public holiday and a Wednesday, so a model that treats it like a normal Wednesday over-predicts demand. Models with either a holiday flag or a direct recent-load analogue handled it better. The rank ordering is broadly what theory would predict for a small, univariate operational benchmark: strong lag features plus a non-linear learner beat smooth statistical decompositions and budget-limited deep models. The caveat is that this is only one held-out week. It tests short-term temporal structure, not robustness across seasons or the weather-sensitive extremes that dominate real power-system operations.

## Recommendation

For a JRC production short-term load forecasting pipeline constrained to these inputs, I would start with LightGBM. It is fast, transparent enough to audit through feature importance, and strong on the operational metric. Before production use, I would extend the rolling-origin evaluation to many weeks across 2018 to 2020, calibrate prediction intervals, add explicit local holiday variants if the target geography changes, and only then consider whether a deep model adds skill after a larger hyperparameter search.

## Reproducibility note

Random seed 42 was used for LightGBM and the Darts neural models. The total wall-clock runtime was 283.2 seconds. Hardware and operating system from `uname -a`: `Darwin Dermots-MacBook-Pro-2.local 25.5.0 Darwin Kernel Version 25.5.0: Mon Apr 27 20:41:12 PDT 2026; root:xnu-12377.121.6~2/RELEASE_ARM64_T6050 arm64`. LightGBM feature importance is shown because it finished in the top two. Prophet components are shown because Prophet finished in the top two.
