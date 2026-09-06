Done. The script `forecast_load.py` runs and produces `forecast_jan2020.png`.

## What it does

Trains a gradient-boosted regression (`HistGradientBoostingRegressor`) on all load history before 2020-01-01 (about 4 years of usable hours after the 1-year lag feature is formed), then forecasts the 168 hours of 1–7 Jan 2020 and compares against the actuals.

Because the target is a full week 168 hours ahead, the model uses only features that are genuinely known at forecast time (no leakage): calendar features (hour, day-of-week, month, weekend flag, German public-holiday flag, plus sin/cos encodings of hour and day-of-year) and three seasonal lags that are fully available from pre-2020 data (load 1 week, 2 weeks, and 364 days earlier). Holidays matter here because 1 Jan is a public holiday, which is the big dip on the left of the plot.

## How accurate it was

| metric | model | naive 1-week baseline |
|---|--:|--:|
| MAPE | **3.6%** | 12.8% |
| MAE | 1,932 MW | 7,238 MW |
| RMSE | 2,264 MW | 8,809 MW |
| max abs error | 5,306 MW | 22,403 MW |
| bias | +1,632 MW | −6,908 MW |

Mean load that week was about 53,400 MW, so the model is off by roughly 3.6% on average (about 1,900 MW) per hour. That is a strong result for load forecasting with no weather inputs. Its RMSE is 74% lower than a same-hour-last-week naive forecast, which does badly here because its reference week is the Christmas/New Year holiday period.

Honest caveats:
- The model has a small positive bias (+1,632 MW): it sits slightly above the actual on the daytime peaks (visible in the plot). Subtracting a constant offset would cut RMSE a little.
- No temperature data is available, so the biggest missing driver of load is absent. Adding weather would likely push MAPE below 3%.
- Epiphany (6 Jan) is a public holiday in only three German states, so it is treated as a normal Monday here; the small over-prediction on Mon 06 Jan is partly that.

## Running it

The system Python has no scientific stack, and this directory has a macOS provenance restriction that breaks venv/pip creation inside it, so I built the venv in your home directory:

```
$HOME/.energy_forecast_venv/bin/python forecast_load.py
```

That venv has pandas, numpy, scikit-learn, and matplotlib installed. The script reads `opsd_de_load.csv` and writes `forecast_jan2020.png` next to itself.