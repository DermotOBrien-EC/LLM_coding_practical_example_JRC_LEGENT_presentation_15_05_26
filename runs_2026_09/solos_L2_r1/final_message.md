## Result

The model forecast all 168 hours from **2020-01-01 00:00 UTC through 2020-01-07 23:00 UTC**.

| Metric | Result |
|---|---:|
| Mean absolute error | **1,872 MW** |
| Mean absolute percentage error | **3.46%** |
| Root mean squared error | **2,418 MW** |
| R² | **0.922** |
| Mean error | **+1,540 MW** |
| Maximum hourly error | **6,790 MW** |

A typical hourly forecast was within about **1.87 GW**, or **3.46%**, of the actual load. The positive mean error shows that the model tended to overforecast, particularly on 2–3 January.

For comparison, using the same hour from the previous year produced an MAE of **4,900 MW**. The trained model reduced that error by **61.8%**.

## Deliverables

- Script: [`forecast_load.py`](forecast_load.py)
- Forecast and hourly errors: [`outputs/forecast_results.csv`](outputs/forecast_results.csv)
- Complete metrics and runtime versions: [`outputs/forecast_metrics.json`](outputs/forecast_metrics.json)
- Forecast versus actual plot: [`outputs/forecast_vs_actual.png`](outputs/forecast_vs_actual.png)
- Dependencies: [`requirements.txt`](requirements.txt)
- Tests: [`test_forecast_load.py`](test_forecast_load.py)

Run with Python 3.12:

```bash
python3.12 -m pip install -r requirements.txt
python3.12 forecast_load.py
```

The Extra Trees model uses only information available before each forecast hour: the previous four weekly loads, the same hour one year earlier, calendar cycles, weekend status, and New Year indicators. Training ends at **2019-12-31 23:00 UTC**, so the actual target-week loads are not used during fitting.

## Assumptions

- Dates are interpreted as UTC because the dataset uses `utc_timestamp`. This is not German local time.
- Only historical load and calendar information were used. No weather forecasts were available.
- Forecast horizons are capped at 168 hours to prevent lag features from accidentally using observed values inside the forecast period.

## Verification

- Six unit tests pass, including an explicit target-leakage test.
- Ruff formatting and lint checks pass.
- Strict mypy passed using mypy 1.19.1. The workspace's mypy 1.20.2 crashed internally before analysis.
- Metrics were independently recomputed from the output CSV.
- A fresh environment installed from `requirements.txt` reproduced the CSV, JSON, and PNG byte-for-byte.
- An independent read-only Claude review initially found reproducibility, longer-horizon leakage, caption, baseline-framing, and test-coverage issues. These were fixed, and its second review found no substantive remaining issues. The Codex review command was not available in this session, so this was not represented as a Codex review.