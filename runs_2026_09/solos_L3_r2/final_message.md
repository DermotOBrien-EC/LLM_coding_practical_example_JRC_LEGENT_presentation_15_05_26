## Study complete

LightGBM won the untouched 168-hour test week:

| Rank | Model | Test MAPE |
|---:|---|---:|
| 1 | LightGBM | **5.105%** |
| 2 | Prophet | 7.614% |
| 3 | TSMixer | 10.246% |
| 4 | SARIMA | 12.219% |
| 5 | N-BEATS | 12.547% |
| 6 | Seasonal naive | 12.781% |

LightGBM also achieved:

- RMSE: **3,676.7 MW**
- MAE: **2,768.3 MW**
- Jan 1 MAPE: **4.498%**
- Jan 2 to Jan 7 MAPE: **5.207%**
- 60.1% lower MAPE than seasonal naive
- 80% interval coverage: **85.12%**
- 95% interval coverage: **100.0%**
- Pinball loss: **553.2 MW** at q=0.1, **1,379.7 MW** at q=0.5, **655.2 MW** at q=0.9

The 100% coverage suggests that the 95% interval was conservative or overwide for this single test week.

## Deliverables

- Full results: [`metrics.json`](metrics.json)
- Flattened results: [`metrics.csv`](metrics.csv)
- Methods, discussion, and recommendation: [`transcript.md`](transcript.md)
- All eight applicable 300 dpi figures: [`figures/`](figures/)
- Reproducible orchestration: [`code/forecast.py:66`](code/forecast.py#L66)

## Assumptions and substitutions

- Darts 0.41.0 does not expose `PatchTSTModel`, so the permitted `TSMixerModel` substitute was used.
- LightGBM forecasts recursively and uses an 8,736-hour, 364-day annual lag to preserve day-of-week alignment.
- SARIMA uses explicit 24-hour differencing with Burg-estimated autoregression, equivalent to the selected `(168,0,0)(0,1,0,24)` specification.
- N-BEATS and TSMixer used validation-monitored, capped training. Both selected the maximum allowed epoch count.
- No external data or test observations entered model fitting or selection.

Verification passed for schemas, finite metrics, JSON and CSV consistency, winner selection, conditional figures, exact figure dimensions, 300 dpi metadata, type-annotated function signatures, Ruff formatting and linting, Python compilation, and visual legibility. Strict mypy could not complete because mypy 1.20.2 raised an internal error. No packages were installed and no commit was created.