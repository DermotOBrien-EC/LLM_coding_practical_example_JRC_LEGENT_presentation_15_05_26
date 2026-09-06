Completed the study in [transcript.md](transcript.md), with all model scripts, **eight 300-dpi figures**, and [JSON](metrics.json) / [CSV](metrics.csv) results.

### Results

| Model | Test MAPE | RMSE (MW) | MAE (MW) |
|---|---:|---:|---:|
| **LightGBM** | **3.96%** | **2,457** | **2,042** |
| Prophet + German holidays | 7.48% | 4,809 | 3,722 |
| Transformer* | 11.68% | 8,427 | 5,700 |
| N-BEATS | 11.92% | 8,395 | 5,855 |
| SARIMA | 12.21% | 8,603 | 6,954 |
| Seasonal naive | 12.78% | 8,809 | 7,238 |

LightGBM reduced MAPE by **69.0%** relative to the baseline. Its nominal **80% interval covered 86.31%** of observations; its nominal **95% interval covered 98.81%**. One holiday-heavy week does not establish production readiness or reliable interval calibration.

### Reproduce

```sh
../../.venv/bin/python code/forecast.py
```

A full independent rerun took **86 seconds** and produced **byte-identical forecasts across all 12 test/validation files**. Seven consumer tests passed, all metrics were independently recomputed, and the independent review approved the study. Details are in [review.md](review.md).

**Explicit choices:** every test forecast uses one fixed origin, with no observed test load entering recursive features. *Darts lacks PatchTST, so the sixth model is its actual `TransformerModel`, clearly labeled throughout.* The metric chart uses baseline-relative bars with raw-value labels to avoid mixing percentages and MW on one numerical axis.

No additional packages were installed and no external forecasting data were used.