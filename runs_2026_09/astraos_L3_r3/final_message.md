Created the study: [transcript.md](transcript.md), [metrics.csv](metrics.csv), [metrics.json](metrics.json), all **eight figures**, and the complete code under `code/`.

### Results

| Model | Test MAPE | RMSE (MW) | MAE (MW) |
|---|---:|---:|---:|
| **LightGBM** | **3.91%** | **2,360** | **1,983** |
| Prophet + German holidays | 7.61% | 4,878 | 3,798 |
| Seasonal naive | 12.78% | 8,809 | 7,238 |
| Transformer, substituting for PatchTST | 12.99% | 8,513 | 6,357 |
| N-BEATS | 13.45% | 8,744 | 6,921 |
| SARIMA + weekly Fourier terms | 19.25% | 11,521 | 10,473 |

LightGBM reduced MAPE by **69.4%** versus seasonal naive and also led validation at **3.30%**. Its interval coverage was **75.6% at 80% nominal** and **95.8% at 95% nominal**.

The final full run took **8.7 minutes**. Eight tests passed, all metrics were recomputed from saved forecasts, and figure dimensions, 300-dpi resolution and deliverable checksums were verified.

### Important limitations

- Installed Darts lacks PatchTST, so the sixth model is explicitly labeled **TransformerModel**.
- The neural epoch cap changed on disk to **2** before fitting and was preserved. These are training-limited comparisons, not evidence about fully trained neural models.
- Independent Claude review did not complete; this is documented without claiming peer approval.
- Initial pytest/Ruff commands unintentionally created parent-directory caches. That deviation and subsequent local-cache controls are recorded in the [methods supplement](artifacts/methods.md).

Reproduce from this directory:

```sh
../../.venv/bin/python code/forecast.py
../../.venv/bin/python code/verify_outputs.py
```