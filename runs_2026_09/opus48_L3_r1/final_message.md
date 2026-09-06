The bake-off is complete. Every deliverable is present, schema-valid, and verified.

## Result

| Rank | Model | Test MAPE | RMSE (MW) | Jan 1 | Jan 2–7 |
|---|---|---|---|---|---|
| 1 | **LightGBM** | **4.83%** | 3079 | 4.06% | 4.96% |
| 2 | Prophet | 7.61% | 4877 | 13.25% | 6.67% |
| 3 | N-BEATS | 9.68% | 6464 | 5.63% | 10.36% |
| 4 | SARIMA | 11.00% | 7592 | 11.32% | 10.94% |
| 5 | PatchTST (Transformer) | 12.30% | 7893 | 6.19% | 13.32% |
| 6 | Seasonal-naive | 12.78% | 8809 | 6.52% | 13.82% |

**LightGBM wins.** All five non-trivial models beat the naive baseline. Winner's interval coverage: 76% (nominal 80) and 98% (nominal 95); pinball q10/q50/q90 = 403/1033/333.

## Key decisions and one honest nuance
- **PatchTST substitution:** `darts.models.PatchTSTModel` does not exist in the installed darts 0.41.0, so I substituted `darts.models.TransformerModel` (a genuine encoder-decoder transformer). Recorded in `transcript.md` and the hyperparameters.
- **LightGBM forecasts recursively** (feeds its own predictions back as lags) so it never uses a test-window actual as a feature — the honest apples-to-apples comparison against the deep models' 168-step direct forecast.
- **The holiday story is subtler than "calendar-blind models fail."** The heatmap showed the naive is accidentally *fine* on Jan 1 (168h earlier was Dec 25, itself a holiday), while Prophet is *worst* on Jan 1 by over-correcting downward despite modelling holidays. I rewrote the discussion and the figure-4 title to reflect this rather than the stock narrative.

## Two bugs I hit and fixed
- **SARIMA** initially returned 100% MAPE: `simple_differencing=True` makes statsmodels forecast the *differenced* series (values near 0), not load levels. Removed it; SARIMA now gives a sensible 11%.
- **Segfault** in the orchestrator from LightGBM and PyTorch both loading libomp in one process (macOS). Fixed by running each model in an isolated subprocess; the parent/assembly process imports no ML libraries.

## Structure
`code/` (common + 6 model modules + `forecast.py`), `figures/` (all 8 PNGs at 300 dpi — 07 and 08 both produced since LightGBM and Prophet are top-2), `metrics.json`, `metrics.csv`, `transcript.md`. Per-model results cache under `code/_cache/` (documented; delete to force a clean refit). Total fit runtime ~1816s (~30 min), dominated by the transformer (947s) and SARIMA (453s).

Run `../../.venv/bin/python code/forecast.py` from this directory to reproduce.

One caveat worth flagging: the deep models (N-BEATS, transformer) train in float32 on the Apple MPS backend, whose reductions are not bit-for-bit deterministic, so their two numbers can shift slightly between runs even at a fixed seed. The other four models are fully deterministic.