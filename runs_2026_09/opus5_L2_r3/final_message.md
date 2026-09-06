Done. `forecast_de_load.py` (plus a test suite), with results in `output/`.

## How accurate it was

Forecasting all 168 hours in one shot from a single origin at midnight on 1 January, with nothing from inside the week fed back in:

| Model | MAPE | MAE | RMSE | Bias |
|---|---|---|---|---|
| **Harmonic regression** | **4.80%** | 2,585 MW | 3,291 MW | +1,789 MW |
| LightGBM | 4.95% | 2,669 MW | 3,207 MW | +2,649 MW |
| Seasonal naive (last week) | 12.69% | 7,187 MW | 8,773 MW | −6,857 MW |

Roughly 2.6 GW average error against a 37–71 GW load range. The week's peak (70,914 MW, Tue 7 Jan 17:00) was over-forecast by 6%.

## What the numbers mean

**This is a hard week, deliberately so.** The usual yardstick for load forecasting — "same hour, last week" — collapses to 12.7% here because last week was Christmas. That single fact drives everything: any model leaning on recent load is being fed the industrial shutdown as its picture of normal.

**That's why the simple model won.** I picked the model on held-out weeks *before* the forecast, and specifically on the 3 folds that also contained a public holiday, since an ordinary October week is a different problem. Harmonic regression won that comparison (3.92% vs LightGBM's 4.34%) and won again on the actual week. The gradient-boosted model has lag features pointing straight into the Christmas lull; the harmonic model uses calendar structure only — hour-of-day shape by day type, holiday flags, seasonal harmonics — so it is immune. Averaged over *all* 9 folds the two are indistinguishable (3.11% vs 3.15%); the gap only opens on holiday weeks.

**The errors are structured, not random.** The model over-forecasts almost the whole week (+1,789 MW bias, visible in the residual panel). It expects Germany back at work faster than it actually was: 1–3 January are the worst days (6.8%, 6.1%, 6.5%), and accuracy is best on the ordinary weekend (2.2% on Sunday 5th). Concretely, 3 January was a Friday sandwiched between New Year and a weekend — my `is_bridge_day` feature only fires when a holiday is directly adjacent, so it never flagged that Friday. Widening that to the whole New Year bridge stretch is the most obvious improvement available.

## Two things worth flagging

- **Window interpretation.** I read "1–7 January" as German local days (`Europe/Berlin`), so the window is 2019-12-31 23:00 UTC onward. The CSV is UTC, so a UTC reading would shift everything by one hour. Change with `--start`.
- **mypy caveat.** `mypy --strict` is clean, but `pandas-stubs` isn't in the dev dependencies, so pandas types resolve to `Any` and much of the checking over those calls is vacuous. I didn't add the dependency without asking.

The leakage boundary is the thing that decides whether any of this means anything, so it's pinned by tests: every lag is ≥168h, and poisoning all 168 in-window actuals leaves every feature value bit-identical. I also included a red-state proof that the check isn't vacuous — a 24h lag *does* pick the poison up. Nine tests pass; ruff clean.

One process note: my first run appeared dead (`ps` returned nothing) so I launched a second, and they competed for CPU for ~19 minutes. The first was alive the whole time. That cost wall-clock but not correctness.