from __future__ import annotations

import json
from typing import Any

import numpy as np

from .common import LABELS, ROOT, TEST_INDEX, ForecastResult, load_data, mape


def write_report(results: list[ForecastResult], summary: dict[str, Any]) -> None:
    manifest = json.loads((ROOT / "artifacts" / "run_manifest.json").read_text())
    by_name = {result.name: result for result in results}
    rows = summary["models"]
    winner = by_name[summary["winner"]]
    actual = load_data().loc[TEST_INDEX].to_numpy()
    validation_scores = {
        result.name: min(
            record["mape_validation_pct"]
            for record in result.validation
            if record.get("converged", True)
        )
        for result in results
    }
    validation_leader = min(validation_scores, key=validation_scores.get)
    baseline = next(row for row in rows if row["name"] == "naive")
    improvement = 100 * (1 - rows[0]["mape_test_pct"] / baseline["mape_test_pct"])
    table = [
        "| Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2-7 MAPE (%) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {LABELS[row['name']]} | {row['mape_test_pct']:.2f} | {row['rmse_test_mw']:,.0f} | {row['mae_test_mw']:,.0f} | {row['mape_jan1_pct']:.2f} | {row['mape_jan2_to_jan7_pct']:.2f} |"
        )
    validation_table = [
        "| Model | Selected setting | Validation MAPE (%) | Runtime (s) |",
        "|---|---|---:|---:|",
    ]
    for result in sorted(results, key=lambda item: validation_scores[item.name]):
        hp = result.hyperparameters
        setting = {
            "naive": "168-hour repeat",
            "sarima": f"{hp.get('order')} x {hp.get('seasonal_order')}",
            "prophet": f"changepoint prior {hp.get('changepoint_prior_scale')}",
            "lightgbm": f"{hp.get('num_leaves')} leaves, {hp.get('n_estimators')} trees",
            "nbeats": f"{hp.get('selected_epochs')} epochs; 30 stacks, width 64",
            "patchtst": f"{hp.get('selected_epochs')} epochs; 2+2 attention layers, width 32",
        }[result.name]
        validation_table.append(
            f"| {LABELS[result.name]} | {setting} | {validation_scores[result.name]:.2f} | {result.runtime_seconds:.1f} |"
        )
    failures = []
    for row in rows:
        result = by_name[row["name"]]
        per_day = [
            mape(actual[d * 24 : (d + 1) * 24], result.point[d * 24 : (d + 1) * 24])
            for d in range(7)
        ]
        worst = int(np.argmax(per_day))
        direction = (
            "underpredicted" if float(np.mean(actual - result.point)) > 0 else "overpredicted"
        )
        failures.append(
            f"{LABELS[result.name]} {direction} on average and was worst on Jan {worst + 1} ({per_day[worst]:.1f}%)"
        )
    mechanisms = {
        "lightgbm": "Calendar indicators and recent/annual load lags let it combine holiday information with nonlinear reuse of past patterns. This is a plausible explanation, not a causal attribution: no feature ablation was run.",
        "prophet": "Its calendar regression can depart from the preceding Christmas week's shape and explicitly represents German holidays. That is a plausible advantage, not proof that holidays caused the gain; no ablation was run.",
        "sarima": "Its fitted daily dynamics and weekly Fourier terms extrapolate a regular pattern beyond the previous week. This does not establish a holiday mechanism: the model has no holiday indicator.",
        "nbeats": "Its direct week-to-week mapping can adjust a past week's shape rather than repeat it. It has no explicit holiday calendar, so any holiday-related advantage is learned indirectly, not identified causally.",
        "patchtst": "Its attention-based forecast can transform a past week's shape. With no calendar inputs, a holiday-specific explanation would be speculative; this is not a result for PatchTST itself.",
        "naive": "Repeating the last observed week beat the fitted alternatives. Complexity is therefore not automatically rewarded; the expectation that every fitted model should beat the baseline is not a guarantee.",
    }
    if winner.quantiles is None:
        uncertainty = "The seasonal-naive winner has no probabilistic model. Coverage and pinball loss are null, not fabricated; Figure 05 explicitly marks intervals unavailable."
    else:
        uncertainty = (
            f"Winner interval coverage: **{summary['winner_coverage_80pct']:.1%} at 80% nominal**, "
            f"**{summary['winner_coverage_95pct']:.1%} at 95% nominal**. "
            f"Pinball losses at q=0.1/0.5/0.9: **{summary['winner_pinball_loss_q10']:,.1f} / "
            f"{summary['winner_pinball_loss_q50']:,.1f} / {summary['winner_pinball_loss_q90']:,.1f} MW** (lower is better)."
        )
    cap = by_name["nbeats"].hyperparameters["maximum_epochs"]
    training_note = f"The live neural cap was {cap} epochs; the best epoch was chosen on validation and refitted from scratch."
    if cap < 6:
        training_note += " This cap is too short for the configured five-epoch patience to trigger; the neural results are budget-limited, not converged comparisons."
    text = f"""# German hourly load: a six-model week-ahead bake-off

## 1. Data

Open Power System Data, derived from ENTSO-E Transparency, supplies **50,400** German national hourly load observations (MW), from 2015-01-01 00:00 to 2020-09-30 23:00 UTC. Checks confirmed exact endpoints, consecutive unique hours, positive finite loads and no missing values; no imputation was needed. Later 2020 data appear only in the overview. No external load drivers were used.

## 2. Why these six models

Seasonal naive repeats the preceding week. SARIMA adds daily dynamics and timestamp-derived weekly Fourier terms. Prophet adds daily, weekly and yearly patterns plus German federal holidays. LightGBM learns nonlinear calendar/lag relationships. N-BEATS learns a direct week-to-week mapping; a transformer adds attention-based sequence processing. **Darts 0.41.0 has no PatchTSTModel**, so TransformerModel substitutes in that slot (JSON key `patchtst`). TSMixer was not used because it is an MLP, not a transformer.

## 3. Validation strategy

Train: **41,616** hours through September 2019; validation: **2,208** hours in October-December; test: **168** hours on January 1-7, 2020. Validation scores thirteen consecutive week-ahead forecasts plus a final 24-hour block. Previous validation observations supply later-origin context, never fitting or gradients. SARIMA filters with all parameters fixed; Prophet remains a fixed calendar regression. Every selected configuration is refitted on **43,824** pre-test hours, maximizing available history without exposing test outcomes.

All test forecasts use a single January 1 origin. LightGBM recursively substitutes predictions for unknown loads; rolling features end at t-1. Neural contexts/outputs are 168 hours, with training-only scaling and 24-hour training stride. {training_note} Candidate grids, selected settings, validation scores and interval construction are in [the protocol supplement](artifacts/methods.md).

## 4. Results

Sorted by held-out MAPE. **Jan 2-7 includes the weekend**, not just working days; federal holidays and daily metrics use UTC calendar dates.

{chr(10).join(table)}

{uncertainty}

Five quantiles support 80% and 95% bands for every fitted model. LightGBM's recursive marginal intervals do not propagate lag uncertainty; no intervals receive test-based calibration. “Winner” describes this week, not a production model selected by test performance.

## 5. Discussion

{LABELS[winner.name]} won with {rows[0]["mape_test_pct"]:.2f}% MAPE, a {improvement:.1f}% reduction relative to seasonal naive. {mechanisms[winner.name]} This is an unusual benchmark: naive copies December 25-31, so the test mixes New Year's Day with a return from the Christmas period. The ranking may differ in ordinary winter weeks.

{"; ".join(failures)}. The holiday breakdown and daily heatmap show whether the overall score hides a calendar-day failure. High Jan 1 error was a hypothesis, not an imposed outcome. Naive and SARIMA stay too low as demand resumes; the neural forecasts fail to suppress the New Year's Day peak. Prophet's holiday indicator helps distinguish calendars but does not eliminate its Jan 1 mismatch.

Theory suggests that holiday calendars can help at calendar breaks and that neural models need sufficient training. It does not imply a universal complexity ranking. Here the short neural budget and narrowed layers are part of the measured configuration, not evidence about the best attainable performance of either model class. The validation origins start on a different weekday from test, but every model shares the same forecast protocol.

There is no weather information, multi-seed analysis or repeated test-week experiment. Hourly errors are dependent: 168 hours are not 168 independent replications. Interval coverage on one holiday week cannot establish calibration, and correlated lag features make split-gain importance descriptive rather than causal. Neither test rank nor a plausible mechanism justifies deployment alone.

## 6. Recommendation

For JRC production qualification I would choose **{LABELS[validation_leader]}**, the validation leader ({validation_scores[validation_leader]:.2f}% MAPE), rather than select a production class by this test week. First evaluate untouched weeks across seasons/holidays, publication delays and daylight-saving boundaries, and calibrate horizon-specific intervals. Repeat neural seeds where relevant. Weather augmentation would be a separate experiment, outside this univariate study.

## 7. Reproducibility note

From this directory, without installing packages:

```sh
../../.venv/bin/python code/forecast.py
../../.venv/bin/python code/verify_outputs.py
```

Seed **2026**; CPU execution, four compute threads. Successful-run wall time: **{summary["total_runtime_seconds"]:.1f} s ({summary["total_runtime_seconds"] / 60:.1f} min)**, including validation, refits, sampling and initial rendering. Hardware:

`{manifest["uname"]}`

Versions and input hash: `artifacts/run_manifest.json`. Saved point/quantile forecasts allow independent rescoring. `--render-only` regenerates figures/report without training. Eight consumer tests passed; independent model review did not complete (see the supplement). Figure 03 uses baseline-normalized bars with raw %/MW labels, avoiding mixed units; all figures use a fixed tab10 subset. Optional figures follow the requested top-two rule.
"""
    supplement = f"""# Protocol, validation and engineering supplement

## Selected configurations

{chr(10).join(validation_table)}

The validation leader is {LABELS[validation_leader]}. The baseline has no tunable parameters. SARIMA compares (1,0,0) and (2,0,0), each with (1,1,0,24), plus three weekly Fourier harmonics derived from UTC timestamps. Daily differencing alone does not create weekly structure. The residual variance is explicitly estimated in training and frozen, like every other parameter, during validation filtering. Prophet compares changepoint priors 0.01 and 0.05 with additive daily/weekly/yearly components and German federal holidays. LightGBM compares 15 leaves/250 trees and 31 leaves/400 trees (learning rate 0.05).

N-BEATS retains the default generic architecture: 30 stacks, one block per stack and four layers per block; layer width is reduced from 256 to 64 for the CPU budget. TransformerModel uses width 32, four heads, two encoder and two decoder layers, feed-forward width 64 and dropout 0.1. Neural training uses Adam at 0.001, batch size 32 and daily-strided examples covering all weekdays. Quantile loss trains the models; full rolling validation MAPE selects epochs. Patience is five epochs with minimum improvement 0.01 percentage points; the lowest recorded MAPE determines the refit epoch count. {training_note} The originally planned cap was 30, but the file changed to 2 before neural fitting; the current value was preserved. No test score motivated this change.

## Features and forecast isolation

LightGBM uses UTC hour, weekday, month, weekend and public-holiday indicators; lags 24/168/8760 hours; and means/population standard deviations for the preceding 24/168 hours, excluding the target hour. It loses 8,760 supervised rows to annual-lag warm-up, but retains them as history. Unknown loads are replaced only by the predicted median, never actuals. The prompt's annual-lag timestamp is incorrect: 2020-01-01 00:00 UTC minus 8,760 hours is 2019-01-01 00:00 UTC. Jan 6 is not a nationwide German holiday. No external covariates were obtained; weather is discussed solely as future work.

Validation parameters and neural scalers are trained on data through September 30 only. The final 24-hour remainder is scored with 24 hours of weight; all other blocks are 168 hours. Neural forecasts for that last origin are generated for 168 hours but only the 24 validation hours are scored. Validation observations enter later prediction contexts, not fitting. Final refits use all pre-test history, except feature warm-up and daily-strided neural sampling as explicitly described. Test scoring happens after all six final forecasts have been saved.

## Probabilistic forecasts

SARIMA uses analytic Gaussian state-space errors conditional on estimated parameters. Prophet uses 1,000 predictive samples from its MAP fit, covering predictive noise and simulated trend changes rather than full posterior parameter uncertainty. Both neural models directly fit marginal quantiles 0.025, 0.1, 0.5, 0.9, 0.975. LightGBM fits those same five quantiles for every validation candidate and for the final refit, so the point-forecast rule is identical at selection and evaluation. Learned quantiles are monotonically rearranged before scoring. Its intervals condition on a recursive median path and do not propagate lag uncertainty. Neural/LightGBM point forecasts are the rearranged median; SARIMA/Prophet use their native conditional mean. The naive baseline has no native probabilistic model.

## Reproduction and checks

```sh
../../.venv/bin/python -m pytest code/test_study.py --import-mode=importlib --rootdir=. -o cache_dir=.cache/pytest -q
../../.venv/bin/python code/verify_outputs.py
../../.venv/bin/python code/forecast.py --render-only
```

`artifacts/*_run.json` retain every candidate or epoch score and exact settings. Per-model runtimes exclude import overhead; total runtime includes imports within the orchestrator plus initial figure/report generation, but not the final small timing-file refresh. Repeated runs overwrite the generated outputs. Hardware, BLAS and software versions can change timings and small floating-point details. The code uses a local namespace so the required `code/prophet.py` never shadows the external Prophet package. The forecasting entrypoint directs caches and temporary files into this directory; no packages are installed. An initial pytest invocation and early Ruff commands inherited the parent project's configuration and created `../../.pytest_cache` and `../../.ruff_cache` outside the requested directory. This was unintended; those caches were left untouched. Subsequent pytest commands explicitly set the local root/cache, and subsequent Ruff commands use `--cache-dir .cache/ruff`.

Tests were written first and initially failed because the implementation was absent. They cover split counts, every validation hour scored once, metric formulas, strictly shifted rolling features, recursive unobserved-lag substitution, no-gradient deep validation, frozen SARIMA variance and identical LightGBM quantile sets in validation and final fitting. The deep callback is also compared with public Darts.predict after training. The independent output checker recomputes six metric rows, winner coverage/pinball losses, quantile ordering, CSV agreement, PNG dimensions/DPI and input SHA-256 from saved files without importing the implementation's scoring functions.

Two runs were stopped before test scoring: the first lacked SARIMA filtering states with low_memory=True (`attempt1.log`); the second exposed residual-scale re-estimation under concentrated-scale validation filtering (`attempt2.log`). The corrected run retains states and holds an explicit sigma2 parameter fixed. A subsequent run completed, but a post-run structural audit found that sorting five fitted quantiles can change LightGBM's median while its validation candidates had fitted only q=0.5. A failing consumer test pinned the mismatch (`quantile_protocol_red.log`); the repair fits all five quantiles during validation too, then reruns all six models. This correction follows protocol identity, not test-error optimization. The superseded result is retained under `artifacts/pre_quantile_protocol_fix/`. No data were changed. The requested independent Claude review produced no findings before it was stopped, so cross-vendor review remains incomplete; `artifacts/review.md` records this without claiming approval.

## Figure encoding

Time-series figures are 11 x 6 inches; square comparisons are 6 x 6 inches, all at 300 dpi. A reordered tab10 subset passed the palette validator's adjacent-pair checks. Direct names and table values provide contrast relief, and forecast panels are faceted rather than relying on six color-only overlaid lines. Heatmap colors encode MAPE magnitude, not model identity. Metric bars show ratios to each metric's own seasonal-naive value, with raw %/MW labels and separate hatches for the three measures. Feature importance is share of total split gain, not causal importance.
"""
    for output in [text, supplement]:
        if chr(0x2014) in output:
            raise ValueError("Prose contains a prohibited em dash")
    (ROOT / "transcript.md").write_text(text)
    (ROOT / "artifacts" / "methods.md").write_text(supplement)
