from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from time import perf_counter
from typing import Any
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from .common import (
    COLORS,
    LABELS,
    NAMES,
    ROOT,
    TEST_START,
    TEST_END,
    ModelResult,
    figure_style,
    load_data,
    mape,
    probability_scores,
    read_result,
    save_figure,
    score,
    write_json,
)

SHORT = {**LABELS, "prophet": "Prophet + holidays", "patchtst": "Transformer substitute"}


def date_axis(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axis.set_xlabel("Date (UTC)")
    axis.grid(axis="y")


def make_figures(
    data: pd.Series,
    results: list[ModelResult],
    rows: list[dict[str, Any]],
    daily: pd.DataFrame,
    probability: dict[str, float | None],
) -> None:
    figure_style()
    lookup = {r.name: r for r in results}
    test = data.loc[(data.index >= TEST_START) & (data.index < TEST_END)]
    times, actual = test.index, test.to_numpy()
    winner_name = rows[0]["name"]
    winner = lookup[winner_name]
    color = COLORS[winner_name]
    fig, axis = plt.subplots(figsize=(11, 6), layout="constrained")
    axis.plot(data.index, data.values, color="#c8c8c8", lw=0.35, label="Observed national load")
    axis.axvspan(TEST_START, TEST_END, color="#d62728", alpha=0.30)
    axis.plot(times, actual, color="#d62728", lw=1.0, label="Held-out test week")
    axis.annotate(
        "Test: 1–7 January 2020",
        xy=(TEST_START, float(actual.max())),
        xytext=(pd.Timestamp("2017-10-01"), float(data.max()) + 1500),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
    )
    axis.set(
        title="German national electricity load: study context",
        xlabel="Date (UTC)",
        ylabel="Load (MW)",
    )
    axis.set_ylim(float(data.min()) - 3000, float(data.max()) + 7500)
    axis.legend(loc="lower left")
    axis.grid(axis="y")
    save_figure(fig, "01_overview.png")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True, layout="constrained")
    bounds = np.concatenate([actual] + [r.point for r in results])
    low, high = float(bounds.min() - 1500), float(bounds.max() + 1500)
    scores = {row["name"]: row["mape_test_pct"] for row in rows}
    for axis, result in zip(axes.flat, results):
        axis.axvspan(TEST_START, TEST_START + pd.Timedelta(days=1), color="#eeeeee", zorder=0)
        axis.plot(times, actual, color="black", label="Observed", lw=1.1)
        axis.plot(times, result.point, color=COLORS[result.name], label="Forecast", lw=1.4)
        axis.set_title(f"{SHORT[result.name]}\nMAPE {scores[result.name]:.2f}%")
        axis.set_ylim(low, high)
        axis.legend(loc="upper left", fontsize=7, ncol=2)
        date_axis(axis)
    for axis in axes[:, 0]:
        axis.set_ylabel("Load (MW)")
    fig.suptitle("Seven-day forecasts from 1 January 2020 00:00 UTC", fontsize=13)
    save_figure(fig, "02_forecast_comparison.png")

    fig, axis = plt.subplots(figsize=(6, 6), layout="constrained")
    baseline = next(row for row in rows if row["name"] == "naive")
    keys = ["mape_test_pct", "rmse_test_mw", "mae_test_mw"]
    hatches = ["", "///", "..."]
    maximum = 0.0
    for rank, row in enumerate(rows):
        for j, key in enumerate(keys):
            height = row[key] / baseline[key] * 100
            maximum = max(maximum, height)
            y = rank + (j - 1) * 0.23
            axis.barh(
                y,
                height,
                height=0.20,
                color=COLORS[row["name"]],
                hatch=hatches[j],
                edgecolor="white",
                linewidth=0.8,
            )
            label = f"{row[key]:.2f}%" if j == 0 else f"{row[key]:,.0f} MW"
            axis.annotate(
                label,
                (height, y),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
            )
    axis.axvline(100, color="#777777", lw=0.7, zorder=0)
    axis.set_yticks(range(6), [SHORT[row["name"]] for row in rows], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0, maximum * 1.38)
    axis.set_xlabel("Score (% of seasonal-naive score)")
    axis.set_ylabel("Model (sorted by test MAPE)")
    axis.set_title("Test error comparison\nBar labels show original units; lower is better")
    axis.legend(
        handles=[
            Patch(facecolor="#888888", edgecolor="white", hatch=h, label=metric_label)
            for h, metric_label in zip(hatches, ["MAPE", "RMSE", "MAE"])
        ],
        loc="upper right",
        fontsize=8,
    )
    axis.grid(axis="x")
    axis.set_axisbelow(True)
    save_figure(fig, "03_metric_comparison.png")

    fig, axis = plt.subplots(figsize=(6, 6), layout="constrained")
    matrix = daily[[row["name"] for row in rows]].to_numpy().T
    image = axis.imshow(matrix, aspect="auto", cmap="Blues", vmin=0)
    axis.set_xticks(
        range(7),
        ["Jan 1*", "Jan 2", "Jan 3", "Jan 4", "Jan 5", "Jan 6", "Jan 7"],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(range(6), [SHORT[row["name"]] for row in rows], fontsize=8)
    for i in range(6):
        for j in range(7):
            axis.text(
                j,
                i,
                f"{matrix[i, j]:.1f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() * 0.58 else "#222222",
                fontsize=9,
            )
    fig.colorbar(image, ax=axis, shrink=0.7, label="MAPE (%)")
    axis.set(
        title="Daily forecast error\n*1 January: federal public holiday",
        xlabel="Day in test week (UTC)",
        ylabel="Model (sorted by test MAPE)",
    )
    save_figure(fig, "04_per_day_mape.png")

    fig, axis = plt.subplots(figsize=(11, 6), layout="constrained")
    if winner.quantiles is not None:
        q = winner.quantiles
        axis.fill_between(
            times, q[:, 0], q[:, 4], color=color, alpha=0.12, label="95% prediction interval"
        )
        axis.fill_between(
            times, q[:, 1], q[:, 3], color=color, alpha=0.27, label="80% prediction interval"
        )
        caption = (
            f"MAPE {rows[0]['mape_test_pct']:.2f}%; observed coverage: "
            f"80% band {100 * probability['winner_coverage_80pct']:.1f}%, "
            f"95% band {100 * probability['winner_coverage_95pct']:.1f}%"
        )
    else:
        caption = "Seasonal naive won; no prediction intervals are defined for this baseline."
    axis.plot(times, actual, color="black", label="Observed")
    axis.plot(times, winner.point, color=color, label="Point forecast")
    axis.set(title=f"{SHORT[winner_name]}: forecast and uncertainty\n{caption}", ylabel="Load (MW)")
    date_axis(axis)
    axis.legend(loc="upper left", ncol=2)
    save_figure(fig, "05_winner_with_intervals.png")

    residual = actual - winner.point
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), layout="constrained")
    for axis, groups, labels, xlabel in [
        (
            axes[0],
            [residual[times.hour == h] for h in range(24)],
            [str(h) if h % 3 == 0 else "" for h in range(24)],
            "Hour of day (UTC)",
        ),
        (
            axes[1],
            [residual[i * 24 : (i + 1) * 24] for i in range(7)],
            ["Wed 1", "Thu 2", "Fri 3", "Sat 4", "Sun 5", "Mon 6", "Tue 7"],
            "Day of test week (UTC)",
        ),
    ]:
        boxes = axis.boxplot(
            groups,
            tick_labels=labels,
            patch_artist=True,
            widths=0.55,
            medianprops={"color": "black", "linewidth": 1.0},
            flierprops={"markersize": 2, "markeredgecolor": "#777777"},
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        axis.axhline(0, color="#444444", lw=0.8)
        axis.set(xlabel=xlabel, ylabel="Observed minus forecast (MW)")
        axis.grid(axis="y")
        if axis is axes[1]:
            axis.tick_params(axis="x", rotation=35)
    fig.suptitle(f"{SHORT[winner_name]} residuals: positive means underprediction", fontsize=13)
    save_figure(fig, "06_residuals.png")

    top_two = {row["name"] for row in rows[:2]}
    if "lightgbm" in top_two:
        gains = pd.Series(lookup["lightgbm"].diagnostics["feature_gain"]).sort_values(
            ascending=True
        )
        fig, axis = plt.subplots(figsize=(6, 6), layout="constrained")
        axis.barh(
            gains.index, gains.to_numpy() / gains.sum() * 100, color=COLORS["lightgbm"], height=0.65
        )
        axis.set(
            title="LightGBM: training feature importance",
            xlabel="Share of total split gain (%)",
            ylabel="Engineered feature",
        )
        axis.grid(axis="x")
        axis.set_axisbelow(True)
        save_figure(fig, "07_feature_importance.png")
    if "prophet" in top_two:
        fig, axes = plt.subplots(3, 1, figsize=(11, 6), layout="constrained")
        for axis, name in zip(axes, ["trend", "weekly", "yearly"]):
            frame = pd.read_csv(ROOT / "artifacts" / f"prophet_{name}.csv", parse_dates=["ds"])
            axis.plot(frame.ds, frame[name], color=COLORS["prophet"])
            axis.set(ylabel=f"{name.capitalize()} (MW)", xlabel="Date (UTC)")
            axis.grid(axis="y")
            if name == "weekly":
                axis.xaxis.set_major_formatter(mdates.DateFormatter("%a"))
            if name == "yearly":
                axis.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        fig.suptitle("Prophet + DE holidays: fitted additive components", fontsize=13)
        save_figure(fig, "08_decomposition.png")


def transcript(
    rows: list[dict[str, Any]],
    results: list[ModelResult],
    daily: pd.DataFrame,
    metrics: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    lookup = {result.name: result for result in results}
    winner = rows[0]
    name = winner["name"]
    baseline = next(row for row in rows if row["name"] == "naive")
    reduction = 100 * (1 - winner["mape_test_pct"] / baseline["mape_test_pct"])
    table = [
        "| Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2–7 MAPE (%) |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for row in rows:
        table.append(
            f"| {SHORT[row['name']]} | {row['mape_test_pct']:.2f} | {row['rmse_test_mw']:,.0f} | {row['mae_test_mw']:,.0f} | {row['mape_jan1_pct']:.2f} | {row['mape_jan2_to_jan7_pct']:.2f} |"
        )
    hypotheses = {
        "lightgbm": "Its combination of calendar rules, federal holidays and load history can describe changes that a repeated week cannot. That is a plausible explanation, not a measured causal attribution: no feature ablation was run.",
        "prophet": "Its explicit holiday and calendar terms offer a plausible advantage around New Year, without carrying the preceding Christmas week's load directly forward. No ablation was run to identify which term explains the advantage.",
        "sarima": "Its daily dependence and explicit weekly Fourier terms appear sufficient for this particular week. This is not evidence that a linear model generally dominates nonlinear models.",
        "nbeats": "Its learned mapping from the preceding week's shape produced the smallest errors here, despite having no holiday indicator. This does not establish that it learned holiday effects rather than other recurring patterns.",
        "patchtst": "The compact transformer captured useful structure in the preceding week's load. This result concerns Darts' TransformerModel, not an actual PatchTST architecture.",
        "naive": "Repeating the preceding week outperformed every fitted alternative here. Greater model complexity did not guarantee better generalization on this small test.",
    }
    failures = []
    for row in rows:
        worst = int(np.argmax(daily[row["name"]].to_numpy()))
        failures.append(
            f"{SHORT[row['name']]} was weakest on Jan {worst + 1} ({daily[row['name']].iloc[worst]:.1f}% MAPE)."
        )
    holiday_better = sum(row["mape_jan1_pct"] > row["mape_jan2_to_jan7_pct"] for row in rows)
    fit_seconds = provenance["fit_and_forecast_wall_seconds"]
    q = metrics
    if lookup[name].quantiles is None:
        uncertainty = "The winning naive baseline has no predictive distribution: coverage and pinball entries are null, not invented."
    else:
        uncertainty = (
            f"For the winner, nominal 80% and 95% intervals cover {100 * q['winner_coverage_80pct']:.1f}% and "
            f"{100 * q['winner_coverage_95pct']:.1f}% of the 168 observations. Pinball losses at q=0.1/0.5/0.9 are "
            f"{q['winner_pinball_loss_q10']:,.1f}/{q['winner_pinball_loss_q50']:,.1f}/{q['winner_pinball_loss_q90']:,.1f} MW (lower is better)."
        )
    convergence = lookup["sarima"].diagnostics["refit_converged"]
    validation_table = ["| Model | Selected validation MAPE (%) | Selection |", "|:--|--:|:--|"]
    for result in results:
        if result.validation:
            value = min(
                item["mape_pct"] for item in result.validation if item.get("converged", True)
            )
            if result.name in ("nbeats", "patchtst"):
                choice = f"epoch {result.hyperparameters['selected_epochs']}"
            elif result.name == "sarima":
                choice = "best converged candidate"
            else:
                choice = "best of two fixed candidates"
            validation_table.append(f"| {SHORT[result.name]} | {value:.2f} | {choice} |")
    return f"""# German hourly load: six-model forecasting bake-off

## 1. Data

The supplied Open Power System Data / ENTSO-E Transparency series contains
50,400 hourly load observations (MW), from 2015-01-01 00:00 to
2020-09-30 23:00 UTC. Exact grid, uniqueness and finite positive values were
verified: no imputation or resampling was needed. Inputs are load and known
calendar features only. Later-2020 data appear only in the overview.

## 2. Why these six models

Weekly naive is the reference. SARIMA adds linear daily dependence and three
weekly Fourier pairs derived from weekday/hour. Prophet adds daily, weekly,
yearly and federal-holiday effects. LightGBM captures nonlinear calendar,
lag and rolling-feature relationships; default-stack N-BEATS learns a
week-to-week mapping. The sixth model tests attention-based forecasting:
Darts 0.41.0 lacks `PatchTSTModel`, so its compact `TransformerModel` was
substituted before scoring. The available TSMixer is not a transformer.

## 3. Validation strategy

Train has 41,616 hours through 2019-09-30; October–December validation has
2,208. Train-fitted candidates forecast 13 seven-day blocks plus 24 hours,
weighted equally per hour. Pre-origin observations may update context or
filter state, never coefficients. Prophet's calendar curve needs no state
update. Every selected model is then fitted afresh on all 43,824 pre-test
hours, maximizing usable history. The 168-hour test forecast is issued at
2020-01-01 00:00 UTC: no within-test observations enter any model's features
or selection. Test ranks are descriptive, not a second selection round.

{chr(10).join(validation_table)}

## 4. Results

{chr(10).join(table)}

Jan 2–7 is the **remainder of the week**, not six working days: it includes
Saturday and Sunday. Jan 6 is a regional, not nationwide, German holiday;
no subdivision-specific holiday inputs were added. All date partitions and
calendar features use UTC, not Europe/Berlin local time.

{uncertainty}
Coverage values in JSON/CSV are fractions (0–1); nominal levels are 0.80 and
0.95. Other models' coverage, widths and pinball scores are available in
`artifacts/probabilistic_metrics.csv`. Intervals are pointwise, not a joint
95% guarantee for the whole week.

## 5. Discussion

{SHORT[name]} wins this week with {winner["mape_test_pct"]:.2f}% MAPE,
a {reduction:.1f}% reduction relative to seasonal naive. {hypotheses[name]}

{" ".join(failures)}
Across the six models, {holiday_better} have higher MAPE on Jan 1 than on the
remaining days. Whether Jan 1 is harder is a measured per-model result,
not an assumption.
The seasonal-naive reference copies Dec 25–31, including Christmas; its
errors reflect both the forecast week's calendar and the unusual reference
week. SARIMA has no holiday indicator, while neither neural model receives
calendar covariates, so their treatment of exceptional days relies on load
history alone. Prophet's additive effects may miss changing load-profile
shape; recursive LightGBM can accumulate errors as forecast values replace
observed lags. The per-day panel localizes these failures but does not prove
their causes.

The ordering is not a universal complexity ranking. These are small,
prespecified searches with different inductive assumptions and equal data
cutoffs, not equally exhaustive optimization budgets. Daily-strided neural
training uses all years but fewer overlapping examples to meet the runtime
budget; a single seed leaves training variability unmeasured. UTC calendar
features also shift relative to German civil time at daylight-saving changes.
The test is just one winter holiday week, with strongly related hourly errors;
168 points are not 168 independent replications. Interval coverage can differ
substantially from nominal levels under this calendar shift. No confidence
claim about year-round superiority or weather sensitivity follows from this
experiment.

## 6. Recommendation

Use {SHORT[name]} as the provisional candidate for a JRC short-term load
pipeline, not as an immediately deployable winner. Before committing to one
production model, freeze its specification and evaluate on many untouched
rolling-origin weeks across seasons, holiday transitions and daylight-saving
changes, repeat neural fits across seeds, and check horizon-specific interval
calibration and operational latency. Keep seasonal naive as a monitored
fallback. These follow-up comparisons should remain univariate unless the
production objective is explicitly broadened; no additional input is needed
to reproduce this study.

## 7. Reproducibility note

Run from this directory, using the existing environment (no installation):

```sh
../../.venv/bin/python code/test_study.py
../../.venv/bin/python -u code/forecast.py
```

Seed: **206** for NumPy sampling, LightGBM and Darts/PyTorch; CPU execution,
four threads for fitting, deterministic PyTorch algorithms. The complete
fit/forecast pass took **{fit_seconds:.1f} s**; total computation including
report rendering took **{metrics["total_runtime_seconds"]:.1f} s**. This is
measured execution, not authoring or peer-review time. Per-model runtimes
include selection, scratch refit and prediction, but exclude module imports.
`--report-only` rebuilds figures and tables from frozen forecast CSVs without
refitting. Full-precision metrics, all five probabilistic quantiles per hour,
validation traces, versions and the input SHA-256 are archived in `artifacts/`.

LightGBM uses the prescribed lag/rolling/calendar features, with the annual
lag set to **8,736 hours (364 days)** and all moments shifted one hour.
Its lag warm-up removes 8,736 rows from the supervised matrix only.
Intervals use 1,000 recursive simulated paths and five quantile regressors;
**{lookup["lightgbm"].diagnostics["crossed_path_hours_before_rearrangement"] / 1680:.1f}%** of path-hours required quantile sorting.
The point path and simulated marginal median differ by up to
**{lookup["lightgbm"].diagnostics["point_vs_marginal_median_max_mw"]:,.0f} MW**.
Independent innovations can produce unrealistic within-path roughness.
SARIMA/neural intervals are Gaussian; Prophet uses 1,000 native samples.
Final SARIMA convergence: **{convergence}**; non-converged candidates are
ineligible. Full assumptions and settings are in the per-model JSON files.

Neural models use 168/168-hour chunks, stride 24, maximum 30 epochs and
validation-MAPE patience four. Refit epochs: {lookup["nbeats"].hyperparameters["selected_epochs"]}
(N-BEATS), {lookup["patchtst"].hyperparameters["selected_epochs"]} (transformer).
Figure 03 uses naive = 100 with original-unit labels, not a mixed-unit axis.
All plots are 300 dpi, using a fixed reordered `tab10` palette and text labels.

Hardware (`uname -a`):

```text
{provenance["uname"]}
```

Exact package versions and artifact digests: `artifacts/provenance.json`.
Runtime compatibility corrections and peer-review decisions:
`artifacts/review_record.md`.
"""


def render_report() -> None:
    started = perf_counter()
    results = [read_result(name) for name in NAMES]
    data = load_data()
    test = data.loc[(data.index >= TEST_START) & (data.index < TEST_END)]
    actual = test.to_numpy()
    rows = [
        {
            "name": r.name,
            **score(actual, r.point),
            "runtime_seconds": r.runtime_seconds,
            "hyperparameters": r.hyperparameters,
        }
        for r in results
    ]
    rows.sort(key=lambda row: row["mape_test_pct"])
    winner = next(r for r in results if r.name == rows[0]["name"])
    probability = (
        {
            key: None
            for key in [
                "winner_coverage_80pct",
                "winner_coverage_95pct",
                "winner_pinball_loss_q10",
                "winner_pinball_loss_q50",
                "winner_pinball_loss_q90",
            ]
        }
        if winner.quantiles is None
        else probability_scores(actual, winner.quantiles)
    )
    metrics: dict[str, Any] = {
        "test_start": "2020-01-01",
        "test_end": "2020-01-07",
        "n_test_observations": 168,
        "models": rows,
        "winner": winner.name,
        **probability,
    }
    daily = pd.DataFrame(
        {
            r.name: [mape(actual[i : i + 24], r.point[i : i + 24]) for i in range(0, 168, 24)]
            for r in results
        },
        index=pd.date_range(TEST_START, periods=7, freq="D"),
    )
    daily.index.name = "date_utc"
    daily.to_csv(ROOT / "artifacts" / "per_day_mape.csv")
    probabilistic = []
    for result in results:
        if result.quantiles is not None:
            p = probability_scores(actual, result.quantiles)
            probabilistic.append(
                {
                    "name": result.name,
                    "nominal_coverage_80": 0.8,
                    "nominal_coverage_95": 0.95,
                    **{key.replace("winner_", ""): value for key, value in p.items()},
                    "mean_width_80_mw": float(
                        np.mean(result.quantiles[:, 3] - result.quantiles[:, 1])
                    ),
                    "mean_width_95_mw": float(
                        np.mean(result.quantiles[:, 4] - result.quantiles[:, 0])
                    ),
                }
            )
    pd.DataFrame(probabilistic).to_csv(
        ROOT / "artifacts" / "probabilistic_metrics.csv", index=False
    )
    make_figures(data, results, rows, daily, probability)
    execution_path = ROOT / "artifacts" / "execution.json"
    execution = (
        json.loads(execution_path.read_text())
        if execution_path.exists()
        else {
            "fit_and_forecast_wall_seconds": sum(r.runtime_seconds for r in results),
            "mode": "sum of independent per-model runs; not concurrent wall time",
        }
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in [
            "pandas",
            "numpy",
            "matplotlib",
            "scipy",
            "statsmodels",
            "darts",
            "lightgbm",
            "holidays",
            "torch",
            "pytorch-lightning",
            "prophet",
        ]
    }
    provenance = {
        **execution,
        "uname": subprocess.check_output(["uname", "-a"], text=True).strip(),
        "python": platform.python_version(),
        "packages": versions,
        "input_sha256": hashlib.sha256((ROOT / "opsd_de_load.csv").read_bytes()).hexdigest(),
        "train_observations": 41616,
        "validation_observations": 2208,
        "refit_observations": 43824,
        "test_observations": 168,
        "seed": 206,
    }
    metrics["total_runtime_seconds"] = (
        execution["fit_and_forecast_wall_seconds"] + perf_counter() - started
    )
    write_json(ROOT / "metrics.json", metrics)
    flat = []
    for row in rows:
        flat.append(
            {
                **row,
                "hyperparameters": json.dumps(row["hyperparameters"], sort_keys=True),
                **{k: v for k, v in metrics.items() if k != "models"},
                "is_winner": row["name"] == winner.name,
            }
        )
    pd.DataFrame(flat).to_csv(ROOT / "metrics.csv", index=False)
    (ROOT / "transcript.md").write_text(transcript(rows, results, daily, metrics, provenance))
    provenance["artifact_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in [ROOT / "code", ROOT / "figures"]
        for path in sorted(directory.glob("*"))
        if path.is_file()
    }
    write_json(ROOT / "artifacts" / "provenance.json", provenance)
    print(pd.DataFrame(rows).drop(columns=["hyperparameters"]).to_string(index=False), flush=True)
    print(f"WINNER {winner.name}", flush=True)
