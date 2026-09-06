from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import (COLORS, LABELS, NAMES, QUANTILES, ROOT, SHORT_LABELS, ModelResult,
                    dump_json, figure_style, mape, metrics, probabilistic_metrics, save_figure)


def time_axis(ax: Any, dates: pd.DatetimeIndex) -> None:
    import matplotlib.dates as mdates
    ax.set_xlim(dates[0], dates[-1])
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlabel("Date (UTC)")
    ax.set_ylabel("Load (MW)")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def overview(data: pd.Series) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.87)
    ax.plot(data.index, data, color="#c9c9c9", linewidth=0.28, label="Observed hourly load")
    test = data.loc["2020-01-01":"2020-01-07"]
    ax.axvspan(test.index[0], test.index[-1], color="#c53030", alpha=0.22)
    ax.plot(test.index, test, color="#c53030", linewidth=0.65, label="Held-out test week")
    ax.annotate("Test: Jan 1–7, 2020 (168 h)", xy=(test.index[84], 74000),
                xytext=(0.58, 0.96), textcoords="axes fraction", fontsize=10,
                arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 0.8})
    ax.set_title("German national electricity load: the test week in context", pad=18)
    ax.set_xlabel("Year (UTC)")
    ax.set_ylabel("Load (MW)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(data.index[0], data.index[-1])
    ax.set_ylim(28000, 83000)
    ax.legend(loc="lower left", ncol=2, fontsize=9)
    fig.text(0.09, 0.025, "OPSD / ENTSO-E  |  50,400 hourly observations  |  Later 2020 values are shown here only, never used for fitting.", fontsize=9)
    save_figure(fig, "01_overview.png")


def forecast_comparison(test: pd.Series, results: list[ModelResult], rows: dict[str, dict[str, Any]]) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.15, top=0.82, hspace=0.4, wspace=0.12)
    minimum = min(test.min(), *(r.point.min() for r in results))
    maximum = max(test.max(), *(r.point.max() for r in results))
    for ax, result in zip(axes.flat, results):
        ax.plot(test.index, test, color="black", linewidth=1.3)
        ax.plot(test.index, result.point, color=COLORS[result.name], linewidth=1.65, linestyle="--")
        ax.axvspan(test.index[0], pd.Timestamp("2020-01-02"), color="#777777", alpha=0.06)
        ax.set_title(f"{SHORT_LABELS[result.name]}  |  {rows[result.name]['mape_test_pct']:.2f}% MAPE", fontsize=10)
        ax.set_ylim(minimum - 2500, maximum + 2500)
        ax.set_xlim(test.index[0], test.index[-1])
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(labelsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Load (MW)")
    for ax in axes[1, :]:
        ax.set_xlabel("Date (UTC)")
    fig.suptitle("One origin, six forecasts: German load, Jan 1–7, 2020", fontsize=15, y=0.98)
    fig.legend(handles=[Line2D([0], [0], color="black", label="Observed"),
                        Line2D([0], [0], color="#555555", linestyle="--", label="Model forecast (panel color)")],
               loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2, fontsize=9)
    fig.text(0.08, 0.025, "Identical scales; shaded region: Jan 1 (UTC).  *Darts TransformerModel substitutes for unavailable PatchTST.", fontsize=9)
    save_figure(fig, "02_forecast_comparison.png")


def metric_comparison(ranked: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fields = ["mape_test_pct", "rmse_test_mw", "mae_test_mw"]
    baseline = next(row for row in ranked if row["name"] == "naive")
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.30, top=0.79)
    positions = np.arange(6)
    width = 0.24
    hatches = ["", "///", "..."]
    for j, field in enumerate(fields):
        values = [row[field] / baseline[field] for row in ranked]
        bars = ax.bar(positions + (j - 1) * width, values, width=width * 0.88,
                      color=[COLORS[row["name"]] for row in ranked], hatch=hatches[j],
                      edgecolor="#444444", linewidth=0.3)
        for bar, row in zip(bars, ranked):
            label = f"{row[field]:.2f}%" if j == 0 else f"{row[field]:,.0f} MW"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                    label, rotation=90, ha="center", va="bottom", fontsize=7, color="#222222")
    ax.set_xticks(positions, [SHORT_LABELS[r["name"]].replace("Seasonal naive", "Naive") for r in ranked],
                  rotation=28, ha="right", fontsize=9)
    ax.set_xlabel("Model (ranked by test MAPE)")
    ax.set_ylabel("Error / seasonal-naive error (ratio)")
    ax.set_ylim(0, 1.4)
    ax.axhline(1, color="#777777", linewidth=0.7, linestyle="--")
    ax.grid(axis="x", visible=False)
    fig.suptitle("LightGBM leads on all three error metrics", fontsize=13, y=0.97)
    fig.legend(handles=[Patch(facecolor="#e0e0e0", edgecolor="#444444", hatch=h, label=l)
                        for h, l in zip(hatches, ["MAPE (%)", "RMSE (MW)", "MAE (MW)"])],
               loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3, fontsize=8)
    fig.text(0.13, 0.055, "Bars: baseline-relative errors, lower is better.\nLabels: raw values. Dashed line: naive = 1.\n*Transformer substitute, not PatchTST.", fontsize=8, linespacing=1.5)
    save_figure(fig, "03_metric_comparison.png")


def per_day(test: pd.Series, results: list[ModelResult], ranked: list[dict[str, Any]]) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    lookup = {r.name: r for r in results}
    days = pd.date_range("2020-01-01", periods=7, freq="D")
    values = []
    for row in ranked:
        p = lookup[row["name"]].point
        values.append([mape(test.to_numpy()[test.index.normalize() == day],
                            p[test.index.normalize() == day]) for day in days])
    matrix = pd.DataFrame(values, index=[r["name"] for r in ranked], columns=days.strftime("%Y-%m-%d"))
    matrix.to_csv(ROOT / "artifacts" / "per_day_mape.csv", index_label="model")
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0.25, right=0.81, bottom=0.30, top=0.82)
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=max(40.0, matrix.to_numpy().max()), aspect="auto")
    ax.set_xticks(range(7), [day.strftime("%b %d\n%a") for day in days], fontsize=8)
    ax.set_yticks(range(6), [SHORT_LABELS[r["name"]].replace("Seasonal naive", "Naive") for r in ranked], fontsize=9)
    ax.set_xlabel("Day (UTC)", labelpad=8)
    ax.set_ylabel("Model (ranked by weekly MAPE)")
    ax.grid(False)
    for i in range(6):
        for j in range(7):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if value > 23 else "#111111")
    cax = fig.add_axes([0.845, 0.30, 0.022, 0.52])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label("MAPE (%)", fontsize=8, labelpad=4)
    cax.tick_params(labelsize=8)
    fig.suptitle("Holiday errors differ by model class", fontsize=13, y=0.97)
    fig.text(0.24, 0.88, "Jan 1 is a federal holiday; Jan 4–5 are the weekend.", fontsize=8)
    fig.text(0.24, 0.055, "Each cell uses 24 observations. Lower is better.\n*Transformer substitute, not PatchTST.\nHue encodes error magnitude, not model identity.", fontsize=8, linespacing=1.5)
    save_figure(fig, "04_per_day_mape.png")
    return matrix


def winner_intervals(test: pd.Series, winner: ModelResult, scores: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    q = winner.quantiles
    if q is None:
        raise ValueError("Winner intervals must be native or explicitly calibrated before plotting.")
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.79)
    ax.fill_between(test.index, q[:, 0], q[:, 4], color=COLORS[winner.name], alpha=0.13, label="95% interval")
    ax.fill_between(test.index, q[:, 1], q[:, 3], color=COLORS[winner.name], alpha=0.29, label="80% interval")
    ax.plot(test.index, test, color="black", linewidth=1.6, label="Observed")
    ax.plot(test.index, winner.point, color=COLORS[winner.name], linewidth=2, linestyle="--", label="Point forecast")
    time_axis(ax, test.index)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=4, fontsize=9)
    fig.suptitle(f"{LABELS[winner.name]}: forecast and prediction intervals", fontsize=15, y=0.97)
    fig.text(0.5, 0.9,
             f"Test MAPE {scores['mape_test_pct']:.2f}%  |  Actual coverage: 80% band {scores['coverage80']:.1%}, 95% band {scores['coverage95']:.1%}",
             ha="center", fontsize=11)
    note = "LightGBM intervals are conditional on the recursive median path; predictor uncertainty is not propagated." if winner.name == "lightgbm" else "Coverage is measured on 168 dependent hourly observations, not an independent calibration sample."
    fig.text(0.09, 0.04, note, fontsize=9)
    save_figure(fig, "05_winner_with_intervals.png")


def residuals(test: pd.Series, winner: ModelResult) -> None:
    import matplotlib.pyplot as plt
    residual = test.to_numpy() - winner.point
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.81, wspace=0.12)
    groupings = [[residual[test.index.hour == h] for h in range(24)],
                 [residual[test.index.normalize() == day] for day in pd.date_range("2020-01-01", periods=7)]]
    for ax, groups in zip(axes, groupings):
        bp = ax.boxplot(groups, patch_artist=True, widths=0.6,
                        medianprops={"color": "#111111", "linewidth": 1.1},
                        flierprops={"markersize": 3, "marker": ".", "markerfacecolor": "#777777", "markeredgecolor": "#777777"})
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS[winner.name])
            patch.set_alpha(0.45)
        ax.axhline(0, color="#444444", linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=8)
    axes[0].set_xticks(range(1, 25, 3), [f"{h:02d}" for h in range(0, 24, 3)])
    axes[0].set_xlabel("Hour of day (UTC)")
    axes[0].set_ylabel("Observed − forecast load (MW)")
    axes[0].set_title("By hour of day (7 values per box)")
    axes[1].set_xticks(range(1, 8), [d.strftime("%a\n%b %d") for d in pd.date_range("2020-01-01", periods=7)])
    axes[1].set_xlabel("Day in test week (UTC)")
    axes[1].set_title("By day (24 values per box)")
    fig.suptitle(f"{LABELS[winner.name]}: where forecast errors remain", fontsize=15, y=0.97)
    fig.text(0.5, 0.88, "Positive residual = underprediction; negative residual = overprediction", ha="center", fontsize=10)
    fig.text(0.09, 0.04, "Boxes: median and interquartile range; whiskers: 1.5 × IQR. Each weekday appears only once.", fontsize=9)
    save_figure(fig, "06_residuals.png")


def feature_importance(result: ModelResult) -> None:
    import matplotlib.pyplot as plt
    gain = pd.Series(result.extras["feature_importance_gain"]).sort_values(ascending=True)
    gain.rename("gain").to_csv(ROOT / "artifacts" / "lightgbm_feature_importance.csv", index_label="feature")
    percentage = gain / gain.sum() * 100
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0.36, right=0.94, bottom=0.15, top=0.83)
    bars = ax.barh(range(len(gain)), percentage, height=0.58, color=COLORS["lightgbm"])
    ax.set_yticks(range(len(gain)), gain.index, fontsize=8)
    ax.set_xlabel("Share of split gain (%)")
    ax.set_ylabel("Engineered feature")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, percentage.max() * 1.25)
    for bar, value in zip(bars, percentage):
        ax.text(value + 0.3, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=8)
    fig.suptitle("LightGBM: fitted median-model feature importance", fontsize=12, y=0.96)
    fig.text(0.36, 0.885, "Gain, normalized to 100%; highest at top", fontsize=8)
    fig.text(0.06, 0.04, "Correlated features share credit. Gain is not a causal effect or held-out importance.", fontsize=8)
    save_figure(fig, "07_feature_importance.png")


def decomposition(result: ModelResult) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 6))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.13, top=0.87, hspace=0.7)
    for ax, component in zip(axes, ["trend", "weekly", "yearly"]):
        frame = pd.read_csv(ROOT / "artifacts" / f"prophet_decomposition_{component}.csv", parse_dates=["ds"])
        values = frame[component].to_numpy()
        multiplicative = result.hyperparameters["seasonality_mode"] == "multiplicative" and component != "trend"
        if multiplicative:
            values = values * 100
        ax.plot(frame.ds, values, color=COLORS["prophet"], linewidth=1.5)
        ax.set_title(component.title(), loc="left", fontsize=10, pad=3)
        ax.set_ylabel("Effect (%)" if multiplicative else "Load (MW)" if component == "trend" else "Effect (MW)")
        if component == "trend":
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.set_xlabel("Year (UTC)")
        elif component == "weekly":
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%a"))
            ax.set_xlabel("Day of week (UTC)")
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.set_xlabel("Month (UTC)")
        ax.set_xlim(frame.ds.iloc[0], frame.ds.iloc[-1])
        ax.tick_params(labelsize=8)
    fig.suptitle("Prophet: trend and recurring seasonal components", fontsize=14, y=0.97)
    fig.text(0.11, 0.025, "Fitted on 2015–2019 only. Seasonal effects are components, not standalone load forecasts. Holiday effects are separate.", fontsize=8)
    save_figure(fig, "08_decomposition.png")


def study_text(ranked: list[dict[str, Any]], results: list[ModelResult], run: dict[str, Any],
               score: dict[str, Any], daily: pd.DataFrame) -> str:
    rows = {r["name"]: r for r in ranked}
    models = {r.name: r for r in results}
    winner = score["winner"]
    best = rows[winner]
    reduction = 100 * (1 - best["mape_test_pct"] / rows["naive"]["mape_test_pct"])
    rest = best["mape_jan2_to_jan7_pct"]
    q = models[winner].quantiles
    width80 = float(np.mean(q[:, 3] - q[:, 1])) if q is not None else float("nan")
    table = ["| Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2–7 MAPE (%) | Fit/run (s) |",
             "|:--|--:|--:|--:|--:|--:|--:|"]
    for row in ranked:
        table.append(f"| {SHORT_LABELS[row['name']]} | {row['mape_test_pct']:.2f} | {row['rmse_test_mw']:,.0f} | {row['mae_test_mw']:,.0f} | {row['mape_jan1_pct']:.2f} | {row['mape_jan2_to_jan7_pct']:.2f} | {row['runtime_seconds']:.1f} |")
    discussion = f"""{LABELS[winner]} won at {best['mape_test_pct']:.2f}% MAPE, a {reduction:.1f}% reduction relative to weekly persistence. Its Jan 1 error was {best['mape_jan1_pct']:.2f}%, compared with {rest:.2f}% over the remaining six days. LightGBM can combine annual load history, recent level and known calendar categories nonlinearly. That is a plausible explanation for its advantage, not a causal conclusion: no feature ablation was run, and split gain cannot establish which feature caused the improvement.

The holiday did not hurt every method most on Jan 1. Naive and SARIMA repeated the preceding Christmas week: Jan 2 inherited Dec 26, and Jan 7 inherited New Year's Eve. Their Jan 7 errors reached {daily.loc['naive'].iloc[6]:.1f}% and {daily.loc['sarima'].iloc[6]:.1f}%. SARIMA's daily residual dynamics improved slightly on persistence but could not restore the post-holiday level. Prophet ranked second, yet its holiday term did not eliminate Jan 1 error ({rows['prophet']['mape_jan1_pct']:.1f}%) or Saturday error ({daily.loc['prophet'].iloc[3]:.1f}%). A fixed additive calendar pattern does not capture every bridge-day or holiday interaction.

N-BEATS and the transformer overpredicted the holiday heavily, with Jan 1 MAPE of {rows['nbeats']['mape_jan1_pct']:.1f}% and {rows['patchtst']['mape_jan1_pct']:.1f}%. Neither receives an explicit holiday flag. The self-imposed sparse weekly-window schedule and short epoch budget also limit conclusions about well-tuned deep learning; measured runtime did not require this handicap. All five fitted models beat naive, but the ordering is not a general complexity ranking: two neural models were below Prophet and LightGBM, while classical weekly structure remained useful.

The winner's nominal 80% and 95% intervals covered {score['winner_coverage_80pct']:.1%} and {score['winner_coverage_95pct']:.1%} of test hours; its 80% band averaged {width80:,.0f} MW wide. These are conditional quantiles along one recursive median path, not full simulated trajectories. Smoother recursive predictions can shrink rolling variability. High coverage on one dependent, holiday-heavy week does not establish calibration, and it is not evidence of accuracy during the later 2020 demand disruption. Jan 6's regional holiday is deliberately absent from the federal calendar."""
    note = f"""# German hourly load: a six-model forecasting bake-off

## 1. Data

Open Power System Data (OPSD), derived from the ENTSO-E Transparency Platform, supplies 50,400 national hourly load observations from 2015-01-01 00:00 to 2020-09-30 23:00 UTC, in megawatts. The CSV was checked against the complete hourly grid: no missing values, duplicate timestamps or gaps; no imputation was needed. No external observations were used. Calendar terms and German federal holidays are knowable from timestamps. Later 2020 observations appear only in the [overview](figures/01_overview.png).

## 2. Why these six models

Weekly seasonal naive anchors persistence; SARIMA adds daily residual dynamics after an explicit weekly difference; Prophet adds smooth trend, daily/weekly/yearly cycles and German holidays; LightGBM combines calendar features, 24/168/8,760-hour lags and strictly past 24/168-hour rolling means and standard deviations. N-BEATS tests a univariate residual MLP; a Darts encoder-decoder TransformerModel tests attention. **PatchTST is unavailable in Darts 0.41.0**, and TSMixer is not a transformer, so this is an explicit substitute, not a PatchTST result (`patchtst` is retained only as the requested machine key).

## 3. Validation strategy

Train has 41,616 hours through September 2019; validation has 2,208 hours (October–December). Initial coefficients/scalers use train only. Score 13 weekly blocks plus 24 hours, weighting hours equally. Earlier validation actuals update contexts/states, never coefficients. Prophet has no state update: its validation spans 1–92 days ahead, so compare validation scores only within models. Select between two SARIMA orders, two Prophet seasonality modes (priors fixed), and two LightGBM leaf counts. Custom all-origin MAPE early stopping selects {models['nbeats'].hyperparameters['selected_epochs']} N-BEATS epochs and {models['patchtst'].hyperparameters['selected_epochs']} transformer epochs (maximum 30, patience 5). Both have 168-hour inputs/outputs and training stride 168; N-BEATS retains default stacks. Freshly refit each selected configuration on all 43,824 pre-test hours, then forecast the complete test week without observed test lags. LightGBM loses 8,760 initial training targets to lag warmup, not source observations. Its calendar is Europe/Berlin; Prophet uses UTC holidays/cycles, with a local-midnight/DST limitation.

## 4. Results

{chr(10).join(table)}

*TransformerModel substitute. Jan 1 is a UTC-date subset; Jan 2–7 includes the weekend, not just working days. Errors are hour-weighted; MAPE = 100 × mean(|actual − forecast| / actual). RMSE and MAE retain MW. Ranking is a retrospective test comparison, not permission to tune on test. All forecasts and candidate scores are in `artifacts/`.*

![Six forecasts on the identical test window](figures/02_forecast_comparison.png)

Winner uncertainty: nominal **80% → {score['winner_coverage_80pct']:.2%} actual**; nominal **95% → {score['winner_coverage_95pct']:.2%} actual**. Pinball loss (MW): q0.1 **{score['winner_pinball_loss_q10']:.1f}**, q0.5 **{score['winner_pinball_loss_q50']:.1f}**, q0.9 **{score['winner_pinball_loss_q90']:.1f}**. JSON coverage uses fractions. SARIMA uses Gaussian errors, Prophet predictive samples, and neural models five-quantile regression; naive has no native intervals. Points are medians except Prophet's `yhat` mean; its q0.5 is the sampled predictive median. LightGBM adds q0.025/q0.975 fits for 95% intervals; crossing correction preserves its median.

## 5. Discussion

{discussion}

## 6. Recommendation

For a JRC short-term forecasting pilot, choose LightGBM as the provisional single candidate, not an already production-qualified model. Before deployment, freeze this configuration and evaluate many untouched rolling origins across seasons, holiday transitions and demand regimes; compare against persistence at each horizon; quantify skill uncertainty using week-level resampling; and calibrate intervals on separate historical origins. Audit timezone and regional-holiday handling, recursive-feature drift and operational latency. Weather may matter in production, but it was neither obtained nor used here: adding it would answer a different question from this deliberately temporal-only study.

## 7. Reproducibility note

From this directory, with the supplied venv, run:

```sh
../../.venv/bin/python code/forecast.py
../../.venv/bin/python -m unittest discover -s code -p 'test_*.py'
../../.venv/bin/python code/verify_study.py
```

No installs. `forecast.py` is the entry point; model modules are imported through its loader. Seed **{run['seed']}** covers Python, NumPy, LightGBM, Prophet and Torch; neural execution is deterministic CPU, four threads. Measured model-and-report wall time: **{score['total_runtime_seconds']:.1f} seconds ({score['total_runtime_seconds']/60:.2f} minutes)**, excluding development, review and cold-cache preparation. Per-model time includes selection/refit/prediction; total also includes model imports and figures. Versions, hashes, hardware, candidate scores and per-hour forecasts are in `artifacts/`. `--report-only` rebuilds figures without fitting. `training_attempt1.log` and `review.md` retain a corrected pre-scoring statsmodels memory-mode failure.

```text
{run['uname']}
```

Python {run['python']}; Darts {run['packages']['darts']}; Torch {run['packages']['torch']}; LightGBM {run['packages']['lightgbm']}; statsmodels {run['packages']['statsmodels']}; Prophet {run['packages']['prophet']}; holidays {run['packages']['holidays']}.

Additional figures: [raw-labeled relative metrics](figures/03_metric_comparison.png), [per-day errors](figures/04_per_day_mape.png), [winner intervals](figures/05_winner_with_intervals.png), [residuals](figures/06_residuals.png), [LightGBM gain](figures/07_feature_importance.png), [Prophet components](figures/08_decomposition.png). Metric bars are normalized to naive = 1 so MW and percentages never share a raw numerical axis.
"""
    if len(discussion.split()) < 200 or len(discussion.split()) > 400:
        raise ValueError(f"Discussion must be 200–400 words, got {len(discussion.split())}.")
    if chr(0x2014) in note:
        raise ValueError("Prose contains a forbidden em dash.")
    return note


def generate(data: pd.Series, results: list[ModelResult]) -> None:
    start = time.perf_counter()
    run = json.loads((ROOT / "artifacts" / "run.json").read_text())
    test = data.loc["2020-01-01":"2020-01-07"]
    validation = data.loc["2019-10-01":"2019-12-31"]
    rows = [{"name": r.name, **metrics(test, r.point), "runtime_seconds": r.runtime_seconds,
             "hyperparameters": r.hyperparameters} for r in results]
    ranked = sorted(rows, key=lambda r: r["mape_test_pct"])
    winner = next(r for r in results if r.name == ranked[0]["name"])
    if winner.quantiles is None:
        errors = validation.to_numpy() - winner.validation_point
        winner.quantiles = winner.point[:, None] + np.quantile(errors, QUANTILES)[None, :]
        winner.hyperparameters["auxiliary_winner_intervals"] = "Pooled validation signed-error quantiles; not native naive intervals"
        pd.DataFrame(winner.quantiles, index=test.index, columns=[f"q{q:g}" for q in QUANTILES]).to_csv(
            ROOT / "artifacts" / "naive_auxiliary_intervals.csv", index_label="utc_timestamp")
    score = {"test_start": "2020-01-01", "test_end": "2020-01-07", "n_test_observations": 168,
             "models": ranked, "winner": winner.name, **probabilistic_metrics(test.to_numpy(), winner.quantiles)}
    figure_style()
    overview(data)
    forecast_comparison(test, results, {r["name"]: r for r in rows})
    metric_comparison(ranked)
    daily = per_day(test, results, ranked)
    winner_intervals(test, winner, {**ranked[0], "coverage80": score["winner_coverage_80pct"],
                                  "coverage95": score["winner_coverage_95pct"]})
    residuals(test, winner)
    top_two = [r["name"] for r in ranked[:2]]
    if "lightgbm" in top_two:
        feature_importance(next(r for r in results if r.name == "lightgbm"))
    if "prophet" in top_two:
        decomposition(next(r for r in results if r.name == "prophet"))
    validation_rows = []
    block_rows = []
    for result in results:
        validation_rows.append({"name": result.name, "selected_validation_mape_pct": mape(validation.to_numpy(), result.validation_point)})
        for i in range(0, len(validation), 168):
            block_rows.append({"name": result.name, "origin_utc": str(validation.index[i]),
                               "n_hours": len(validation.iloc[i:i+168]),
                               "mape_pct": mape(validation.iloc[i:i+168].to_numpy(), result.validation_point[i:i+168])})
    pd.DataFrame(validation_rows).to_csv(ROOT / "artifacts" / "validation_summary.csv", index=False)
    pd.DataFrame(block_rows).to_csv(ROOT / "artifacts" / "validation_blocks.csv", index=False)
    score["total_runtime_seconds"] = run["training_wall_seconds"] + time.perf_counter() - start
    dump_json(ROOT / "metrics.json", score)
    flat = []
    global_values = {k: v for k, v in score.items() if k != "models"}
    for row in ranked:
        flat.append({**global_values, **row, "hyperparameters": json.dumps(row["hyperparameters"], sort_keys=True)})
    pd.DataFrame(flat).to_csv(ROOT / "metrics.csv", index=False, float_format="%.12g")
    (ROOT / "transcript.md").write_text(study_text(ranked, results, run, score, daily), encoding="utf-8")
    print(f"REPORT_DONE winner={winner.name} mape={ranked[0]['mape_test_pct']:.4f}% wall={score['total_runtime_seconds']:.2f}s", flush=True)
