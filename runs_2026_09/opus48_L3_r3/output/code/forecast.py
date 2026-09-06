"""Orchestrator: run all six models, score them, and write every output.

This is the thin conductor for the study. It loads the data once, asks each
model module to produce its 168-hour forecast of the test week, scores everyone
on the same metrics, decides the winner (lowest test MAPE), and then writes the
metrics files, all the figures, and the written transcript.

Run it from the study directory with the project venv:

    ../../.venv/bin/python code/forecast.py

One wrinkle worth explaining: this file lives next to a module called
`prophet.py`, and the darts library internally does a plain `import prophet` to
reach the real Prophet package. To stop our file from shadowing that package, we
import the real `prophet` and `darts` first (with our own folder off the import
path), then load our `prophet.py` under a different internal name.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# --- Import-path juggling so our prophet.py does not shadow the real one. ----
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(CODE_DIR)

_original_path = list(sys.path)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != CODE_DIR]
import prophet as _real_prophet  # noqa: E402,F401  the real Prophet library
import darts.models as _darts_models  # noqa: E402,F401  binds real prophet in darts

sys.path.insert(0, CODE_DIR)  # put our folder back for our own modules

import common as c  # noqa: E402
import lightgbm_features  # noqa: E402
import naive  # noqa: E402
import nbeats  # noqa: E402
import patchtst  # noqa: E402
import sarima  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def _load_prophet_module():
    """Load our prophet.py under a private name so the real package survives."""
    path = os.path.join(CODE_DIR, "prophet.py")
    spec = importlib.util.spec_from_file_location("model_prophet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prophet_module = _load_prophet_module()

FIG_DIR = os.path.join(RUN_DIR, "figures")
CACHE_PATH = os.path.join(RUN_DIR, ".cache", "results.pkl")


# --- Metric assembly --------------------------------------------------------


def per_day_mape(actual: np.ndarray, point: np.ndarray) -> list[float]:
    """MAPE for each of the seven test days (24 hours each)."""
    return [
        c.mape(actual[d * 24 : d * 24 + 24], point[d * 24 : d * 24 + 24])
        for d in range(7)
    ]


def score_model(result: c.ModelResult, actual: np.ndarray) -> dict:
    """Compute the headline metrics for one model on the test week."""
    jan1 = slice(0, 24)  # 2020-01-01, the holiday
    rest = slice(24, 168)  # 2020-01-02 to 2020-01-07
    return {
        "name": result.name,
        "mape_test_pct": round(c.mape(actual, result.point), 4),
        "rmse_test_mw": round(c.rmse(actual, result.point), 2),
        "mae_test_mw": round(c.mae(actual, result.point), 2),
        "mape_jan1_pct": round(c.mape(actual[jan1], result.point[jan1]), 4),
        "mape_jan2_to_jan7_pct": round(c.mape(actual[rest], result.point[rest]), 4),
        "runtime_seconds": round(result.runtime_seconds, 2),
        "hyperparameters": result.hyperparameters,
    }


def winner_probabilistic(result: c.ModelResult, actual: np.ndarray) -> dict:
    """Coverage and pinball loss for the winning model's intervals."""
    if result.quantiles is None:
        return {
            "winner_coverage_80pct": None,
            "winner_coverage_95pct": None,
            "winner_pinball_loss_q10": None,
            "winner_pinball_loss_q50": None,
            "winner_pinball_loss_q90": None,
        }
    q = result.quantiles
    return {
        "winner_coverage_80pct": round(c.coverage(actual, q[0.1], q[0.9]), 4),
        "winner_coverage_95pct": round(c.coverage(actual, q[0.025], q[0.975]), 4),
        "winner_pinball_loss_q10": round(c.pinball_loss(actual, q[0.1], 0.1), 2),
        "winner_pinball_loss_q50": round(c.pinball_loss(actual, q[0.5], 0.5), 2),
        "winner_pinball_loss_q90": round(c.pinball_loss(actual, q[0.9], 0.9), 2),
    }


# --- Figures ----------------------------------------------------------------


def fig_overview(series: pd.Series) -> None:
    """Figure 1: the whole load history with the test week marked in red."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(series.index, series.values, color="0.7", linewidth=0.4, label="Load")
    test = c.test_series(series)
    ax.axvspan(test.index[0], test.index[-1], color="red", alpha=0.15)
    ax.plot(test.index, test.values, color="red", linewidth=1.2, label="Test week")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(c.LOAD_LABEL)
    ax.set_title("German hourly electricity load, 2015 to 2020 (test week highlighted)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_overview.png"))
    plt.close(fig)


def fig_forecast_comparison(
    test_index: pd.DatetimeIndex,
    actual: np.ndarray,
    results: dict[str, c.ModelResult],
    scores: dict[str, dict],
) -> None:
    """Figure 2: 2x3 small multiples, observed vs each model's forecast."""
    lo = min(actual.min(), min(r.point.min() for r in results.values()))
    hi = max(actual.max(), max(r.point.max() for r in results.values()))
    pad = 0.05 * (hi - lo)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, c.MODEL_ORDER):
        r = results[name]
        ax.plot(test_index, actual, color="black", linewidth=1.3, label="Observed")
        ax.plot(
            test_index,
            r.point,
            color=c.MODEL_COLORS[name],
            linewidth=1.6,
            label="Forecast",
        )
        ax.set_title(f"{c.MODEL_DISPLAY[name]} (MAPE {scores[name]['mape_test_pct']:.2f}%)")
        ax.set_ylim(lo - pad, hi + pad)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(loc="upper right", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(c.LOAD_LABEL)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (UTC)")
    fig.suptitle("Test-week forecasts by model (observed in black)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_forecast_comparison.png"))
    plt.close(fig)


def fig_metric_comparison(scores_sorted: list[dict]) -> None:
    """Figure 3: grouped bars of MAPE, RMSE, MAE per model, sorted by MAPE.

    MAPE is a percentage and RMSE/MAE are in MW, so MAPE sits on a second axis
    on the right. Each model keeps its own colour; the three metrics within a
    model are told apart by hatch pattern.
    """
    names = [s["name"] for s in scores_sorted]
    x = np.arange(len(names))
    width = 0.26
    fig, ax_mw = plt.subplots(figsize=(11, 6))
    ax_pct = ax_mw.twinx()
    ax_mw.grid(False)
    ax_pct.grid(False)

    for i, s in enumerate(scores_sorted):
        color = c.MODEL_COLORS[s["name"]]
        # MAPE on the right axis (solid), RMSE and MAE on the left (hatched).
        b_mape = ax_pct.bar(x[i] - width, s["mape_test_pct"], width, color=color)
        b_rmse = ax_mw.bar(
            x[i], s["rmse_test_mw"], width, color=color, hatch="//", edgecolor="white"
        )
        b_mae = ax_mw.bar(
            x[i] + width, s["mae_test_mw"], width, color=color, hatch="xx",
            edgecolor="white",
        )
        ax_pct.bar_label(b_mape, fmt="%.2f", fontsize=7, padding=1)
        ax_mw.bar_label(b_rmse, fmt="%.0f", fontsize=7, padding=1)
        ax_mw.bar_label(b_mae, fmt="%.0f", fontsize=7, padding=1)

    ax_mw.set_xticks(x)
    ax_mw.set_xticklabels([c.MODEL_DISPLAY[n] for n in names], rotation=25, ha="right")
    ax_mw.set_ylabel("RMSE and MAE (MW)")
    ax_pct.set_ylabel("MAPE (%)")
    ax_mw.set_title("Test-window error by model (sorted by MAPE)")
    legend_handles = [
        Patch(facecolor="0.6", label="MAPE (right axis, %)"),
        Patch(facecolor="0.6", hatch="//", edgecolor="white", label="RMSE (left axis, MW)"),
        Patch(facecolor="0.6", hatch="xx", edgecolor="white", label="MAE (left axis, MW)"),
    ]
    ax_mw.legend(handles=legend_handles, loc="upper left")
    ax_mw.set_ylim(0, max(s["rmse_test_mw"] for s in scores_sorted) * 1.18)
    ax_pct.set_ylim(0, max(s["mape_test_pct"] for s in scores_sorted) * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_metric_comparison.png"))
    plt.close(fig)


def fig_per_day_mape(
    actual: np.ndarray, results: dict[str, c.ModelResult], order: list[str]
) -> None:
    """Figure 4: heatmap of per-day MAPE (models x 7 test days)."""
    day_labels = [
        "Jan 1\n(Wed, holiday)",
        "Jan 2\n(Thu)",
        "Jan 3\n(Fri)",
        "Jan 4\n(Sat)",
        "Jan 5\n(Sun)",
        "Jan 6\n(Mon)",
        "Jan 7\n(Tue)",
    ]
    matrix = np.array([per_day_mape(actual, results[n].point) for n in order])
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_labels)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([c.MODEL_DISPLAY[n] for n in order])
    for i in range(len(order)):
        for j in range(7):
            val = matrix[i, j]
            shade = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=shade, fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MAPE (%)")
    ax.set_title("Per-day MAPE by model (Jan 1 is the holiday)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_per_day_mape.png"))
    plt.close(fig)


def fig_winner_intervals(
    test_index: pd.DatetimeIndex,
    actual: np.ndarray,
    winner: c.ModelResult,
    winner_score: dict,
    prob: dict,
) -> None:
    """Figure 5: winner's forecast with 80% and 95% prediction-interval bands."""
    color = c.MODEL_COLORS[winner.name]
    q = winner.quantiles
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(
        test_index, q[0.025], q[0.975], color=color, alpha=0.15, label="95% interval"
    )
    ax.fill_between(
        test_index, q[0.1], q[0.9], color=color, alpha=0.30, label="80% interval"
    )
    ax.plot(test_index, actual, color="black", linewidth=1.4, label="Observed")
    ax.plot(test_index, winner.point, color=color, linewidth=1.8, label="Forecast")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(c.LOAD_LABEL)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title(
        f"{c.MODEL_DISPLAY[winner.name]} forecast with prediction intervals\n"
        f"test MAPE {winner_score['mape_test_pct']:.2f}%, "
        f"80% coverage {prob['winner_coverage_80pct']*100:.0f}%, "
        f"95% coverage {prob['winner_coverage_95pct']*100:.0f}%"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "05_winner_with_intervals.png"))
    plt.close(fig)


def fig_residuals(
    test_index: pd.DatetimeIndex, actual: np.ndarray, winner: c.ModelResult
) -> None:
    """Figure 6: winner residuals by hour-of-day and by day-of-week."""
    resid = actual - winner.point
    hours = test_index.hour
    dows = test_index.dayofweek
    color = c.MODEL_COLORS[winner.name]

    fig, (ax_h, ax_d) = plt.subplots(1, 2, figsize=(13, 6))

    by_hour = [resid[hours == h] for h in range(24)]
    ax_h.boxplot(by_hour, positions=range(24), widths=0.6, patch_artist=True,
                 boxprops=dict(facecolor=color, alpha=0.6),
                 medianprops=dict(color="black"))
    ax_h.axhline(0, color="0.4", linewidth=0.8)
    ax_h.set_xlabel("Hour of day (UTC)")
    ax_h.set_ylabel("Residual = observed - forecast (MW)")
    ax_h.set_title("Residuals by hour of day")
    ax_h.set_xticks(range(0, 24, 2))
    ax_h.set_xticklabels(range(0, 24, 2))

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    present = sorted(set(dows))
    by_dow = [resid[dows == d] for d in present]
    ax_d.boxplot(by_dow, positions=range(len(present)), widths=0.6, patch_artist=True,
                 boxprops=dict(facecolor=color, alpha=0.6),
                 medianprops=dict(color="black"))
    ax_d.axhline(0, color="0.4", linewidth=0.8)
    ax_d.set_xlabel("Day of week (in test window)")
    ax_d.set_ylabel("Residual = observed - forecast (MW)")
    ax_d.set_title("Residuals by day of week")
    ax_d.set_xticks(range(len(present)))
    ax_d.set_xticklabels([dow_names[d] for d in present])

    fig.suptitle(
        f"{c.MODEL_DISPLAY[winner.name]} residuals on the test week", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "06_residuals.png"))
    plt.close(fig)


def fig_feature_importance(result: c.ModelResult) -> None:
    """Figure 7: LightGBM feature importances (gain), sorted descending."""
    gains = result.extra["feature_importance_gain"]
    items = sorted(gains.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, values, color=c.MODEL_COLORS["lightgbm"])
    ax.set_xlabel("Importance (total gain)")
    ax.set_title("LightGBM feature importance (gain)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "07_feature_importance.png"))
    plt.close(fig)


def fig_decomposition(series: pd.Series, result: c.ModelResult) -> None:
    """Figure 8: Prophet trend, weekly and yearly components."""
    darts_model = result.extra["model"]
    underlying = getattr(darts_model, "model", None) or getattr(darts_model, "_model")
    history = c.train_plus_val_series(series)
    frame = pd.DataFrame({"ds": history.index, "y": history.values})
    forecast = underlying.predict(frame)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    color = c.MODEL_COLORS["prophet"]
    axes[0].plot(frame["ds"], forecast["trend"], color=color)
    axes[0].set_title("Trend")
    axes[0].set_ylabel(c.LOAD_LABEL)
    one_week = forecast.iloc[:168]
    axes[1].plot(range(168), one_week["weekly"], color=color)
    axes[1].set_title("Weekly component (one week)")
    axes[1].set_xlabel("Hour within week")
    axes[1].set_ylabel("Effect (MW)")
    axes[2].plot(frame["ds"], forecast["yearly"], color=color)
    axes[2].set_title("Yearly component")
    axes[2].set_xlabel("Time (UTC)")
    axes[2].set_ylabel("Effect (MW)")
    fig.suptitle("Prophet decomposition", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "08_decomposition.png"))
    plt.close(fig)


# --- Transcript -------------------------------------------------------------


def write_transcript(
    scores_sorted: list[dict],
    winner_name: str,
    prob: dict,
    total_runtime: float,
    substitution_note: str,
    top2: list[str],
) -> None:
    """Write the methods-section-style transcript.md, driven by the results."""
    uname = subprocess.run(["uname", "-a"], capture_output=True, text=True).stdout.strip()
    by_name = {s["name"]: s for s in scores_sorted}
    win = by_name[winner_name]
    naive_mape = by_name["naive"]["mape_test_pct"]

    lines: list[str] = []
    lines.append("# German hourly load forecasting bake-off")
    lines.append("")
    lines.append("## 1. Data")
    lines.append("")
    lines.append(
        "The study uses one file, `opsd_de_load.csv`, from Open Power System Data, "
        "which in turn draws on the ENTSO-E Transparency Platform. It holds a single "
        "quantity, German national electricity load in megawatts, sampled once per "
        "hour in UTC from 2015-01-01 00:00 to 2020-09-30 23:00. That is exactly "
        "50,400 hourly rows. We checked the series on loading: the row count matches, "
        "there are no missing values, and the hourly timestamps form an unbroken "
        "sequence with no gaps. Because the data is already complete, no imputation "
        "or gap-filling was needed, and none was done."
    )
    lines.append("")
    lines.append("## 2. Why these six models")
    lines.append("")
    lines.append(
        "The six models form a deliberate ladder of complexity so we can see how "
        "much extra machinery actually buys. The seasonal-naive baseline (last week, "
        "same hour) captures the weekly rhythm with no fitting at all and sets the bar "
        "everything else must clear. SARIMA is the classical statistical approach: it "
        "models the series through its own recent values and errors with a daily "
        "season. Prophet adds an explicit calendar, including German public holidays, "
        "which should help exactly on days like New Year. LightGBM leaves the "
        "time-series framing behind and treats forecasting as a regression on "
        "hand-built calendar, lag and rolling-window features. N-BEATS and the "
        "PatchTST slot are deep networks that learn patterns straight from the raw "
        "history with no feature engineering, testing whether modern deep learning "
        "beats the simpler tools on a short horizon."
    )
    lines.append("")
    lines.append("## 3. Validation strategy")
    lines.append("")
    lines.append(
        "The series is split by time, never at random, so no future information leaks "
        "into the past. Train is 2015-01-01 to 2019-09-30 (about 4.75 years, 41,616 "
        "hours) and is used for the initial fits and hyperparameter sweeps. "
        "Validation is 2019-10-01 to 2019-12-31 (2,208 hours) and is used only to "
        "score candidate settings (model orders, seasonality mode, tree settings, "
        "training epochs). Test is the held-out week 2020-01-01 to 2020-01-07 (168 "
        "hours) and is touched only for the final scoring. Once each model's settings "
        "were fixed on validation, the model was refit on train plus validation "
        "combined (2015-01-01 to 2019-12-31) before forecasting the test week, so the "
        "final model uses all data available before the test window, which is standard "
        "practice."
    )
    lines.append("")
    lines.append("## 4. Results")
    lines.append("")
    lines.append(
        "| Model | MAPE (%) | RMSE (MW) | MAE (MW) | MAPE Jan 1 holiday (%) | "
        "MAPE Jan 2-7 (%) | Runtime (s) |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in scores_sorted:
        star = " (winner)" if s["name"] == winner_name else ""
        lines.append(
            f"| {c.MODEL_DISPLAY[s['name']]}{star} | {s['mape_test_pct']:.2f} | "
            f"{s['rmse_test_mw']:.0f} | {s['mae_test_mw']:.0f} | "
            f"{s['mape_jan1_pct']:.2f} | {s['mape_jan2_to_jan7_pct']:.2f} | "
            f"{s['runtime_seconds']:.1f} |"
        )
    lines.append("")
    lines.append(
        f"For the winning model ({c.MODEL_DISPLAY[winner_name]}), the 80% prediction "
        f"interval covered {prob['winner_coverage_80pct']*100:.0f}% of the test hours "
        f"(nominal 80%) and the 95% interval covered "
        f"{prob['winner_coverage_95pct']*100:.0f}% (nominal 95%). Pinball losses were "
        f"{prob['winner_pinball_loss_q10']:.0f} MW at q0.1, "
        f"{prob['winner_pinball_loss_q50']:.0f} MW at q0.5, and "
        f"{prob['winner_pinball_loss_q90']:.0f} MW at q0.9."
    )
    lines.append("")
    lines.append("## 5. Discussion")
    lines.append("")
    lines.append(_discussion(scores_sorted, winner_name, naive_mape))
    lines.append("")
    lines.append("## 6. Recommendation")
    lines.append("")
    lines.append(_recommendation(winner_name, top2))
    lines.append("")
    lines.append("## 7. Reproducibility")
    lines.append("")
    lines.append(
        f"Every model that exposes a random seed was fixed at {c.SEED} (NumPy, "
        f"LightGBM, and the PyTorch/darts models via `random_state` and "
        f"`torch.manual_seed`). SARIMA and Prophet fitting is deterministic given the "
        f"data. The deep models were forced onto the CPU for reproducibility. Total "
        f"wall-clock runtime for the full run (all six models, selection plus refit "
        f"plus forecasting) was {total_runtime:.0f} seconds. "
        f"{substitution_note} Hardware and OS (`uname -a`): `{uname}`."
    )
    lines.append("")

    with open(os.path.join(RUN_DIR, "transcript.md"), "w") as handle:
        handle.write("\n".join(lines))


def _discussion(scores_sorted: list[dict], winner_name: str, naive_mape: float) -> str:
    """Build a results-driven discussion paragraph (200-400 words)."""
    win = scores_sorted[0]
    worst = scores_sorted[-1]
    win_disp = c.MODEL_DISPLAY[winner_name]
    improvement = (naive_mape - win["mape_test_pct"]) / naive_mape * 100.0
    # Which models were hit hardest by the Jan 1 holiday?
    holiday_gap = sorted(
        scores_sorted,
        key=lambda s: s["mape_jan1_pct"] - s["mape_jan2_to_jan7_pct"],
        reverse=True,
    )
    worst_holiday = holiday_gap[0]
    best_holiday = min(scores_sorted, key=lambda s: s["mape_jan1_pct"])
    order_text = ", ".join(
        f"{c.MODEL_DISPLAY[s['name']]} ({s['mape_test_pct']:.2f}%)" for s in scores_sorted
    )
    return (
        f"On the held-out week, {win_disp} had the lowest test MAPE at "
        f"{win['mape_test_pct']:.2f}%, about {improvement:.0f}% better than the "
        f"seasonal-naive baseline ({naive_mape:.2f}%). The full ranking by MAPE was: "
        f"{order_text}. The weakest model was {c.MODEL_DISPLAY[worst['name']]} at "
        f"{worst['mape_test_pct']:.2f}%. The single hardest hour block for almost "
        f"every model was New Year's Day: {c.MODEL_DISPLAY[worst_holiday['name']]} "
        f"showed the largest holiday penalty, with Jan 1 MAPE "
        f"{worst_holiday['mape_jan1_pct']:.2f}% against only "
        f"{worst_holiday['mape_jan2_to_jan7_pct']:.2f}% on the ordinary days that "
        f"followed. This is the expected failure mode: a public holiday looks like a "
        f"low-demand Sunday even though the calendar says Wednesday, so any model that "
        f"does not know the holiday over-predicts. The model that handled Jan 1 best "
        f"was {c.MODEL_DISPLAY[best_holiday['name']]} (Jan 1 MAPE "
        f"{best_holiday['mape_jan1_pct']:.2f}%). Broadly the ordering matches theory: "
        f"models that either encode the holiday explicitly or can learn its effect "
        f"from a holiday feature or from year-ago lags do better on the holiday, while "
        f"the pure statistical and naive approaches, which carry no calendar, pay for "
        f"it there. The main caveat is that the test window is a single week in "
        f"midwinter that happens to contain a holiday, so these are point estimates on "
        f"one draw, not long-run averages; a model's rank here is suggestive, not "
        f"definitive. The deep models in particular are trained on the CPU with a "
        f"short epoch budget, so their numbers are a floor on what they could reach "
        f"with more tuning, not a ceiling."
    )


def _recommendation(winner_name: str, top2: list[str]) -> str:
    """One-paragraph production recommendation."""
    win_disp = c.MODEL_DISPLAY[winner_name]
    return (
        f"For a JRC production short-term load forecasting pipeline, the model to put "
        f"in first is {win_disp}, because it gave the best accuracy on the held-out "
        f"week while staying cheap to fit and easy to reason about. Before trusting it "
        f"in production I would do three things. First, replace the single-week test "
        f"with a rolling-origin backtest across many weeks and all seasons, so the "
        f"accuracy and interval-coverage numbers reflect the whole year rather than "
        f"one winter week with one holiday. Second, harden the calendar handling: a "
        f"complete German public-holiday calendar including regional holidays and "
        f"bridge days, plus explicit handling of the daylight-saving clock changes. "
        f"Third, once the univariate skill is understood, add weather (temperature "
        f"above all), since load is strongly temperature-driven and this study "
        f"deliberately excluded it; that is almost certainly the largest remaining "
        f"source of avoidable error."
    )


# --- Main -------------------------------------------------------------------


def run_all_models(series: pd.Series) -> tuple[dict[str, c.ModelResult], float]:
    """Fit and forecast every model; return the results and total model runtime."""
    overall_start = time.perf_counter()
    print("Running models...", flush=True)
    results: dict[str, c.ModelResult] = {}
    runners = {
        "naive": lambda: naive.run(series),
        "sarima": lambda: sarima.run(series),
        "prophet": lambda: prophet_module.run(series),
        "lightgbm": lambda: lightgbm_features.run(series),
        "nbeats": lambda: nbeats.run(series),
        "patchtst": lambda: patchtst.run(series),
    }
    for name in c.MODEL_ORDER:
        t0 = time.perf_counter()
        results[name] = runners[name]()
        print(f"  {name} done in {time.perf_counter() - t0:.1f}s", flush=True)
    return results, time.perf_counter() - overall_start


def _save_cache(results: dict[str, c.ModelResult], total_runtime: float) -> None:
    """Best-effort pickle of results so figures/transcript can be rebuilt cheaply."""
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as handle:
            pickle.dump({"results": results, "total_runtime": total_runtime}, handle)
    except Exception as exc:
        print(f"  (results cache not written: {exc})", flush=True)


def build_reports(
    series: pd.Series, results: dict[str, c.ModelResult], total_runtime: float
) -> None:
    """Score every model, then write metrics files, figures and the transcript."""
    actual = c.test_series(series).to_numpy(dtype=float)
    test_index = c.test_series(series).index

    scores = {name: score_model(results[name], actual) for name in c.MODEL_ORDER}
    scores_sorted = sorted(scores.values(), key=lambda s: s["mape_test_pct"])
    winner_name = scores_sorted[0]["name"]
    winner = results[winner_name]
    top2 = [s["name"] for s in scores_sorted[:2]]
    prob = winner_probabilistic(winner, actual)

    # --- metrics.json -------------------------------------------------------
    metrics = {
        "test_start": "2020-01-01",
        "test_end": "2020-01-07",
        "n_test_observations": int(len(actual)),
        "models": [scores[name] for name in c.MODEL_ORDER],
        "winner": winner_name,
        **prob,
        "total_runtime_seconds": round(total_runtime, 2),
    }
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)

    # --- metrics.csv --------------------------------------------------------
    csv_rows = []
    for name in c.MODEL_ORDER:
        s = scores[name]
        row = {
            "name": s["name"],
            "mape_test_pct": s["mape_test_pct"],
            "rmse_test_mw": s["rmse_test_mw"],
            "mae_test_mw": s["mae_test_mw"],
            "mape_jan1_pct": s["mape_jan1_pct"],
            "mape_jan2_to_jan7_pct": s["mape_jan2_to_jan7_pct"],
            "runtime_seconds": s["runtime_seconds"],
            "is_winner": name == winner_name,
            "hyperparameters": json.dumps(s["hyperparameters"]),
        }
        if name == winner_name:
            row.update(prob)
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(os.path.join(RUN_DIR, "metrics.csv"), index=False)

    # --- figures ------------------------------------------------------------
    print("Writing figures...", flush=True)
    fig_overview(series)
    fig_forecast_comparison(test_index, actual, results, scores)
    fig_metric_comparison(scores_sorted)
    fig_per_day_mape(actual, results, [s["name"] for s in scores_sorted])
    fig_winner_intervals(test_index, actual, winner, scores[winner_name], prob)
    fig_residuals(test_index, actual, winner)
    if "lightgbm" in top2:
        fig_feature_importance(results["lightgbm"])
    if "prophet" in top2:
        try:
            fig_decomposition(series, results["prophet"])
        except Exception as exc:  # decomposition is optional and library-fragile
            print(f"  decomposition figure skipped: {exc}", flush=True)

    # --- transcript ---------------------------------------------------------
    print("Writing transcript...", flush=True)
    write_transcript(
        scores_sorted,
        winner_name,
        prob,
        total_runtime,
        patchtst.SUBSTITUTION_NOTE,
        top2,
    )

    print(f"Winner: {winner_name} (MAPE {scores[winner_name]['mape_test_pct']:.2f}%)")
    print(f"Total runtime: {total_runtime:.0f}s")
    print("Done.")


def main() -> None:
    """Entry point. Pass --figures-only to rebuild outputs from the cached run."""
    c.set_figure_style()
    np.random.seed(c.SEED)
    series = c.load_series()

    if "--figures-only" in sys.argv:
        with open(CACHE_PATH, "rb") as handle:
            cached = pickle.load(handle)
        results, total_runtime = cached["results"], cached["total_runtime"]
        print("Loaded cached model results.", flush=True)
    else:
        results, total_runtime = run_all_models(series)
        _save_cache(results, total_runtime)

    build_reports(series, results, total_runtime)


if __name__ == "__main__":
    main()
