"""Orchestrator: run all six models, score them, write figures and tables.

Usage (from the run directory):

    ../../.venv/bin/python code/forecast.py            # everything
    ../../.venv/bin/python code/forecast.py --only sarima,nbeats
    ../../.venv/bin/python code/forecast.py --figures-only

Each model's forecast is pickled under cache/ as soon as it finishes, so a
partial run can be resumed and the figures can be redrawn without refitting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import platform
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    FIG_DIR,
    HORIZON,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_NAMES,
    OBSERVED_COLOR,
    QUANTILES,
    RUN_DIR,
    SEED,
    TEST_END,
    TEST_START,
    ForecastResult,
    evaluate_intervals,
    evaluate_point,
    load_series,
    mape,
    save_figure,
    set_style,
    split,
)

CACHE_DIR = RUN_DIR / "cache"
CODE_DIR = RUN_DIR / "code"


# --------------------------------------------------------------------------
# Running the models
# --------------------------------------------------------------------------
def run_model(name: str, train: pd.Series, val: pd.Series) -> ForecastResult:
    """Import the model's module lazily so a broken model cannot block the others."""
    if name == "naive":
        import naive as module
    elif name == "sarima":
        import sarima as module
    elif name == "prophet":
        # Loaded by file path: the name "prophet" belongs to the installed package.
        spec = importlib.util.spec_from_file_location("bakeoff_prophet", CODE_DIR / "prophet.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif name == "lightgbm":
        import lightgbm_features as module
    elif name == "nbeats":
        import nbeats as module
    elif name == "patchtst":
        import patchtst as module
    else:
        raise ValueError(name)
    return module.run(train, val, HORIZON)


def cache_path(name: str) -> Any:
    return CACHE_DIR / f"{name}.pkl"


def load_cached(name: str) -> ForecastResult:
    with open(cache_path(name), "rb") as fh:
        return pickle.load(fh)


def store(result: ForecastResult) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    with open(cache_path(result.name), "wb") as fh:
        pickle.dump(result, fh)


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def write_metrics(
    scores: list[dict[str, Any]],
    winner: str,
    winner_intervals: dict[str, float],
    total_runtime: float,
    notes: dict[str, Any],
) -> None:
    ordered = sorted(scores, key=lambda s: s["mape_test_pct"])
    payload = {
        "test_start": TEST_START.strftime("%Y-%m-%d"),
        "test_end": TEST_END.strftime("%Y-%m-%d"),
        "n_test_observations": HORIZON,
        "models": ordered,
        "winner": winner,
        "winner_coverage_80pct": winner_intervals["coverage_80pct"],
        "winner_coverage_95pct": winner_intervals["coverage_95pct"],
        "winner_pinball_loss_q10": winner_intervals["pinball_loss_q10"],
        "winner_pinball_loss_q50": winner_intervals["pinball_loss_q50"],
        "winner_pinball_loss_q90": winner_intervals["pinball_loss_q90"],
        "total_runtime_seconds": total_runtime,
        "random_seed": SEED,
        "environment": {"uname": " ".join(platform.uname()), "python": platform.python_version()},
        "notes": notes,
    }
    with open(RUN_DIR / "metrics.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=float)

    rows = []
    for s in ordered:
        row = {k: v for k, v in s.items() if k not in ("per_day_mape_pct", "hyperparameters")}
        row["label"] = MODEL_LABELS[s["name"]]
        for day, value in s["per_day_mape_pct"].items():
            row[f"mape_{day}_pct"] = value
        if s["name"] == winner:
            row.update({f"winner_{k}": v for k, v in winner_intervals.items()})
        row["hyperparameters"] = json.dumps(s["hyperparameters"], default=str)
        rows.append(row)
    pd.DataFrame(rows).to_csv(RUN_DIR / "metrics.csv", index=False)


def write_forecasts(results: dict[str, ForecastResult], test: pd.Series) -> None:
    frame = pd.DataFrame({"observed_mw": test})
    for name, res in results.items():
        frame[f"{name}_point_mw"] = res.point.reindex(test.index)
        if res.quantiles is not None:
            for tau in QUANTILES:
                frame[f"{name}_q{tau:g}_mw"] = res.quantiles[tau].reindex(test.index)
    frame.to_csv(RUN_DIR / "forecasts.csv", float_format="%.1f")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_overview(series: pd.Series, test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(series.index, series.values / 1000, color="0.75", linewidth=0.4, label="Observed load")
    ax.axvspan(test.index[0], test.index[-1], color="red", alpha=0.25, label="Test window (2020-01-01 to 07)")
    ax.plot(test.index, test.values / 1000, color="red", linewidth=1.0)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (GW)")
    ax.set_title("German national electricity load, hourly, 2015-01-01 to 2020-09-30")
    ax.legend(loc="upper right", frameon=False)
    save_figure(fig, "01_overview.png")


def fig_forecast_comparison(results: dict[str, ForecastResult], test: pd.Series, scores: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey=True)
    all_values = [test.values] + [results[n].point.reindex(test.index).values for n in MODEL_NAMES]
    lo, hi = np.min(all_values) / 1000, np.max(all_values) / 1000
    pad = (hi - lo) * 0.05
    for ax, name in zip(axes.ravel(), MODEL_NAMES):
        fc = results[name].point.reindex(test.index)
        ax.plot(test.index, test.values / 1000, color=OBSERVED_COLOR, linewidth=1.2, label="Observed")
        ax.plot(fc.index, fc.values / 1000, color=MODEL_COLORS[name], linewidth=1.2, label="Forecast")
        ax.set_title(f"{MODEL_LABELS[name]}: MAPE {scores[name]['mape_test_pct']:.2f} %")
        ax.set_ylim(lo - pad, hi + pad)
        ax.legend(loc="lower right", frameon=False)
        ax.tick_params(axis="x", labelrotation=30)
    for ax in axes[:, 0]:
        ax.set_ylabel("Load (GW)")
    for ax in axes[1, :]:
        ax.set_xlabel("Time (UTC)")
    fig.suptitle("Test week forecasts, 2020-01-01 to 2020-01-07 (Wednesday 1 January is a public holiday)")
    fig.tight_layout()
    save_figure(fig, "02_forecast_comparison.png")


def fig_metric_comparison(scores: dict[str, dict]) -> None:
    order = sorted(MODEL_NAMES, key=lambda n: scores[n]["mape_test_pct"])
    metrics = [("mape_test_pct", "MAPE (%)", 1.0), ("rmse_test_mw", "RMSE (GW)", 1e-3), ("mae_test_mw", "MAE (GW)", 1e-3)]
    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    x = np.arange(len(order))
    for ax, (key, label, scale) in zip(axes, metrics):
        values = [scores[n][key] * scale for n in order]
        bars = ax.bar(x, values, color=[MODEL_COLORS[n] for n in order], width=0.7)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(label)
        ax.set_ylim(0, max(values) * 1.18)
        ax.grid(axis="x", visible=False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([MODEL_LABELS[n].replace(" (", "\n(") for n in order], fontsize=7)
    axes[0].set_title("Test-week error by model, sorted by MAPE")
    fig.tight_layout()
    save_figure(fig, "03_metric_comparison.png")


def fig_per_day_mape(scores: dict[str, dict], test: pd.Series) -> None:
    days = sorted(scores["naive"]["per_day_mape_pct"])
    day_labels = []
    for d in days:
        ts = pd.Timestamp(d)
        tag = " (holiday)" if ts.month == 1 and ts.day == 1 else ""
        day_labels.append(f"{ts.strftime('%a %d %b')}{tag}")
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.8 / len(MODEL_NAMES)
    x = np.arange(len(days))
    for i, name in enumerate(MODEL_NAMES):
        vals = [scores[name]["per_day_mape_pct"][d] for d in days]
        ax.bar(x + (i - 2.5) * width, vals, width=width, color=MODEL_COLORS[name], label=MODEL_LABELS[name])
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels)
    ax.set_xlabel("Test day (UTC calendar day)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Per-day MAPE on the test week")
    top = max(scores[n]["per_day_mape_pct"][d] for n in MODEL_NAMES for d in days)
    ax.set_ylim(0, top * 1.3)  # headroom so the legend never covers a bar
    ax.legend(loc="upper left", frameon=False, ncol=3)
    ax.grid(axis="x", visible=False)
    save_figure(fig, "04_per_day_mape.png")


def fig_winner_intervals(res: ForecastResult, test: pd.Series, score: dict, intervals: dict[str, float]) -> None:
    name = res.name
    colour = MODEL_COLORS[name]
    q = {tau: res.quantiles[tau].reindex(test.index) / 1000 for tau in QUANTILES}
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.fill_between(test.index, q[0.025], q[0.975], color=colour, alpha=0.15, label="95 % interval")
    ax.fill_between(test.index, q[0.1], q[0.9], color=colour, alpha=0.35, label="80 % interval")
    ax.plot(test.index, res.point.reindex(test.index) / 1000, color=colour, linewidth=1.4, label="Point forecast")
    ax.plot(test.index, test.values / 1000, color=OBSERVED_COLOR, linewidth=1.2, label="Observed")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (GW)")
    ax.set_title(
        f"Winner: {MODEL_LABELS[name]}. Test MAPE {score['mape_test_pct']:.2f} %; "
        f"coverage 80 % band: {intervals['coverage_80pct'] * 100:.0f} %, "
        f"95 % band: {intervals['coverage_95pct'] * 100:.0f} %"
    )
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, "05_winner_with_intervals.png")


def fig_residuals(res: ForecastResult, test: pd.Series) -> None:
    resid = (test - res.point.reindex(test.index)) / 1000
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), gridspec_kw={"width_ratios": [3, 2]})
    colour = MODEL_COLORS[res.name]
    hours = range(24)
    axes[0].boxplot([resid[resid.index.hour == h].values for h in hours], positions=list(hours), widths=0.6,
                    patch_artist=True, boxprops={"facecolor": colour, "alpha": 0.5}, medianprops={"color": "black"})
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Hour of day (UTC)")
    axes[0].set_ylabel("Residual, observed minus forecast (GW)")
    axes[0].set_title("By hour of day (7 values per box)")
    axes[0].set_xticks(list(hours)[::3])
    axes[0].set_xticklabels([str(h) for h in hours][::3])
    days = sorted(set(resid.index.date))
    axes[1].boxplot([resid[resid.index.date == d].values for d in days], positions=range(len(days)), widths=0.6,
                    patch_artist=True, boxprops={"facecolor": colour, "alpha": 0.5}, medianprops={"color": "black"})
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(range(len(days)))
    axes[1].set_xticklabels([pd.Timestamp(d).strftime("%a\n%d %b") for d in days])
    axes[1].set_xlabel("Test day")
    axes[1].set_ylabel("Residual, observed minus forecast (GW)")
    axes[1].set_title("By day of the test week (24 values per box)")
    fig.suptitle(f"Residuals of the winning model ({MODEL_LABELS[res.name]}) on the test week")
    fig.tight_layout()
    save_figure(fig, "06_residuals.png")


def fig_feature_importance(res: ForecastResult) -> None:
    imp = res.extras["feature_importance_gain"].sort_values()
    share = imp / imp.sum() * 100
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(share.index, share.values, color=MODEL_COLORS["lightgbm"])
    for i, v in enumerate(share.values):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_ylabel("Feature")
    ax.set_title("LightGBM feature importance (gain), final model")
    ax.set_xlim(0, share.max() * 1.15)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, "07_feature_importance.png")


def fig_decomposition(res: ForecastResult, history: pd.Series) -> None:
    model = res.extras["prophet_model"]
    frame = pd.DataFrame({"ds": history.index.tz_localize(None)})
    comp = model.predict(frame).set_index(history.index)
    fig, axes = plt.subplots(4, 1, figsize=(11, 9))
    colour = MODEL_COLORS["prophet"]
    axes[0].plot(comp.index, comp["trend"] / 1000, color=colour)
    axes[0].set_title("Trend")
    axes[0].set_ylabel("Load (GW)")
    axes[0].set_xlabel("Time (UTC)")
    one_year = comp.loc["2019-01-01":"2019-12-31 23:00"]
    axes[1].plot(one_year.index, one_year["yearly"] / 1000, color=colour)
    axes[1].set_title("Yearly seasonality (shown over 2019)")
    axes[1].set_ylabel("Effect (GW)")
    axes[1].set_xlabel("Time (UTC)")
    one_week = comp.loc["2019-12-02":"2019-12-08 23:00"]  # a plain Monday-to-Sunday week
    axes[2].plot(one_week.index, one_week["weekly"] / 1000, color=colour)
    axes[2].set_title("Weekly seasonality (Monday 2 Dec to Sunday 8 Dec 2019)")
    axes[2].set_ylabel("Effect (GW)")
    axes[2].set_xlabel("Time (UTC)")
    one_day = comp.loc["2019-12-03":"2019-12-03 23:00"]
    axes[3].plot(one_day.index.hour, one_day["daily"] / 1000, color=colour)
    axes[3].set_title("Daily seasonality")
    axes[3].set_ylabel("Effect (GW)")
    axes[3].set_xlabel("Hour of day (UTC)")
    fig.suptitle("Prophet components of the final model (fitted on 2015-01-01 to 2019-12-31)")
    fig.tight_layout()
    save_figure(fig, "08_decomposition.png")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma-separated model names to (re)run")
    parser.add_argument("--figures-only", action="store_true", help="reuse every cached forecast")
    args = parser.parse_args()

    started = time.perf_counter()
    set_style()
    series = load_series()
    train, val, test = split(series)
    print(f"train {len(train)} rows, validation {len(val)} rows, test {len(test)} rows", flush=True)

    to_run = [] if args.figures_only else (args.only.split(",") if args.only else MODEL_NAMES)
    results: dict[str, ForecastResult] = {}
    for name in MODEL_NAMES:
        if name in to_run:
            print(f"== {name}", flush=True)
            result = run_model(name, train, val)
            store(result)
            print(f"   done in {result.runtime_seconds:.0f} s, validation MAPE {result.validation_mape_pct:.2f} %", flush=True)
        else:
            result = load_cached(name)
        results[name] = result

    scores = {name: evaluate_point(test, res) for name, res in results.items()}
    winner = min(scores, key=lambda n: scores[n]["mape_test_pct"])
    winner_intervals = evaluate_intervals(test, results[winner])
    ranking = sorted(MODEL_NAMES, key=lambda n: scores[n]["mape_test_pct"])
    print("ranking by test MAPE:", ", ".join(f"{n} {scores[n]['mape_test_pct']:.2f} %" for n in ranking))

    notes: dict[str, Any] = {
        "patchtst_slot": "darts 0.41.0 has no PatchTSTModel; TSMixerModel is used in the patchtst slot",
    }
    # Supplementary, not part of the bake-off: how LightGBM would score if it
    # were run one day ahead, with real observations in its 24 h lag and
    # rolling features. This uses test observations as inputs, so it is
    # reported separately and never enters the ranking.
    from lightgbm_features import FEATURES, build_features

    table = build_features(pd.concat([train, val, test])).loc[test.index]
    known_lag_fc = results["lightgbm"].extras["point_model"].predict(table[FEATURES])
    notes["lightgbm_day_ahead_mode_mape_pct"] = mape(test.values, known_lag_fc)

    fig_overview(series, test)
    fig_forecast_comparison(results, test, scores)
    fig_metric_comparison(scores)
    fig_per_day_mape(scores, test)
    fig_winner_intervals(results[winner], test, scores[winner], winner_intervals)
    fig_residuals(results[winner], test)
    if "lightgbm" in ranking[:2]:
        fig_feature_importance(results["lightgbm"])
    if "prophet" in ranking[:2]:
        fig_decomposition(results["prophet"], pd.concat([train, val]))

    write_forecasts(results, test)
    total = time.perf_counter() - started
    write_metrics(list(scores.values()), winner, winner_intervals, total, notes)
    print(f"winner {winner}; wrote metrics.json, metrics.csv, forecasts.csv and figures to {FIG_DIR}")
    print(f"orchestrator wall-clock {total:.0f} s (model fitting {sum(r.runtime_seconds for r in results.values()):.0f} s)")


if __name__ == "__main__":
    main()
