"""Orchestrator: run the six models, score them, draw the figures, write the tables.

Usage (from the run directory):

    ../../.venv/bin/python code/forecast.py            # everything, from scratch
    ../../.venv/bin/python code/forecast.py --reuse    # reuse cached model results

Each model's result is cached as a pickle under code/_scratch/results so
the figures and tables can be regenerated without refitting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    FIG_DIR,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    OBSERVED_COLOR,
    RUN_DIR,
    SCRATCH_DIR,
    SEED,
    SQ_FIGSIZE,
    TEST_END,
    TEST_HIGHLIGHT_COLOR,
    TEST_START,
    TS_FIGSIZE,
    ForecastResult,
    Splits,
    apply_figure_style,
    load_series,
    log,
    per_day_mape,
    score_test,
    seed_everything,
    split_series,
)

RESULTS_DIR: Path = SCRATCH_DIR / "results"


MODEL_MODULES: dict[str, str] = {
    "naive": "naive",
    "sarima": "sarima",
    "prophet": "prophet",
    "lightgbm": "lightgbm_features",
    "nbeats": "nbeats",
    "patchtst": "patchtst",
}


def load_model_module(name: str) -> ModuleType:
    """Import exactly one model script.

    prophet.py is loaded by file path because the name `prophet` already
    belongs to the Prophet library once common.py has been imported.
    """
    if name == "prophet":
        spec = importlib.util.spec_from_file_location(
            "prophet_bakeoff", Path(__file__).with_name("prophet.py")
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not load code/prophet.py")
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        return loaded
    module: ModuleType = importlib.import_module(MODEL_MODULES[name])
    return module


def fit_one(name: str, splits: Splits) -> ForecastResult:
    """Fit one model in this process and cache its result."""
    log(f"{name}: fitting")
    seed_everything(SEED)
    result: ForecastResult = load_model_module(name).run(
        splits.train, splits.val, splits.test.index
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / f"{name}.pkl").open("wb") as fh:
        pickle.dump(result, fh)
    log(f"{name}: done in {result.runtime_seconds:.0f} s")
    return result


def run_models(
    splits: Splits, reuse: bool, refit: list[str], subset: list[str] | None = None
) -> dict[str, ForecastResult]:
    """Fit every model (or only `subset`), or load it from the cache when `reuse` allows it.

    Each model is fitted in a fresh child process that imports only its own
    script. LightGBM and PyTorch each bring their own copy of the OpenMP
    runtime, and loading both into one process crashes PyTorch training on
    macOS, so the two never share a process. Models named in `refit` are
    always fitted afresh.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, ForecastResult] = {}
    for name in MODEL_ORDER:
        if subset is not None and name not in subset:
            continue
        cache = RESULTS_DIR / f"{name}.pkl"
        if reuse and name not in refit and cache.exists():
            log(f"{name}: reused cached result")
        else:
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--fit", name], check=True
            )
        with cache.open("rb") as fh:
            results[name] = pickle.load(fh)
    return results


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def environment_info(results: dict[str, ForecastResult]) -> dict[str, Any]:
    import darts
    import lightgbm
    import statsmodels
    import torch

    return {
        "uname": " ".join(platform.uname()),
        "python": sys.version.split()[0],
        "packages": {
            "darts": darts.__version__,
            "lightgbm": lightgbm.__version__,
            "statsmodels": statsmodels.__version__,
            "torch": torch.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "torch_accelerator": next(
            (
                r.hyperparameters.get("accelerator")
                for r in results.values()
                if "accelerator" in r.hyperparameters
            ),
            None,
        ),
        "seed": SEED,
    }


def write_metrics(
    scores: dict[str, dict[str, Any]],
    results: dict[str, ForecastResult],
    actual: pd.Series,
    winner: str,
    total_runtime: float,
) -> None:
    ranked = sorted(scores, key=lambda n: scores[n]["mape_test_pct"])
    models: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for name in ranked:
        sc = scores[name]
        per_day = per_day_mape(actual, results[name].point)
        entry: dict[str, Any] = {
            "name": name,
            "mape_test_pct": round(sc["mape_test_pct"], 4),
            "rmse_test_mw": round(sc["rmse_test_mw"], 2),
            "mae_test_mw": round(sc["mae_test_mw"], 2),
            "mape_jan1_pct": round(sc["mape_jan1_pct"], 4),
            "mape_jan2_to_jan7_pct": round(sc["mape_jan2_to_jan7_pct"], 4),
            "runtime_seconds": round(sc["runtime_seconds"], 1),
            "hyperparameters": sc["hyperparameters"],
            "mape_per_day_pct": {d.strftime("%Y-%m-%d"): round(v, 4) for d, v in per_day.items()},
            "validation": results[name].validation,
        }
        for key in (
            "coverage_80pct",
            "coverage_95pct",
            "pinball_loss_q10",
            "pinball_loss_q50",
            "pinball_loss_q90",
        ):
            if key in sc:
                entry[key] = round(sc[key], 4)
        models.append(entry)
        row = {
            k: v
            for k, v in entry.items()
            if k not in ("hyperparameters", "mape_per_day_pct", "validation")
        }
        row["validation_mape_pct"] = results[name].validation.get("val_mape_pct")
        row["hyperparameters"] = json.dumps(sc["hyperparameters"], default=str)
        rows.append(row)

    wsc = scores[winner]
    payload: dict[str, Any] = {
        "test_start": TEST_START.strftime("%Y-%m-%d"),
        "test_end": TEST_END.strftime("%Y-%m-%d"),
        "n_test_observations": int(len(actual)),
        "models": models,
        "winner": winner,
        "winner_coverage_80pct": round(wsc["coverage_80pct"], 4),
        "winner_coverage_95pct": round(wsc["coverage_95pct"], 4),
        "winner_pinball_loss_q10": round(wsc["pinball_loss_q10"], 2),
        "winner_pinball_loss_q50": round(wsc["pinball_loss_q50"], 2),
        "winner_pinball_loss_q90": round(wsc["pinball_loss_q90"], 2),
        "total_runtime_seconds": round(total_runtime, 1),
        "environment": environment_info(results),
    }
    (RUN_DIR / "metrics.json").write_text(
        json.dumps(payload, indent=2, default=_json_default) + "\n"
    )
    pd.DataFrame(rows).to_csv(RUN_DIR / "metrics.csv", index=False)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def fig_overview(series: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=TS_FIGSIZE)
    ax.plot(
        series.index,
        series / 1000.0,
        color="0.75",
        linewidth=0.4,
        label="Hourly load, 2015-01 to 2020-09",
    )
    test = series.loc[TEST_START:TEST_END]
    ax.axvspan(TEST_START, TEST_END, color=TEST_HIGHLIGHT_COLOR, alpha=0.25, linewidth=0)
    ax.plot(
        test.index,
        test / 1000.0,
        color=TEST_HIGHLIGHT_COLOR,
        linewidth=1.2,
        label="Held-out test week (2020-01-01 to 2020-01-07)",
    )
    ax.annotate(
        "test week",
        xy=(TEST_START, 60),
        xytext=(pd.Timestamp("2019-03-01"), 78),
        color=TEST_HIGHLIGHT_COLOR,
        arrowprops={"arrowstyle": "->", "color": TEST_HIGHLIGHT_COLOR},
    )
    ax.set_title("German national electricity load, hourly (ENTSO-E via Open Power System Data)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (GW)")
    ax.set_ylim(25, 85)
    ax.legend(loc="upper right", ncol=2)
    fig.savefig(FIG_DIR / "01_overview.png")
    plt.close(fig)


def fig_forecast_comparison(
    results: dict[str, ForecastResult], actual: pd.Series, scores: dict[str, dict[str, Any]]
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=TS_FIGSIZE, sharex=True, sharey=True)
    all_values = [actual.to_numpy() / 1000.0] + [
        results[n].point.to_numpy() / 1000.0 for n in MODEL_ORDER
    ]
    lo, hi = float(np.min(np.concatenate(all_values))), float(np.max(np.concatenate(all_values)))
    pad = 0.05 * (hi - lo)
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        ax.plot(
            actual.index, actual / 1000.0, color=OBSERVED_COLOR, linewidth=1.2, label="Observed"
        )
        ax.plot(
            results[name].point.index,
            results[name].point / 1000.0,
            color=MODEL_COLORS[name],
            linewidth=1.4,
            label="Forecast",
        )
        ax.set_title(f"{MODEL_LABELS[name]}: MAPE {scores[name]['mape_test_pct']:.2f} %")
        ax.set_ylim(lo - pad, hi + pad)
        ax.tick_params(axis="x", labelrotation=0)
    for ax in axes[:, 0]:
        ax.set_ylabel("Load (GW)")
    days = pd.date_range(TEST_START, TEST_END + pd.Timedelta(hours=1), freq="D")
    for ax in axes[-1, :]:
        ax.set_xlabel("Date (UTC), January 2020")
        ax.set_xticks(days)
        ax.set_xticklabels([f"{d:%a}\n{d:%d}" for d in days], fontsize=8)
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Test week forecasts, one panel per model (observed in black)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_forecast_comparison.png")
    plt.close(fig)


def fig_metric_comparison(scores: dict[str, dict[str, Any]]) -> None:
    ranked = sorted(scores, key=lambda n: scores[n]["mape_test_pct"])
    metrics = [
        ("mape_test_pct", "MAPE (%)", "{:.2f}"),
        ("rmse_test_mw", "RMSE (MW)", "{:,.0f}"),
        ("mae_test_mw", "MAE (MW)", "{:,.0f}"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=SQ_FIGSIZE, sharey=True)
    y = np.arange(len(ranked))[::-1]
    for ax, (key, label, fmt) in zip(axes, metrics):
        values = [scores[n][key] for n in ranked]
        ax.barh(y, values, color=[MODEL_COLORS[n] for n in ranked], height=0.65)
        for yi, v in zip(y, values):
            ax.text(v, yi, " " + fmt.format(v), va="center", ha="left", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([MODEL_LABELS[n] for n in ranked], fontsize=8)
        ax.set_xlabel(label)
        ax.set_xlim(0, max(values) * 1.22)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Test-week error by model, sorted by MAPE (best at top)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_metric_comparison.png")
    plt.close(fig)


def fig_per_day_mape(
    results: dict[str, ForecastResult], actual: pd.Series, scores: dict[str, dict[str, Any]]
) -> None:
    ranked = sorted(scores, key=lambda n: scores[n]["mape_test_pct"])
    table = pd.DataFrame({n: per_day_mape(actual, results[n].point) for n in ranked})
    fig, ax = plt.subplots(figsize=SQ_FIGSIZE)
    im = ax.imshow(table.to_numpy().T, cmap="Reds", aspect="auto", vmin=0)
    ax.set_xticks(range(len(table.index)))
    ax.set_xticklabels(
        [f"{d:%a}\n{d:%d %b}" + ("\n(holiday)" if d == TEST_START else "") for d in table.index],
        fontsize=8,
    )
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels([MODEL_LABELS[n] for n in ranked], fontsize=8)
    vmax = float(np.nanmax(table.to_numpy()))
    for i, n in enumerate(ranked):
        for j, d in enumerate(table.index):
            v = table.loc[d, n]
            ax.text(
                j,
                i,
                f"{v:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 0.55 * vmax else "black",
            )
    ax.grid(False)
    ax.set_xlabel("Day of the test week (UTC), January 2020")
    ax.set_ylabel("Model (sorted by weekly MAPE)")
    ax.set_title("Per-day MAPE (%) on the test week")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("MAPE (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_per_day_mape.png")
    plt.close(fig)


def fig_winner_with_intervals(
    result: ForecastResult, actual: pd.Series, sc: dict[str, Any]
) -> None:
    q = result.quantiles
    if q is None:
        raise ValueError("winner has no prediction intervals")
    color = MODEL_COLORS[result.name]
    fig, ax = plt.subplots(figsize=TS_FIGSIZE)
    ax.fill_between(
        actual.index,
        q[0.025] / 1000.0,
        q[0.975] / 1000.0,
        color=color,
        alpha=0.15,
        linewidth=0,
        label="95 % prediction interval",
    )
    ax.fill_between(
        actual.index,
        q[0.1] / 1000.0,
        q[0.9] / 1000.0,
        color=color,
        alpha=0.35,
        linewidth=0,
        label="80 % prediction interval",
    )
    ax.plot(
        result.point.index,
        result.point / 1000.0,
        color=color,
        linewidth=1.5,
        label="Point forecast",
    )
    ax.plot(actual.index, actual / 1000.0, color=OBSERVED_COLOR, linewidth=1.2, label="Observed")
    ax.set_title(
        f"{MODEL_LABELS[result.name]}: test MAPE {sc['mape_test_pct']:.2f} %, "
        f"80 % interval covers {100 * sc['coverage_80pct']:.1f} % of hours, 95 % interval covers {100 * sc['coverage_95pct']:.1f} %"
    )
    ax.set_xlabel("Date (UTC), January 2020")
    ax.set_ylabel("Load (GW)")
    ax.legend(loc="lower right", ncol=4)
    fig.savefig(FIG_DIR / "05_winner_with_intervals.png")
    plt.close(fig)


def fig_residuals(result: ForecastResult, actual: pd.Series) -> None:
    resid = (actual - result.point) / 1000.0
    color = MODEL_COLORS[result.name]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=TS_FIGSIZE, gridspec_kw={"width_ratios": [3, 1.6]})
    by_hour = [resid[resid.index.hour == h].to_numpy() for h in range(24)]
    ax1.axhline(0, color="0.4", linewidth=0.8)
    ax1.boxplot(
        by_hour,
        positions=range(24),
        widths=0.6,
        patch_artist=True,
        boxprops={"facecolor": color, "alpha": 0.5},
        medianprops={"color": "black"},
        flierprops={"markersize": 3},
    )
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([str(h) for h in range(0, 24, 2)])
    ax1.set_xlabel("Hour of day (UTC)")
    ax1.set_ylabel("Residual, observed minus forecast (GW)")
    ax1.set_title("By hour of day (7 values per box)")
    days = pd.date_range(TEST_START, TEST_END, freq="D")
    by_day = [resid[resid.index.normalize() == d].to_numpy() for d in days]
    ax2.axhline(0, color="0.4", linewidth=0.8)
    ax2.boxplot(
        by_day,
        positions=range(7),
        widths=0.6,
        patch_artist=True,
        boxprops={"facecolor": color, "alpha": 0.5},
        medianprops={"color": "black"},
        flierprops={"markersize": 3},
    )
    ax2.set_xticks(range(7))
    ax2.set_xticklabels([f"{d:%a %d}" + ("*" if d == TEST_START else "") for d in days], fontsize=8)
    ax2.set_xlabel("Day of test week (* = public holiday)")
    ax2.set_ylabel("Residual, observed minus forecast (GW)")
    ax2.set_title("By day (24 values per box)")
    fig.suptitle(
        f"{MODEL_LABELS[result.name]} residuals on the test week (positive = under-forecast)"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_residuals.png")
    plt.close(fig)


def fig_feature_importance(result: ForecastResult) -> None:
    imp: pd.Series = result.extras["feature_importance_gain"]
    share = 100.0 * imp / imp.sum()
    fig, ax = plt.subplots(figsize=SQ_FIGSIZE)
    y = np.arange(len(share))[::-1]
    ax.barh(y, share.to_numpy(), color=MODEL_COLORS["lightgbm"], height=0.65)
    for yi, v in zip(y, share.to_numpy()):
        ax.text(v, yi, f" {v:.1f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(share.index, fontsize=8)
    ax.set_xlabel("Share of total split gain (%)")
    ax.set_xlim(0, float(share.max()) * 1.15)
    ax.set_title("LightGBM feature importance (gain), point model refit on train + validation")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_feature_importance.png")
    plt.close(fig)


def fig_decomposition(result: ForecastResult) -> None:
    comps: pd.DataFrame = result.extras["components"]
    color = MODEL_COLORS["prophet"]
    fig, axes = plt.subplots(2, 2, figsize=TS_FIGSIZE)
    ax = axes[0, 0]
    ax.plot(comps.index, comps["trend"] / 1000.0, color=color)
    ax.set_title("Trend")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (GW)")
    ax = axes[0, 1]
    year = comps.loc[pd.Timestamp("2019-01-01") : pd.Timestamp("2019-12-31 23:00")]
    ax.plot(year.index, year["yearly"] / 1000.0, color=color)
    ax.set_title("Yearly component (shown over 2019)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Effect on load (GW)")
    ax = axes[1, 0]
    week = comps.loc[pd.Timestamp("2019-12-02") : pd.Timestamp("2019-12-08 23:00")]
    ax.plot(np.arange(len(week)) / 24.0, week["weekly"] / 1000.0, color=color)
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_title("Weekly component")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Effect on load (GW)")
    ax = axes[1, 1]
    daily_cols = [
        c
        for c in comps.columns
        if (c == "daily" or c.startswith("daily_")) and not c.endswith(("_lower", "_upper"))
    ]
    for col, style in zip(daily_cols, ["-", "--", ":"]):
        if col == "daily_weekday":
            day = comps.loc[pd.Timestamp("2019-12-03") : pd.Timestamp("2019-12-03 23:00")]
        elif col == "daily_weekend":
            day = comps.loc[pd.Timestamp("2019-12-07") : pd.Timestamp("2019-12-07 23:00")]
        else:
            day = comps.loc[pd.Timestamp("2019-12-03") : pd.Timestamp("2019-12-03 23:00")]
        ax.plot(
            range(24), day[col] / 1000.0, color=color, linestyle=style, label=col.replace("_", " ")
        )
    ax.set_xticks(range(0, 24, 4))
    ax.set_title("Daily component(s)")
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Effect on load (GW)")
    ax.legend(loc="best")
    fig.suptitle("Prophet components, model refit on train + validation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_decomposition.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse", action="store_true", help="reuse cached per-model results when present"
    )
    parser.add_argument(
        "--only", default=None, help="comma-separated models to refit; the rest come from the cache"
    )
    parser.add_argument("--fit", default=None, help=argparse.SUPPRESS)  # child-process entry point
    parser.add_argument(
        "--models",
        default=None,
        help="comma-separated models to fit and cache in this invocation, then stop (no figures or tables)",
    )
    args = parser.parse_args()

    apply_figure_style()
    FIG_DIR.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    series = load_series()
    splits = split_series(series)
    if args.fit:
        fit_one(args.fit, splits)
        return
    if args.models:
        # Fit a subset now (useful for long sessions); figures come later with --reuse.
        run_models(splits, reuse=True, refit=[], subset=args.models.split(","))
        return
    refit = args.only.split(",") if args.only else []
    results = run_models(splits, reuse=args.reuse or bool(refit), refit=refit)

    actual = splits.test
    scores = {name: score_test(res, actual) for name, res in results.items()}
    ranked = sorted(scores, key=lambda n: scores[n]["mape_test_pct"])
    winner = ranked[0]
    total_runtime = sum(res.runtime_seconds for res in results.values())
    log(
        "ranking by test MAPE: "
        + ", ".join(f"{n} {scores[n]['mape_test_pct']:.2f} %" for n in ranked)
    )

    write_metrics(scores, results, actual, winner, total_runtime)
    fig_overview(series)
    fig_forecast_comparison(results, actual, scores)
    fig_metric_comparison(scores)
    fig_per_day_mape(results, actual, scores)
    fig_winner_with_intervals(results[winner], actual, scores[winner])
    fig_residuals(results[winner], actual)
    if "lightgbm" in ranked[:2]:
        fig_feature_importance(results["lightgbm"])
    if "prophet" in ranked[:2]:
        fig_decomposition(results["prophet"])
    log(
        f"done: winner {winner}, wall clock for this invocation {time.perf_counter() - t0:.0f} s, model runtime {total_runtime:.0f} s"
    )


if __name__ == "__main__":
    main()
