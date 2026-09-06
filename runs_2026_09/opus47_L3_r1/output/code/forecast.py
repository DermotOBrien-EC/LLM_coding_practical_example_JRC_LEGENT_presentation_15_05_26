"""Orchestrator: run all six models, score, plot, and write outputs."""
from __future__ import annotations

import sys
from pathlib import Path

# Python auto-inserts the script's directory at sys.path[0]. Because this
# script sits in code/ alongside a file named prophet.py, that would shadow
# the installed `prophet` package that darts imports internally. Remove it
# before any other imports.
_this_dir = str(Path(__file__).resolve().parent)
if sys.path and (sys.path[0] == "" or sys.path[0] == _this_dir):
    sys.path.pop(0)

import json
import platform
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load sibling model files by explicit path so that a file named
# `code/prophet.py` does not shadow the installed `prophet` package that
# darts imports internally. We register `common` in sys.modules so that
# each model file's `from common import ...` still resolves.
import importlib.util as _il

_CODE_DIR = Path(__file__).resolve().parent


def _load(local_name: str, filename: str):
    path = _CODE_DIR / filename
    spec = _il.spec_from_file_location(local_name, str(path))
    mod = _il.module_from_spec(spec)
    sys.modules[local_name] = mod
    spec.loader.exec_module(mod)
    return mod


# common must be loaded and registered as "common" first so that every
# model file's `from common import ...` finds it.
common = _load("common", "common.py")
naive = _load("_naive_local", "naive.py")
sarima = _load("_sarima_local", "sarima.py")
prophet_mod = _load("_prophet_local", "prophet.py")
lightgbm_features = _load("_lightgbm_local", "lightgbm_features.py")
nbeats_mod = _load("_nbeats_local", "nbeats.py")
patchtst_mod = _load("_patchtst_local", "patchtst.py")
from common import (
    FIG_DIR,
    MODEL_COLOURS,
    MODEL_DISPLAY,
    MODEL_ORDER,
    RANDOM_SEED,
    Splits,
    apply_plot_style,
    coverage,
    make_splits,
    pinball_loss,
    score_point,
)


import pickle

_CACHE_DIR = _CODE_DIR.parent / ".cache"


def _run_model(name: str, fn, splits: Splits) -> dict[str, Any]:
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_path = _CACHE_DIR / f"{name}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            result = pickle.load(fh)
        print(
            f"[run] {name}: CACHED  MAPE={result['mape_test_pct']:.2f}%  "
            f"RMSE={result['rmse_test_mw']:.0f} MW",
            flush=True,
        )
        return result

    print(f"[run] {name}", flush=True)
    t0 = time.time()
    result: dict[str, Any] = {"name": name}
    if name == "naive":
        pred = fn(splits)
        intervals = None
        hp = naive.hyperparameters()
    elif name == "lightgbm":
        r = fn(splits)
        pred = r.mean
        intervals = r.intervals
        hp = r.hyperparameters
        result["feature_importances"] = r.feature_importances
    else:
        pred, intervals, hp = fn(splits)
    runtime = time.time() - t0
    scores = score_point(splits.test, pred)
    result.update(scores)
    result["runtime_seconds"] = round(runtime, 2)
    result["hyperparameters"] = hp
    result["_pred"] = pred
    # Strip un-picklable fitted-model references from intervals.attrs.
    if intervals is not None:
        # Keep a lightweight copy for pickling without the fitted model reference.
        pickleable_intervals = intervals.copy()
        pickleable_intervals.attrs = {}
        result["_intervals_pickleable"] = pickleable_intervals
    result["_intervals"] = intervals
    print(
        f"[run] {name}: MAPE={scores['mape_test_pct']:.2f}%  "
        f"RMSE={scores['rmse_test_mw']:.0f} MW  runtime={runtime:.1f}s",
        flush=True,
    )
    # Write cache without the fitted-model reference.
    to_cache = {k: v for k, v in result.items() if k != "_intervals"}
    to_cache["_intervals"] = result.get("_intervals_pickleable")
    try:
        with cache_path.open("wb") as fh:
            pickle.dump(to_cache, fh)
    except Exception as e:
        print(f"[warn] {name}: cache write failed: {e}", flush=True)
    return result


def _fig_overview(splits: Splits) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(splits.series.index, splits.series.values, color="0.6", lw=0.5,
            label="Load (2015-2020)")
    test_range = splits.test
    ax.plot(test_range.index, test_range.values, color="tab:red", lw=1.2,
            label="Test week (2020-01-01 to 2020-01-07)")
    ax.axvspan(test_range.index[0], test_range.index[-1], color="tab:red",
               alpha=0.10)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (MW)")
    ax.set_title("German hourly load, 2015-2020, with held-out test week")
    ax.legend(loc="upper right")
    fig.savefig(FIG_DIR / "01_overview.png")
    plt.close(fig)


def _fig_forecast_comparison(splits: Splits, results: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    order = MODEL_ORDER
    ymin = min(splits.test.min(), *(r["_pred"].min() for r in results))
    ymax = max(splits.test.max(), *(r["_pred"].max() for r in results))
    pad = 0.05 * (ymax - ymin)
    for ax, name in zip(axes.flat, order):
        r = next(x for x in results if x["name"] == name)
        pred = r["_pred"]
        ax.plot(splits.test.index, splits.test.values, color="black", lw=1.4,
                label="Observed")
        ax.plot(pred.index, pred.values, color=MODEL_COLOURS[name], lw=1.4,
                label="Forecast")
        ax.set_title(f"{MODEL_DISPLAY[name]}  |  MAPE = {r['mape_test_pct']:.2f}%")
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.tick_params(axis="x", rotation=25)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (UTC)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Load (MW)")
    axes[0, 0].legend(loc="lower left", framealpha=0.9)
    fig.suptitle("Test-week forecasts, six models (2020-01-01 to 2020-01-07)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_forecast_comparison.png")
    plt.close(fig)


def _fig_metric_comparison(results: list[dict]) -> None:
    ordered = sorted(results, key=lambda r: r["mape_test_pct"])
    names = [r["name"] for r in ordered]
    display = [MODEL_DISPLAY[n] for n in names]
    mape_vals = [r["mape_test_pct"] for r in ordered]
    rmse_vals = [r["rmse_test_mw"] / 1000.0 for r in ordered]  # GW for scale
    mae_vals = [r["mae_test_mw"] / 1000.0 for r in ordered]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colours = [MODEL_COLOURS[n] for n in names]

    for ax, values, title, unit, fmt in [
        (axes[0], mape_vals, "MAPE", "%", "{:.2f}"),
        (axes[1], rmse_vals, "RMSE", "GW", "{:.2f}"),
        (axes[2], mae_vals, "MAE", "GW", "{:.2f}"),
    ]:
        bars = ax.bar(range(len(values)), values, color=colours)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(display, rotation=30, ha="right")
        ax.set_title(f"{title}  (test week, sorted by MAPE)")
        ax.set_ylabel(f"{title} ({unit})")
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    fmt.format(v), ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(values) * 1.15)
    fig.suptitle("Point-forecast accuracy on the 168-hour test window",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_metric_comparison.png")
    plt.close(fig)


def _fig_per_day_mape(splits: Splits, results: list[dict]) -> None:
    days = pd.date_range("2020-01-01", "2020-01-07", freq="D", tz="UTC")
    day_labels = [d.strftime("%a %d-%b") for d in days]
    matrix = np.zeros((len(MODEL_ORDER), len(days)))
    for i, name in enumerate(MODEL_ORDER):
        r = next(x for x in results if x["name"] == name)
        pred = r["_pred"]
        for j, d in enumerate(days):
            mask = (splits.test.index >= d) & (splits.test.index < d + pd.Timedelta(days=1))
            y = splits.test[mask].to_numpy()
            p = pred[mask].to_numpy()
            matrix[i, j] = float(np.mean(np.abs((y - p) / y)) * 100.0)

    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.13
    x = np.arange(len(days))
    for i, name in enumerate(MODEL_ORDER):
        offset = (i - 2.5) * width
        ax.bar(x + offset, matrix[i], width, color=MODEL_COLOURS[name],
               label=MODEL_DISPLAY[name])
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Per-day MAPE, test week. Jan 1 is a German public holiday.")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    ax.axvspan(-0.5, 0.5, color="tab:red", alpha=0.08)
    ax.text(0, ax.get_ylim()[1] * 0.95, "Holiday", ha="center", va="top",
            color="tab:red", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_per_day_mape.png")
    plt.close(fig)


def _fig_winner_intervals(splits: Splits, winner: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    obs = splits.test
    pred = winner["_pred"]
    iv = winner["_intervals"]
    colour = MODEL_COLOURS[winner["name"]]
    ax.fill_between(iv.index, iv["lower_95"], iv["upper_95"],
                    color=colour, alpha=0.15, label="95% interval")
    ax.fill_between(iv.index, iv["lower_80"], iv["upper_80"],
                    color=colour, alpha=0.30, label="80% interval")
    ax.plot(pred.index, pred.values, color=colour, lw=1.6, label="Forecast")
    ax.plot(obs.index, obs.values, color="black", lw=1.4, label="Observed")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (MW)")
    cov80 = winner.get("_cov80", float("nan"))
    cov95 = winner.get("_cov95", float("nan"))
    ax.set_title(
        f"Winner: {MODEL_DISPLAY[winner['name']]}. "
        f"Test MAPE = {winner['mape_test_pct']:.2f}%. "
        f"Actual coverage: 80% = {cov80*100:.0f}%, 95% = {cov95*100:.0f}%"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_winner_with_intervals.png")
    plt.close(fig)


def _fig_residuals(splits: Splits, winner: dict) -> None:
    resid = splits.test.to_numpy() - winner["_pred"].reindex(splits.test.index).to_numpy()
    idx = splits.test.index
    hours = idx.hour
    days = idx.dayofweek
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    hour_data = [resid[hours == h] for h in range(24)]
    axes[0].boxplot(hour_data, positions=range(24), widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor=MODEL_COLOURS[winner["name"]],
                                  alpha=0.5),
                    medianprops=dict(color="black"))
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_xlabel("Hour of day (UTC)")
    axes[0].set_ylabel("Residual (MW)  =  observed - forecast")
    axes[0].set_title("Residuals by hour of day")
    axes[0].set_xticks(range(0, 24, 2))

    unique_days = sorted(set(days.tolist()))
    day_data = [resid[days == d] for d in unique_days]
    labels = [day_names[d] for d in unique_days]
    axes[1].boxplot(day_data, positions=range(len(unique_days)), widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor=MODEL_COLOURS[winner["name"]],
                                  alpha=0.5),
                    medianprops=dict(color="black"))
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set_xlabel("Day of test week")
    axes[1].set_ylabel("Residual (MW)")
    axes[1].set_title("Residuals by day of week (test window)")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels)

    fig.suptitle(
        f"Where the winner ({MODEL_DISPLAY[winner['name']]}) misses",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_residuals.png")
    plt.close(fig)


def _fig_feature_importance(lightgbm_result: dict) -> None:
    fi: pd.Series = lightgbm_result["feature_importances"]
    fig, ax = plt.subplots(figsize=(8, 6))
    fi_sorted = fi.sort_values()
    ax.barh(range(len(fi_sorted)), fi_sorted.values,
            color=MODEL_COLOURS["lightgbm"])
    ax.set_yticks(range(len(fi_sorted)))
    ax.set_yticklabels(fi_sorted.index)
    ax.set_xlabel("Gain-based importance (LightGBM)")
    ax.set_title("LightGBM feature importance (final refit on train+val)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_feature_importance.png")
    plt.close(fig)


def _fig_decomposition(prophet_intervals: pd.DataFrame, splits: Splits) -> None:
    m = prophet_intervals.attrs.get("fitted_model")
    if m is None:
        return
    fitted = m.model  # underlying facebook Prophet object
    # Build a future frame spanning fit + test for component extraction.
    fit_series = splits.train_val
    idx = fit_series.index.union(splits.test.index)
    future = pd.DataFrame({"ds": idx.tz_convert("UTC").tz_localize(None)})
    forecast_df = fitted.predict(future)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(forecast_df["ds"], forecast_df["trend"],
                 color=MODEL_COLOURS["prophet"])
    axes[0].set_title("Trend")
    axes[0].set_ylabel("MW")
    if "weekly" in forecast_df.columns:
        axes[1].plot(forecast_df["ds"], forecast_df["weekly"],
                     color=MODEL_COLOURS["prophet"])
    axes[1].set_title("Weekly seasonality")
    axes[1].set_ylabel("MW")
    if "yearly" in forecast_df.columns:
        axes[2].plot(forecast_df["ds"], forecast_df["yearly"],
                     color=MODEL_COLOURS["prophet"])
    axes[2].set_title("Yearly seasonality")
    axes[2].set_ylabel("MW")
    axes[2].set_xlabel("Time (UTC)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_decomposition.png")
    plt.close(fig)


def main() -> None:
    apply_plot_style()
    FIG_DIR.mkdir(exist_ok=True)
    splits = make_splits()
    print(f"[data] rows={len(splits.series)}  "
          f"train={len(splits.train)}  val={len(splits.val)}  "
          f"test={len(splits.test)}  train+val={len(splits.train_val)}",
          flush=True)

    t_all = time.time()
    results = []
    results.append(_run_model("naive", naive.forecast, splits))
    results.append(_run_model("sarima", sarima.forecast, splits))
    results.append(_run_model("prophet", prophet_mod.forecast, splits))
    results.append(_run_model("lightgbm", lightgbm_features.forecast, splits))
    results.append(_run_model("nbeats", nbeats_mod.forecast, splits))
    results.append(_run_model("patchtst", patchtst_mod.forecast, splits))

    winner = min(results, key=lambda r: r["mape_test_pct"])
    if winner["_intervals"] is not None:
        iv = winner["_intervals"]
        y = splits.test.to_numpy()
        winner["_cov80"] = coverage(y, iv["lower_80"].to_numpy(), iv["upper_80"].to_numpy())
        winner["_cov95"] = coverage(y, iv["lower_95"].to_numpy(), iv["upper_95"].to_numpy())
        # For pinball at q=0.5 we use the point forecast (posterior median for
        # SARIMA/NB is the mean; for our quantile-LGBM the mean is the mean regressor).
        winner["_pinball_50"] = pinball_loss(y, winner["_pred"].to_numpy(), 0.5)
        winner["_pinball_10"] = pinball_loss(y, iv["lower_80"].to_numpy(), 0.1)
        winner["_pinball_90"] = pinball_loss(y, iv["upper_80"].to_numpy(), 0.9)
    else:
        winner["_cov80"] = float("nan")
        winner["_cov95"] = float("nan")
        winner["_pinball_50"] = float("nan")
        winner["_pinball_10"] = float("nan")
        winner["_pinball_90"] = float("nan")

    # ---- figures ----
    _fig_overview(splits)
    _fig_forecast_comparison(splits, results)
    _fig_metric_comparison(results)
    _fig_per_day_mape(splits, results)
    _fig_winner_intervals(splits, winner)
    _fig_residuals(splits, winner)

    # Rank models by MAPE for top-2 rules.
    ranked = sorted(results, key=lambda r: r["mape_test_pct"])
    top2 = [r["name"] for r in ranked[:2]]

    lgb_result = next(r for r in results if r["name"] == "lightgbm")
    if "lightgbm" in top2:
        _fig_feature_importance(lgb_result)

    if "prophet" in top2:
        prophet_result = next(r for r in results if r["name"] == "prophet")
        _fig_decomposition(prophet_result["_intervals"], splits)

    # ---- metrics.json / metrics.csv ----
    # Sum of per-model wall-clock at fit time; the current-turn elapsed time
    # includes cache hits and is not the true cost of the study.
    total_runtime = round(sum(r["runtime_seconds"] for r in results), 2)
    metrics = {
        "test_start": "2020-01-01",
        "test_end": "2020-01-07",
        "n_test_observations": int(len(splits.test)),
        "random_seed": RANDOM_SEED,
        "models": [
            {
                "name": r["name"],
                "mape_test_pct": round(r["mape_test_pct"], 4),
                "rmse_test_mw": round(r["rmse_test_mw"], 2),
                "mae_test_mw": round(r["mae_test_mw"], 2),
                "mape_jan1_pct": round(r["mape_jan1_pct"], 4),
                "mape_jan2_to_jan7_pct": round(r["mape_jan2_to_jan7_pct"], 4),
                "runtime_seconds": r["runtime_seconds"],
                "hyperparameters": _sanitize(r["hyperparameters"]),
            }
            for r in results
        ],
        "winner": winner["name"],
        "winner_coverage_80pct": round(winner["_cov80"], 4),
        "winner_coverage_95pct": round(winner["_cov95"], 4),
        "winner_pinball_loss_q10": round(winner["_pinball_10"], 4),
        "winner_pinball_loss_q50": round(winner["_pinball_50"], 4),
        "winner_pinball_loss_q90": round(winner["_pinball_90"], 4),
        "total_runtime_seconds": total_runtime,
    }
    out_dir = Path(__file__).resolve().parent.parent
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # Flatten to csv.
    csv_rows = []
    for r in metrics["models"]:
        csv_rows.append({
            "name": r["name"],
            "mape_test_pct": r["mape_test_pct"],
            "rmse_test_mw": r["rmse_test_mw"],
            "mae_test_mw": r["mae_test_mw"],
            "mape_jan1_pct": r["mape_jan1_pct"],
            "mape_jan2_to_jan7_pct": r["mape_jan2_to_jan7_pct"],
            "runtime_seconds": r["runtime_seconds"],
            "hyperparameters": json.dumps(r["hyperparameters"], default=str),
        })
    pd.DataFrame(csv_rows).to_csv(out_dir / "metrics.csv", index=False)

    # Also cache winner artefacts for the transcript writer.
    (out_dir / ".winner_summary.json").write_text(json.dumps({
        "name": winner["name"],
        "cov80": winner["_cov80"],
        "cov95": winner["_cov95"],
        "pinball_10": winner["_pinball_10"],
        "pinball_50": winner["_pinball_50"],
        "pinball_90": winner["_pinball_90"],
        "top2": top2,
        "uname": platform.platform(),
        "python": platform.python_version(),
    }, indent=2, default=str))
    print(f"[done] total runtime: {total_runtime:.1f}s  winner: {winner['name']}",
          flush=True)


def _sanitize(obj):
    """Make hyperparameter dicts JSON-safe."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


if __name__ == "__main__":
    main()
