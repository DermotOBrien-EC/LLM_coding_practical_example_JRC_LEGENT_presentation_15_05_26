"""Orchestrator: run the six models, score them, draw the figures.

    ../../.venv/bin/python code/forecast.py                 # everything
    ../../.venv/bin/python code/forecast.py --models sarima,prophet
    ../../.venv/bin/python code/forecast.py --from-cache    # figures + metrics only

Each model writes its test-week forecast and metadata to forecasts/ (a
small cache), and this script reads all six back to produce metrics.json,
metrics.csv, the figures, and forecasts/summary_tables.md (the markdown
tables pasted into transcript.md).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

RUNNERS: dict[str, str] = {
    "naive": "naive",
    "sarima": "sarima",
    "prophet": "prophet",
    "lightgbm": "lightgbm_features",
    "nbeats": "nbeats",
    "patchtst": "patchtst",
}
DAY_LABELS: list[str] = ["Wed 1 Jan (holiday)", "Thu 2 Jan", "Fri 3 Jan", "Sat 4 Jan", "Sun 5 Jan", "Mon 6 Jan", "Tue 7 Jan"]
DAY_TICKS: list[str] = ["Wed 1\n(holiday)", "Thu 2", "Fri 3", "Sat 4", "Sun 5", "Mon 6", "Tue 7"]
SHORT_LABELS: dict[str, str] = {
    "naive": "Naive",
    "sarima": "SARIMA",
    "prophet": "Prophet",
    "lightgbm": "LightGBM",
    "nbeats": "N-BEATS",
    "patchtst": "TSMixer",
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_models(results: dict[str, common.ModelResult], actual: pd.Series) -> pd.DataFrame:
    rows = []
    jan1 = actual.loc["2020-01-01"]
    rest = actual.loc["2020-01-02":"2020-01-07"]
    for name in common.MODEL_ORDER:
        r = results[name]
        rows.append(
            {
                "name": name,
                "mape_test_pct": common.mape(actual, r.point),
                "rmse_test_mw": common.rmse(actual, r.point),
                "mae_test_mw": common.mae(actual, r.point),
                "mape_jan1_pct": common.mape(jan1, r.point),
                "mape_jan2_to_jan7_pct": common.mape(rest, r.point),
                "runtime_seconds": r.runtime_seconds,
                "validation_mape_pct": r.validation_mape_pct,
            }
        )
    return pd.DataFrame(rows).set_index("name").sort_values("mape_test_pct")


def winner_stats(r: common.ModelResult, actual: pd.Series) -> dict[str, float]:
    assert r.quantiles is not None
    q = r.quantiles
    return {
        "winner_coverage_80pct": common.coverage(actual, q[0.1], q[0.9]),
        "winner_coverage_95pct": common.coverage(actual, q[0.025], q[0.975]),
        "winner_pinball_loss_q10": common.pinball_loss(actual, q[0.1], 0.1),
        "winner_pinball_loss_q50": common.pinball_loss(actual, q[0.5], 0.5),
        "winner_pinball_loss_q90": common.pinball_loss(actual, q[0.9], 0.9),
    }


def write_metrics(scores: pd.DataFrame, results: dict[str, common.ModelResult], wstats: dict[str, float], winner: str, total_runtime: float) -> None:
    models = []
    for name, row in scores.iterrows():
        models.append(
            {
                "name": name,
                "mape_test_pct": round(float(row["mape_test_pct"]), 4),
                "rmse_test_mw": round(float(row["rmse_test_mw"]), 2),
                "mae_test_mw": round(float(row["mae_test_mw"]), 2),
                "mape_jan1_pct": round(float(row["mape_jan1_pct"]), 4),
                "mape_jan2_to_jan7_pct": round(float(row["mape_jan2_to_jan7_pct"]), 4),
                "runtime_seconds": round(float(row["runtime_seconds"]), 1),
                "hyperparameters": results[str(name)].hyperparameters,
            }
        )
    payload: dict[str, Any] = {
        "test_start": "2020-01-01",
        "test_end": "2020-01-07",
        "n_test_observations": 168,
        "models": models,
        "winner": winner,
        **{k: round(v, 4) for k, v in wstats.items()},
        "total_runtime_seconds": round(total_runtime, 1),
    }
    (common.RUN_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))

    flat = []
    for m in models:
        rec = {k: v for k, v in m.items() if k != "hyperparameters"}
        rec["hyperparameters"] = json.dumps(m["hyperparameters"])
        rec["is_winner"] = m["name"] == winner
        for k, v in wstats.items():
            rec[k] = round(v, 4) if m["name"] == winner else None
        rec.update({"test_start": "2020-01-01", "test_end": "2020-01-07", "n_test_observations": 168, "total_runtime_seconds": round(total_runtime, 1)})
        flat.append(rec)
    pd.DataFrame(flat).to_csv(common.RUN_DIR / "metrics.csv", index=False)


def write_summary_tables(scores: pd.DataFrame, per_day: pd.DataFrame, results: dict[str, common.ModelResult], wstats: dict[str, float], winner: str) -> None:
    lines = ["| Model | Test MAPE (%) | RMSE (MW) | MAE (MW) | MAPE 1 Jan, holiday (%) | MAPE 2 to 7 Jan (%) | Validation MAPE (%) | Runtime (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in scores.iterrows():
        v = row["validation_mape_pct"]
        lines.append(
            f"| {common.MODEL_LABELS[str(name)]} | {row['mape_test_pct']:.2f} | {row['rmse_test_mw']:.0f} | {row['mae_test_mw']:.0f} | {row['mape_jan1_pct']:.2f} | {row['mape_jan2_to_jan7_pct']:.2f} | {v:.2f} | {row['runtime_seconds']:.0f} |"
        )
    lines += ["", "| Model | " + " | ".join(DAY_LABELS) + " |", "|---|" + "---:|" * 7]
    for name in scores.index:
        lines.append(f"| {common.MODEL_LABELS[str(name)]} | " + " | ".join(f"{v:.2f}" for v in per_day.loc[str(name)]) + " |")
    lines += ["", f"Winner: {winner}", *[f"{k}: {v:.4f}" for k, v in wstats.items()]]
    lines += ["", "Selected hyperparameters:"]
    for name in common.MODEL_ORDER:
        lines.append(f"- {name}: {json.dumps(results[name].hyperparameters)}")
    lines += ["", "Validation tables:"]
    for name in common.MODEL_ORDER:
        lines.append(f"- {name}: {json.dumps(results[name].validation_table)}")
    common.FORECAST_DIR.mkdir(exist_ok=True)
    (common.FORECAST_DIR / "summary_tables.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, name: str) -> None:
    common.FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(common.FIGURE_DIR / name, dpi=common.DPI, bbox_inches="tight")
    plt.close(fig)


def _day_ticks(ax: plt.Axes) -> None:
    days = pd.date_range("2020-01-01", "2020-01-08", freq="D")
    ax.set_xticks(days)
    ax.set_xticklabels([d.strftime("%a %d") for d in days])
    ax.set_xlim(days[0], days[-1])


def fig01_overview(s: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=common.TS_FIGSIZE)
    ax.plot(s.index, s.to_numpy(), color="0.75", linewidth=0.3, label="Observed load, 2015 to 2020")
    test = common.test_slice(s)
    ax.plot(test.index, test.to_numpy(), color=common.TEST_WINDOW_COLOR, linewidth=1.2, label="Test window, 1 to 7 Jan 2020 (168 h)")
    ax.axvspan(common.TEST_START, common.TEST_END, color=common.TEST_WINDOW_COLOR, alpha=0.15)
    ax.axvline(common.TRAIN_END, color="0.3", linestyle="--", linewidth=0.8, label="End of Train (30 Sep 2019); Validation runs to 31 Dec 2019")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (MW)")
    ax.set_title("German national electricity load, hourly, ENTSO-E via Open Power System Data")
    ax.legend(loc="upper left", frameon=True)
    _save(fig, "01_overview.png")


def fig02_forecast_comparison(results: dict[str, common.ModelResult], actual: pd.Series, scores: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=common.TS_FIGSIZE, sharey=True, sharex=True)
    for ax, name in zip(axes.ravel(), common.MODEL_ORDER):
        r = results[name]
        ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), color="0.9", label="1 Jan (public holiday)")
        ax.plot(actual.index, actual.to_numpy(), color=common.OBSERVED_COLOR, linewidth=1.0, label="Observed")
        ax.plot(r.point.index, r.point.to_numpy(), color=common.MODEL_COLORS[name], linewidth=1.2, label="Point forecast (model colour)")
        ax.set_title(f"{common.MODEL_LABELS[name]}: MAPE {scores.loc[name, 'mape_test_pct']:.2f} %")
        _day_ticks(ax)
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Load (MW)")
    for ax in axes[1, :]:
        ax.set_xlabel("Test week, 2020 (UTC date)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Point forecasts for the held-out test week (1 to 7 January 2020)")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, "02_forecast_comparison.png")


def fig03_metric_comparison(scores: pd.DataFrame) -> None:
    metrics = [("mape_test_pct", "MAPE (%)", "{:.2f}"), ("rmse_test_mw", "RMSE (MW)", "{:.0f}"), ("mae_test_mw", "MAE (MW)", "{:.0f}")]
    fig, axes = plt.subplots(1, 3, figsize=common.TS_FIGSIZE)
    names = [str(n) for n in scores.index]
    x = np.arange(len(names))
    for ax, (col, label, fmt) in zip(axes, metrics):
        vals = scores[col].to_numpy(dtype=float)
        bars = ax.bar(x, vals, color=[common.MODEL_COLORS[n] for n in names], width=0.7)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS[n] for n in names], rotation=35, ha="right")
        ax.set_ylabel(label)
        ax.set_title(label.split(" (")[0])
        ax.set_ylim(0, vals.max() * 1.15)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Test-week error metrics, models sorted by MAPE (lower is better)")
    fig.tight_layout()
    _save(fig, "03_metric_comparison.png")


def fig04_per_day_mape(per_day: pd.DataFrame, scores: pd.DataFrame) -> None:
    order = [str(n) for n in scores.index]
    mat = per_day.loc[order].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=common.SQUARE_FIGSIZE)
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(7))
    ax.set_xticklabels(DAY_TICKS, fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([common.MODEL_LABELS[n] for n in order], fontsize=8)
    for tick, name in zip(ax.get_yticklabels(), order):
        tick.set_color(common.MODEL_COLORS[name])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8, color="white" if mat[i, j] > 0.6 * mat.max() else "black")
    ax.grid(False)
    ax.set_xlabel("Test day, January 2020 (UTC date)")
    ax.set_ylabel("Model (sorted by weekly MAPE)")
    ax.set_title("Per-day MAPE (%) on the test week")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MAPE (%)")
    fig.tight_layout()
    _save(fig, "04_per_day_mape.png")


def fig05_winner_intervals(r: common.ModelResult, actual: pd.Series, scores: pd.DataFrame, wstats: dict[str, float]) -> None:
    assert r.quantiles is not None
    q = r.quantiles
    color = common.MODEL_COLORS[r.name]
    fig, ax = plt.subplots(figsize=common.TS_FIGSIZE)
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), color="0.92", label="1 Jan (public holiday)")
    ax.fill_between(q[0.5].index, q[0.025].to_numpy(), q[0.975].to_numpy(), color=color, alpha=0.15, linewidth=0, label="95 % prediction interval")
    ax.fill_between(q[0.5].index, q[0.1].to_numpy(), q[0.9].to_numpy(), color=color, alpha=0.35, linewidth=0, label="80 % prediction interval")
    ax.plot(r.point.index, r.point.to_numpy(), color=color, linewidth=1.5, label=f"{common.MODEL_LABELS[r.name]} point forecast")
    ax.plot(actual.index, actual.to_numpy(), color=common.OBSERVED_COLOR, linewidth=1.2, label="Observed")
    _day_ticks(ax)
    ax.set_xlabel("Test week, 2020 (UTC date)")
    ax.set_ylabel("Load (MW)")
    caption = (
        f"{common.MODEL_LABELS[r.name]}: test MAPE {scores.loc[r.name, 'mape_test_pct']:.2f} %, "
        f"80 % interval covers {wstats['winner_coverage_80pct'] * 100:.1f} % of hours, "
        f"95 % interval covers {wstats['winner_coverage_95pct'] * 100:.1f} %"
    )
    ax.set_title(caption)
    ax.legend(loc="upper left", ncol=2, frameon=True)
    _save(fig, "05_winner_with_intervals.png")


def fig06_residuals(r: common.ModelResult, actual: pd.Series) -> None:
    resid = actual - r.point.reindex(actual.index)
    color = common.MODEL_COLORS[r.name]
    fig, axes = plt.subplots(1, 2, figsize=common.TS_FIGSIZE, gridspec_kw={"width_ratios": [3, 2]})
    by_hour = [resid[resid.index.hour == h].to_numpy() for h in range(24)]
    axes[0].boxplot(by_hour, positions=range(24), widths=0.6, patch_artist=True, boxprops={"facecolor": color, "alpha": 0.5}, medianprops={"color": "black"})
    axes[0].axhline(0, color="0.3", linewidth=0.8)
    axes[0].set_xlabel("Hour of day (UTC)")
    axes[0].set_ylabel("Residual, observed minus forecast (MW)")
    axes[0].set_title("By hour of day (7 values per box)")
    axes[0].set_xticks(range(0, 24, 3))
    axes[0].set_xticklabels([f"{h:02d}" for h in range(0, 24, 3)])
    by_day = [resid[resid.index.normalize() == d].to_numpy() for d in pd.date_range("2020-01-01", periods=7, freq="D")]
    axes[1].boxplot(by_day, positions=range(7), widths=0.6, patch_artist=True, boxprops={"facecolor": color, "alpha": 0.5}, medianprops={"color": "black"})
    axes[1].axhline(0, color="0.3", linewidth=0.8)
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(DAY_TICKS, fontsize=8)
    axes[1].set_xlabel("Test day, January 2020 (UTC date)")
    axes[1].set_ylabel("Residual, observed minus forecast (MW)")
    axes[1].set_title("By day of the test week (24 values per box)")
    fig.suptitle(f"{common.MODEL_LABELS[r.name]} residuals on the test week; positive = model under-predicted")
    fig.tight_layout()
    _save(fig, "06_residuals.png")


def fig07_feature_importance(r: common.ModelResult) -> None:
    imp = pd.Series(r.extra["feature_importance_gain"]).sort_values()
    fig, ax = plt.subplots(figsize=common.SQUARE_FIGSIZE)
    ax.barh(imp.index, imp.to_numpy() / imp.sum() * 100, color=common.MODEL_COLORS["lightgbm"])
    ax.set_xlabel("Share of total gain (%)")
    ax.set_ylabel("Feature")
    ax.set_title("LightGBM feature importance (gain), final Train+Val model")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "07_feature_importance.png")


def fig08_decomposition(r: common.ModelResult) -> None:
    e = r.extra
    unit = "MW" if e["seasonality_mode"] == "additive" else "fraction of trend"
    fig, axes = plt.subplots(3, 1, figsize=common.TS_FIGSIZE)
    color = common.MODEL_COLORS["prophet"]
    t = pd.to_datetime(e["trend_daily"]["index"])
    axes[0].plot(t, e["trend_daily"]["values"], color=color)
    axes[0].set_title("Trend (daily mean of the fitted trend, Train+Val period)")
    axes[0].set_ylabel("Load (MW)")
    axes[0].set_xlabel("Time (UTC)")
    w = pd.to_datetime(e["weekly"]["index"])
    axes[1].plot(w, e["weekly"]["values"], color=color)
    axes[1].set_xticks(w[::24])
    axes[1].set_xticklabels([d.strftime("%a") for d in w[::24]])
    axes[1].set_title("Weekly component (one week, hourly)")
    axes[1].set_ylabel(f"Effect ({unit})")
    axes[1].set_xlabel("Day of week")
    y = pd.to_datetime(e["yearly_daily"]["index"])
    axes[2].plot(y, e["yearly_daily"]["values"], color=color)
    axes[2].set_title("Yearly component (daily mean over 2019)")
    axes[2].set_ylabel(f"Effect ({unit})")
    axes[2].set_xlabel("Day of year")
    for ax in axes:
        ax.axhline(0, color="0.5", linewidth=0.5) if ax is not axes[0] else None
    fig.suptitle(f"Prophet decomposition ({e['seasonality_mode']} seasonality), final Train+Val model")
    fig.tight_layout()
    _save(fig, "08_decomposition.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_module(file_stem: str) -> Any:
    """Import code/<file_stem>.py under a private name.

    A plain `import prophet` would bind the name to our prophet.py and hide
    the real prophet package from darts, so that one script is loaded from
    its file path under the name `bakeoff_prophet` instead.
    """
    import importlib
    import importlib.util

    if file_stem != "prophet":
        # Normal import: joblib worker processes (SARIMA) must be able to
        # re-import the module by name, which a file-path load cannot offer.
        return importlib.import_module(file_stem)
    path = Path(__file__).resolve().parent / f"{file_stem}.py"
    spec = importlib.util.spec_from_file_location(f"bakeoff_{file_stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_models(names: list[str], s: pd.Series) -> None:
    for name in names:
        module = _load_module(RUNNERS[name])
        print(f"=== {name} ===", flush=True)
        result = module.run(s)
        result.save()
        print(f"{name}: test MAPE {common.mape(common.test_slice(s), result.point):.2f} %, runtime {result.runtime_seconds:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(common.MODEL_ORDER), help="comma-separated subset to (re)run")
    parser.add_argument("--from-cache", action="store_true", help="skip model runs, rebuild outputs from forecasts/")
    args = parser.parse_args()
    t0 = time.perf_counter()
    common.silence_warnings()
    common.apply_figure_style()
    common.set_seeds()
    s = common.load_series()
    if not args.from_cache:
        run_models([m.strip() for m in args.models.split(",") if m.strip()], s)

    results = {name: common.ModelResult.load(name) for name in common.MODEL_ORDER}
    actual = common.test_slice(s)
    scores = score_models(results, actual)
    per_day = pd.DataFrame({name: common.per_day_mape(actual, results[name].point).to_numpy() for name in common.MODEL_ORDER}, index=DAY_LABELS).T
    winner = str(scores.index[0])
    wstats = winner_stats(results[winner], actual)
    top2 = [str(n) for n in scores.index[:2]]

    fig01_overview(s)
    fig02_forecast_comparison(results, actual, scores)
    fig03_metric_comparison(scores)
    fig04_per_day_mape(per_day, scores)
    fig05_winner_intervals(results[winner], actual, scores, wstats)
    fig06_residuals(results[winner], actual)
    if "lightgbm" in top2:
        fig07_feature_importance(results["lightgbm"])
    if "prophet" in top2:
        fig08_decomposition(results["prophet"])

    model_runtime = float(sum(r.runtime_seconds for r in results.values()))
    total_runtime = (time.perf_counter() - t0) if not args.from_cache else model_runtime + (time.perf_counter() - t0)
    write_metrics(scores, results, wstats, winner, total_runtime)
    write_summary_tables(scores, per_day, results, wstats, winner)
    print(scores.round(2).to_string())
    print(f"winner: {winner}; " + ", ".join(f"{k}={v:.4f}" for k, v in wstats.items()))
    print(f"total runtime {total_runtime:.0f} s on {platform.platform()}")


if __name__ == "__main__":
    main()
