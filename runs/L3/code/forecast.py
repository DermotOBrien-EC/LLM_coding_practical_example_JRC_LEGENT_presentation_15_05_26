"""Orchestrator for the six-model forecasting bake-off.

This script fits every model on the same train/validation/test split, scores
each one against the held-out 168-hour test window, and writes:

  * `metrics.json` / `metrics.csv` — the headline scoreboard.
  * `figures/01..08_*.png` — publication-quality figures.

The per-model modules in this package each expose a single `run_*` function
that returns a `ModelForecast`. The orchestrator is thin on purpose: it
sequences the runs, computes derived metrics, picks the winner, and draws.
"""

from __future__ import annotations

import json
import os
import pickle
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from code.common import (
    FIG_DIR,
    METRICS_CSV,
    METRICS_JSON,
    MODEL_COLOR,
    MODEL_LABEL,
    MODEL_ORDER,
    N_TEST,
    TEST_END,
    TEST_START,
    ModelForecast,
    coverage,
    is_public_holiday_de,
    jan1_mask,
    load_load_series,
    mae,
    mape,
    pinball_loss,
    rmse,
    set_figure_style,
    split_series,
)
from code.lightgbm_features import run_lightgbm
from code.naive import run_naive
from code.nbeats import run_nbeats
from code.patchtst import run_patchtst
from code.prophet import run_prophet
from code.sarima import run_sarima

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


# ---------------------------------------------------------------------------
# Run every model, return forecasts keyed by name
# ---------------------------------------------------------------------------


def _cached_or_run(name: str, runner) -> ModelForecast:
    """Run `runner()` unless a cached `ModelForecast` for `name` is on disk.

    Cache invalidation is intentionally manual: delete `cache/<name>.pkl` to
    force a refit. Useful so a re-run after a deep-model crash doesn't redo
    SARIMA / Prophet / LightGBM. Set env var `FORECAST_FORCE=1` to bypass.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists() and not os.environ.get("FORECAST_FORCE"):
        with path.open("rb") as fh:
            cached: ModelForecast = pickle.load(fh)
        print(f"      cached — runtime {cached.runtime_seconds:.1f}s", flush=True)
        return cached
    result = runner()
    with path.open("wb") as fh:
        pickle.dump(result, fh)
    return result


def run_all_models(
    train: pd.Series, val: pd.Series, trainval: pd.Series, test: pd.Series
) -> dict[str, ModelForecast]:
    forecasts: dict[str, ModelForecast] = {}

    print("[1/6] naive seasonal baseline ...", flush=True)
    forecasts["naive"] = _cached_or_run("naive", lambda: run_naive(pd.concat([trainval, test])))

    print("[2/6] SARIMA ...", flush=True)
    forecasts["sarima"] = _cached_or_run("sarima", lambda: run_sarima(train, val, trainval))
    print(f"      SARIMA runtime: {forecasts['sarima'].runtime_seconds:.1f}s", flush=True)

    print("[3/6] Prophet ...", flush=True)
    forecasts["prophet"] = _cached_or_run("prophet", lambda: run_prophet(train, val, trainval))
    print(f"      Prophet runtime: {forecasts['prophet'].runtime_seconds:.1f}s", flush=True)

    print("[4/6] LightGBM on engineered features ...", flush=True)
    forecasts["lightgbm"] = _cached_or_run("lightgbm", lambda: run_lightgbm(train, val, trainval, test))
    print(f"      LightGBM runtime: {forecasts['lightgbm'].runtime_seconds:.1f}s", flush=True)

    print("[5/6] N-BEATS ...", flush=True)
    forecasts["nbeats"] = _cached_or_run("nbeats", lambda: run_nbeats(train, val, trainval))
    print(f"      N-BEATS runtime: {forecasts['nbeats'].runtime_seconds:.1f}s", flush=True)

    print("[6/6] TSMixer (PatchTST substitute) ...", flush=True)
    forecasts["patchtst"] = _cached_or_run("patchtst", lambda: run_patchtst(train, val, trainval))
    print(f"      TSMixer runtime: {forecasts['patchtst'].runtime_seconds:.1f}s", flush=True)

    return forecasts


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def score_models(
    forecasts: dict[str, ModelForecast], test: pd.Series
) -> list[dict]:
    actual = test.to_numpy()
    jan1 = jan1_mask(test.index)
    rest = ~jan1
    rows: list[dict] = []
    for name in MODEL_ORDER:
        f = forecasts[name]
        row = {
            "name": name,
            "mape_test_pct": mape(actual, f.point),
            "rmse_test_mw": rmse(actual, f.point),
            "mae_test_mw": mae(actual, f.point),
            "mape_jan1_pct": mape(actual[jan1], f.point[jan1]),
            "mape_jan2_to_jan7_pct": mape(actual[rest], f.point[rest]),
            "runtime_seconds": f.runtime_seconds,
            "hyperparameters": f.hyperparameters or {},
        }
        rows.append(row)
    return rows


def pick_winner(rows: list[dict]) -> str:
    return min(rows, key=lambda r: r["mape_test_pct"])["name"]


def winner_diagnostics(
    forecasts: dict[str, ModelForecast], test: pd.Series, winner: str
) -> dict:
    actual = test.to_numpy()
    f = forecasts[winner]
    out: dict = {}
    if f.lower80 is not None and f.upper80 is not None:
        out["winner_coverage_80pct"] = coverage(actual, f.lower80, f.upper80)
    else:
        out["winner_coverage_80pct"] = None
    if f.lower95 is not None and f.upper95 is not None:
        out["winner_coverage_95pct"] = coverage(actual, f.lower95, f.upper95)
    else:
        out["winner_coverage_95pct"] = None
    if f.q10 is not None:
        out["winner_pinball_loss_q10"] = pinball_loss(actual, f.q10, 0.10)
    if f.q50 is not None:
        out["winner_pinball_loss_q50"] = pinball_loss(actual, f.q50, 0.50)
    if f.q90 is not None:
        out["winner_pinball_loss_q90"] = pinball_loss(actual, f.q90, 0.90)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_overview(series: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(series.index, series.values, color="0.6", linewidth=0.5, label="Hourly load")
    ax.axvspan(TEST_START, TEST_END, color="crimson", alpha=0.35, label="Test window (168 h)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (MW)")
    ax.set_title("German hourly load — 2015-01-01 to 2020-09-30")
    ax.legend(loc="upper left")
    fig.savefig(FIG_DIR / "01_overview.png")
    plt.close(fig)


def figure_forecast_comparison(forecasts: dict[str, ModelForecast], test: pd.Series, rows: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharey=True)
    name_to_row = {r["name"]: r for r in rows}
    ymin = min(test.min(), min(f.point.min() for f in forecasts.values())) * 0.95
    ymax = max(test.max(), max(f.point.max() for f in forecasts.values())) * 1.03
    for ax, name in zip(axes.ravel(), MODEL_ORDER):
        f = forecasts[name]
        ax.plot(test.index, test.values, color="black", linewidth=1.2, label="Observed")
        ax.plot(test.index, f.point, color=MODEL_COLOR[name], linewidth=1.4, label="Forecast")
        m = name_to_row[name]["mape_test_pct"]
        ax.set_title(f"{MODEL_LABEL[name]} — MAPE {m:.2f}%")
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="x", rotation=30)
        for lab in ax.get_xticklabels():
            lab.set_horizontalalignment("right")
    for ax in axes[-1]:
        ax.set_xlabel("Time (UTC)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Load (MW)")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Test-week point forecasts vs observed load (2020-01-01 → 2020-01-07)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_forecast_comparison.png")
    plt.close(fig)


def figure_metric_comparison(rows: list[dict]) -> None:
    sorted_rows = sorted(rows, key=lambda r: r["mape_test_pct"])
    names = [r["name"] for r in sorted_rows]
    labels = [MODEL_LABEL[n] for n in names]
    mape_vals = np.array([r["mape_test_pct"] for r in sorted_rows])
    rmse_vals = np.array([r["rmse_test_mw"] for r in sorted_rows])
    mae_vals = np.array([r["mae_test_mw"] for r in sorted_rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.1]})
    x = np.arange(len(names))
    bar_colors = [MODEL_COLOR[n] for n in names]

    bars = ax1.bar(x, mape_vals, color=bar_colors)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("MAPE on test window (%)")
    ax1.set_title("MAPE — sorted ascending (lower is better)")
    for b, v in zip(bars, mape_vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + max(mape_vals) * 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    width = 0.4
    bars_rmse = ax2.bar(x - width / 2, rmse_vals, width=width, label="RMSE", color="#4c72b0")
    bars_mae = ax2.bar(x + width / 2, mae_vals, width=width, label="MAE", color="#dd8452")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Error (MW)")
    ax2.set_title("RMSE & MAE on the same ordering")
    ax2.legend(loc="upper left")
    for b, v in zip(bars_rmse, rmse_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + max(rmse_vals) * 0.01, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    for b, v in zip(bars_mae, mae_vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + max(rmse_vals) * 0.01, f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Headline error metrics across all six models", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_metric_comparison.png")
    plt.close(fig)


def figure_per_day_mape(forecasts: dict[str, ModelForecast], test: pd.Series) -> None:
    actual = test.to_numpy()
    days = pd.DatetimeIndex(test.index.date)
    unique_days = sorted(set(days))
    day_labels = [pd.Timestamp(d).strftime("%a %b %d") for d in unique_days]

    matrix = np.zeros((len(MODEL_ORDER), len(unique_days)))
    for i, name in enumerate(MODEL_ORDER):
        f = forecasts[name]
        for j, d in enumerate(unique_days):
            mask = days == pd.Timestamp(d)
            matrix[i, j] = mape(actual[mask], f.point[mask])

    fig, (ax_heat, ax_bars) = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1.05, 1]})

    im = ax_heat.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax_heat.set_xticks(np.arange(len(unique_days)))
    ax_heat.set_xticklabels(day_labels, rotation=30, ha="right")
    ax_heat.set_yticks(np.arange(len(MODEL_ORDER)))
    ax_heat.set_yticklabels([MODEL_LABEL[n] for n in MODEL_ORDER])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            txt_color = "white" if matrix[i, j] > matrix.max() * 0.55 else "black"
            ax_heat.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color=txt_color, fontsize=9)
    ax_heat.set_title("Per-day MAPE (%) — Wed Jan 1 is a German public holiday")
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.85)
    cbar.set_label("MAPE (%)")

    # Right panel: grouped bars per model showing the Jan 1 vs. rest contrast.
    jan1 = np.array([(d == pd.Timestamp("2020-01-01").date()) for d in days])
    rest = ~jan1
    actual_w = test.to_numpy()
    jan1_mape = []
    rest_mape = []
    for name in MODEL_ORDER:
        f = forecasts[name]
        jan1_mape.append(mape(actual_w[jan1], f.point[jan1]))
        rest_mape.append(mape(actual_w[rest], f.point[rest]))

    x = np.arange(len(MODEL_ORDER))
    w = 0.4
    ax_bars.bar(x - w / 2, jan1_mape, width=w, label="Jan 1 (holiday)", color="crimson")
    ax_bars.bar(x + w / 2, rest_mape, width=w, label="Jan 2 – Jan 7", color="steelblue")
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels([MODEL_LABEL[n] for n in MODEL_ORDER], rotation=20, ha="right")
    ax_bars.set_ylabel("MAPE (%)")
    ax_bars.set_title("Holiday vs working-week MAPE")
    ax_bars.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_per_day_mape.png")
    plt.close(fig)


def figure_winner_with_intervals(
    forecasts: dict[str, ModelForecast], test: pd.Series, winner: str, diag: dict
) -> None:
    f = forecasts[winner]
    fig, ax = plt.subplots(figsize=(11, 6))
    if f.lower95 is not None and f.upper95 is not None:
        ax.fill_between(test.index, f.lower95, f.upper95, color=MODEL_COLOR[winner], alpha=0.15, label="95% interval")
    if f.lower80 is not None and f.upper80 is not None:
        ax.fill_between(test.index, f.lower80, f.upper80, color=MODEL_COLOR[winner], alpha=0.3, label="80% interval")
    ax.plot(test.index, test.values, color="black", linewidth=1.4, label="Observed")
    ax.plot(test.index, f.point, color=MODEL_COLOR[winner], linewidth=1.6, label=f"{MODEL_LABEL[winner]} forecast")
    test_mape = mape(test.to_numpy(), f.point)
    c80 = diag.get("winner_coverage_80pct")
    c95 = diag.get("winner_coverage_95pct")
    title = (
        f"Winner: {MODEL_LABEL[winner]} — test MAPE {test_mape:.2f}%; "
        f"80% coverage {None if c80 is None else f'{c80*100:.1f}%'}; "
        f"95% coverage {None if c95 is None else f'{c95*100:.1f}%'}"
    )
    ax.set_title(title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_winner_with_intervals.png")
    plt.close(fig)


def figure_residuals(forecasts: dict[str, ModelForecast], test: pd.Series, winner: str) -> None:
    f = forecasts[winner]
    actual = test.to_numpy()
    resid = actual - f.point
    hours = test.index.hour
    days = pd.DatetimeIndex(test.index.date)
    day_labels = pd.Series([pd.Timestamp(d).strftime("%a %b %d") for d in days])

    fig, (ax_h, ax_d) = plt.subplots(1, 2, figsize=(13, 6))

    hour_groups = [resid[hours == h] for h in range(24)]
    bp = ax_h.boxplot(hour_groups, positions=range(24), widths=0.7, patch_artist=True, showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(MODEL_COLOR[winner])
        patch.set_alpha(0.6)
    ax_h.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_h.set_xticks(range(0, 24, 3))
    ax_h.set_xticklabels(range(0, 24, 3))
    ax_h.set_xlabel("Hour of day (UTC)")
    ax_h.set_ylabel("Residual (observed − forecast, MW)")
    ax_h.set_title(f"{MODEL_LABEL[winner]} residuals by hour of day")

    unique_days = list(dict.fromkeys(day_labels))
    day_groups = [resid[day_labels.values == d] for d in unique_days]
    bp2 = ax_d.boxplot(day_groups, positions=range(len(unique_days)), widths=0.5, patch_artist=True)
    for patch in bp2["boxes"]:
        patch.set_facecolor(MODEL_COLOR[winner])
        patch.set_alpha(0.6)
    ax_d.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_d.set_xticks(range(len(unique_days)))
    ax_d.set_xticklabels(unique_days, rotation=30, ha="right")
    ax_d.set_xlabel("Day in test week")
    ax_d.set_ylabel("Residual (MW)")
    ax_d.set_title(f"{MODEL_LABEL[winner]} residuals by day")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_residuals.png")
    plt.close(fig)


def figure_feature_importance(forecasts: dict[str, ModelForecast]) -> None:
    imp = forecasts["lightgbm"].hyperparameters.get("feature_importance_gain", {})
    if not imp:
        return
    pairs = sorted(imp.items(), key=lambda kv: kv[1])
    names = [k for k, _ in pairs]
    vals = [v for _, v in pairs]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(names, vals, color=MODEL_COLOR["lightgbm"])
    ax.set_xlabel("Gain importance (LightGBM)")
    ax.set_title("LightGBM feature importance (gain)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_feature_importance.png")
    plt.close(fig)


def figure_prophet_decomposition(train: pd.Series, val: pd.Series, trainval: pd.Series) -> None:
    """Decomposition plot for Prophet, generated only if Prophet is in top 2."""
    import logging
    for name in ("cmdstanpy", "prophet", "prophet.plot"):
        logging.getLogger(name).setLevel(logging.ERROR)
    from darts import TimeSeries
    from darts.models import Prophet as DartsProphet

    ts = TimeSeries.from_series(trainval)
    m = DartsProphet(
        country_holidays="DE",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    m.fit(ts)
    inner = m.model  # the underlying Prophet model
    # Forecast across the historical range for the decomposition.
    future = inner.make_future_dataframe(periods=N_TEST, freq="h")
    forecast = inner.predict(future)
    fig = inner.plot_components(forecast)
    fig.set_size_inches(11, 9)
    fig.suptitle("Prophet decomposition: trend, weekly, daily, yearly, holidays", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_decomposition.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_metrics(rows: list[dict], diag: dict, winner: str, total_runtime: float) -> None:
    payload = {
        "test_start": "2020-01-01",
        "test_end": "2020-01-07",
        "n_test_observations": N_TEST,
        "models": rows,
        "winner": winner,
        "winner_coverage_80pct": diag.get("winner_coverage_80pct"),
        "winner_coverage_95pct": diag.get("winner_coverage_95pct"),
        "winner_pinball_loss_q10": diag.get("winner_pinball_loss_q10"),
        "winner_pinball_loss_q50": diag.get("winner_pinball_loss_q50"),
        "winner_pinball_loss_q90": diag.get("winner_pinball_loss_q90"),
        "total_runtime_seconds": total_runtime,
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2, default=float))

    # CSV: one row per model. Include the winner-level metrics in a final
    # 'overall' row so the file stays self-describing.
    csv_rows = []
    for r in rows:
        csv_rows.append(
            {
                "name": r["name"],
                "mape_test_pct": r["mape_test_pct"],
                "rmse_test_mw": r["rmse_test_mw"],
                "mae_test_mw": r["mae_test_mw"],
                "mape_jan1_pct": r["mape_jan1_pct"],
                "mape_jan2_to_jan7_pct": r["mape_jan2_to_jan7_pct"],
                "runtime_seconds": r["runtime_seconds"],
                "is_winner": r["name"] == winner,
            }
        )
    df = pd.DataFrame(csv_rows)
    df.to_csv(METRICS_CSV, index=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    set_figure_style()
    FIG_DIR.mkdir(exist_ok=True)

    overall_start = time.perf_counter()
    series = load_load_series()
    train, val, trainval, test = split_series(series)

    forecasts = run_all_models(train, val, trainval, test)
    rows = score_models(forecasts, test)
    winner = pick_winner(rows)
    diag = winner_diagnostics(forecasts, test, winner)

    # Figures.
    print("Writing figures ...", flush=True)
    figure_overview(series)
    figure_forecast_comparison(forecasts, test, rows)
    figure_metric_comparison(rows)
    figure_per_day_mape(forecasts, test)
    figure_winner_with_intervals(forecasts, test, winner, diag)
    figure_residuals(forecasts, test, winner)

    # Conditional figures.
    rank = sorted(rows, key=lambda r: r["mape_test_pct"])
    top2 = [r["name"] for r in rank[:2]]
    if "lightgbm" in top2:
        figure_feature_importance(forecasts)
    if "prophet" in top2:
        try:
            figure_prophet_decomposition(train, val, trainval)
        except Exception as exc:  # noqa: BLE001
            print(f"Decomposition figure failed: {exc}", flush=True)

    total_runtime = time.perf_counter() - overall_start
    write_metrics(rows, diag, winner, total_runtime)

    # Save the raw point and quantile forecasts so transcript.md / replication
    # don't need to rerun every model.
    fcst_df = pd.DataFrame({"timestamp": test.index, "actual": test.values})
    for name, f in forecasts.items():
        fcst_df[f"{name}_point"] = f.point
        if f.q10 is not None:
            fcst_df[f"{name}_q10"] = f.q10
        if f.q50 is not None:
            fcst_df[f"{name}_q50"] = f.q50
        if f.q90 is not None:
            fcst_df[f"{name}_q90"] = f.q90
    fcst_df.to_csv(METRICS_CSV.parent / "forecasts_test_window.csv", index=False)

    print()
    print(f"Winner: {winner}  (test MAPE {min(r['mape_test_pct'] for r in rows):.3f}%)")
    print(f"Total runtime: {total_runtime:.1f}s")


if __name__ == "__main__":
    main()
