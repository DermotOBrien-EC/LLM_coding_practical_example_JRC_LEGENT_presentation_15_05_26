"""Orchestrator: run all six models, score them, and write every output.

This is the thin conductor. It does no modelling of its own. It:

1. Loads the data once and hands the same slices to every model.
2. Runs the six model modules and collects their forecasts.
3. Scores each on the held-out test week (MAPE, RMSE, MAE, plus the
   holiday-vs-working-day split).
4. Picks the winner (lowest test MAPE) and works out its interval coverage
   and pinball loss.
5. Writes metrics.json, metrics.csv, the eight figures, and transcript.md.

Run it with the project venv:
    ../../.venv/bin/python code/forecast.py
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import platform
import subprocess
import time
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

import common as C

# Load the real Prophet library into the import cache before any model module
# pulls in darts (which imports `prophet` internally); otherwise our own
# prophet.py would shadow it.
C.ensure_real_prophet()

import matplotlib.pyplot as plt  # noqa: E402

CODE_DIR = Path(__file__).resolve().parent

# Which file provides each model. Loaded by path so that our prophet.py never
# gets imported under the name `prophet`.
MODEL_FILES: dict[str, str] = {
    "naive": "naive.py",
    "sarima": "sarima.py",
    "prophet": "prophet.py",
    "lightgbm": "lightgbm_features.py",
    "nbeats": "nbeats.py",
    "patchtst": "patchtst.py",
}


def _load_module(alias: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(alias, CODE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _jan1_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True for the 24 UTC hours of 2020-01-01 (the public holiday)."""
    target = dt.date(2020, 1, 1)
    return np.array([ts.date() == target for ts in index], dtype=bool)


def _per_day_mape(actual: np.ndarray, forecast: np.ndarray, index: pd.DatetimeIndex) -> dict[str, float]:
    """MAPE for each of the seven UTC calendar days in the test week."""
    out: dict[str, float] = {}
    for day in sorted({ts.date() for ts in index}):
        mask = np.array([ts.date() == day for ts in index], dtype=bool)
        out[day.isoformat()] = C.mape(actual[mask], forecast[mask])
    return out


def score_model(res: C.ForecastResult, actual: np.ndarray, index: pd.DatetimeIndex) -> dict:
    jan1 = _jan1_mask(index)
    workday = ~jan1
    return {
        "name": res.name,
        "mape_test_pct": C.mape(actual, res.point),
        "rmse_test_mw": C.rmse(actual, res.point),
        "mae_test_mw": C.mae(actual, res.point),
        "mape_jan1_pct": C.mape(actual[jan1], res.point[jan1]),
        "mape_jan2_to_jan7_pct": C.mape(actual[workday], res.point[workday]),
        "runtime_seconds": res.runtime_seconds,
        "val_mape_pct": res.val_mape,
        "hyperparameters": res.hyperparameters,
        "per_day_mape": _per_day_mape(actual, res.point, index),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _global_yrange(actual: np.ndarray, results: dict[str, C.ForecastResult]) -> tuple[float, float]:
    lo = actual.min()
    hi = actual.max()
    for r in results.values():
        lo = min(lo, r.point.min())
        hi = max(hi, r.point.max())
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def fig_overview(full: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=C.TS_FIGSIZE)
    ax.plot(full.index, full.to_numpy() / 1000.0, color="0.7", lw=0.6, label="Hourly load")
    test = C.slice_inclusive(full, C.TEST_START, C.TEST_END)
    ax.plot(test.index, test.to_numpy() / 1000.0, color="#d62728", lw=1.5,
            label="Test week (2020-01-01 to 2020-01-07)")
    ax.axvspan(C.TEST_START, C.TEST_END, color="#d62728", alpha=0.15)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (GW)")
    ax.set_title("German hourly electricity load, 2015-2020, with the held-out test week")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "01_overview.png")
    plt.close(fig)


def fig_forecast_comparison(actual: np.ndarray, index: pd.DatetimeIndex,
                            results: dict[str, C.ForecastResult], scores: dict[str, dict]) -> None:
    lo, hi = _global_yrange(actual, results)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, C.MODEL_ORDER):
        r = results[name]
        ax.plot(index, actual / 1000.0, color="black", lw=1.2, label="Observed")
        ax.plot(index, r.point / 1000.0, color=C.MODEL_COLORS[name], lw=1.5, label="Forecast")
        ax.set_title(f"{C.MODEL_LABELS[name]}  (MAPE {scores[name]['mape_test_pct']:.2f}%)")
        ax.set_ylim(lo / 1000.0, hi / 1000.0)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Load (GW)")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Point forecasts vs observed load, test week (same y-axis on every panel)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "02_forecast_comparison.png")
    plt.close(fig)


def fig_metric_comparison(scores: dict[str, dict], order: list[str]) -> None:
    metrics = [("mape_test_pct", "MAPE (%)"), ("rmse_test_mw", "RMSE (MW)"), ("mae_test_mw", "MAE (MW)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    labels = [C.MODEL_LABELS[n] for n in order]
    colors = [C.MODEL_COLORS[n] for n in order]
    for ax, (key, title) in zip(axes, metrics):
        vals = [scores[n][key] for n in order]
        bars = ax.bar(range(len(order)), vals, color=colors)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(title)
        top = max(vals)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01 * top, f"{v:,.1f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, top * 1.15)
    fig.suptitle("Test-window error by model (sorted by MAPE, lowest first)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "03_metric_comparison.png")
    plt.close(fig)


def fig_per_day_mape(scores: dict[str, dict], order: list[str]) -> None:
    days = sorted(scores[order[0]]["per_day_mape"].keys())
    matrix = np.array([[scores[n]["per_day_mape"][d] for d in days] for n in order])
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(days)))
    day_labels = [f"{d}\n({dt.date.fromisoformat(d).strftime('%a')})" for d in days]
    ax.set_xticklabels(day_labels, fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([C.MODEL_LABELS[n] for n in order])
    for i in range(len(order)):
        for j in range(len(days)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    color="black" if v < matrix.max() * 0.6 else "white", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MAPE (%)")
    ax.set_title("Per-day MAPE by model (Jan 1 is the New Year public holiday)")
    ax.set_xlabel("Test day (UTC)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "04_per_day_mape.png")
    plt.close(fig)


def fig_winner_intervals(actual: np.ndarray, index: pd.DatetimeIndex,
                         winner: C.ForecastResult, score: dict,
                         cov80: float, cov95: float) -> None:
    color = C.MODEL_COLORS[winner.name]
    q = winner.quantiles
    fig, ax = plt.subplots(figsize=C.TS_FIGSIZE)
    ax.fill_between(index, q[0.025] / 1000.0, q[0.975] / 1000.0, color=color, alpha=0.15,
                    label="95% interval")
    ax.fill_between(index, q[0.1] / 1000.0, q[0.9] / 1000.0, color=color, alpha=0.30,
                    label="80% interval")
    ax.plot(index, actual / 1000.0, color="black", lw=1.4, label="Observed")
    ax.plot(index, winner.point / 1000.0, color=color, lw=1.6, label="Point forecast")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (GW)")
    ax.tick_params(axis="x", rotation=30)
    ax.set_title(
        f"Winner: {C.MODEL_LABELS[winner.name]}  |  test MAPE {score['mape_test_pct']:.2f}%  |  "
        f"80% coverage {cov80 * 100:.0f}%  |  95% coverage {cov95 * 100:.0f}%"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "05_winner_with_intervals.png")
    plt.close(fig)


def fig_residuals(actual: np.ndarray, index: pd.DatetimeIndex, winner: C.ForecastResult) -> None:
    resid = (actual - winner.point) / 1000.0  # GW
    hours = np.array([ts.hour for ts in index])
    days = np.array([ts.date().isoformat() for ts in index])
    color = C.MODEL_COLORS[winner.name]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    hour_groups = [resid[hours == h] for h in range(24)]
    bp1 = ax1.boxplot(hour_groups, positions=range(24), widths=0.6, patch_artist=True)
    for box in bp1["boxes"]:
        box.set(facecolor=color, alpha=0.5)
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_xlabel("Hour of day (UTC)")
    ax1.set_ylabel("Residual: observed - forecast (GW)")
    ax1.set_title("Residuals by hour of day")
    ax1.set_xticks(range(0, 24, 3))
    ax1.set_xticklabels(range(0, 24, 3))

    uniq_days = sorted(set(days))
    day_groups = [resid[days == d] for d in uniq_days]
    bp2 = ax2.boxplot(day_groups, positions=range(len(uniq_days)), widths=0.6, patch_artist=True)
    for box in bp2["boxes"]:
        box.set(facecolor=color, alpha=0.5)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("Test day (UTC)")
    ax2.set_ylabel("Residual: observed - forecast (GW)")
    ax2.set_title("Residuals by day of test week")
    ax2.set_xticks(range(len(uniq_days)))
    ax2.set_xticklabels([f"{d}\n({dt.date.fromisoformat(d).strftime('%a')})" for d in uniq_days],
                        fontsize=8)

    fig.suptitle(f"Winner residuals: {C.MODEL_LABELS[winner.name]}")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "06_residuals.png")
    plt.close(fig)


def fig_feature_importance(lgbm: C.ForecastResult) -> None:
    imp = lgbm.extra.get("feature_importance_gain", {})
    if not imp:
        return
    items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(names)), vals, color=C.MODEL_COLORS["lightgbm"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (total gain)")
    ax.set_title("LightGBM feature importance (gain)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "07_feature_importance.png")
    plt.close(fig)


def fig_prophet_decomposition(bundle: C.DataBundle, params: dict) -> None:
    """Trend / weekly / yearly components of a Prophet refit on Train+Val."""
    C.ensure_real_prophet()
    from prophet import Prophet as RawProphet

    trainval = C.slice_inclusive(bundle.full, C.TRAIN_START, C.VAL_END)
    dfp = pd.DataFrame({"ds": trainval.index.tz_localize(None), "y": trainval.to_numpy()})
    m = RawProphet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode=params.get("seasonality_mode", "additive"),
        changepoint_prior_scale=params.get("changepoint_prior_scale", 0.05),
    )
    m.add_country_holidays(country_name="DE")
    m.fit(dfp)
    forecast = m.predict(dfp)
    fig = m.plot_components(forecast)
    fig.set_size_inches(9, 9)
    fig.suptitle("Prophet components (trend, holidays, weekly, yearly, daily)", y=1.02)
    fig.savefig(C.FIGURES_DIR / "08_decomposition.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# transcript.md
# --------------------------------------------------------------------------


def _results_table(scores: dict[str, dict], order: list[str]) -> str:
    header = (
        "| Model | MAPE (%) | RMSE (MW) | MAE (MW) | MAPE Jan 1 (%) | "
        "MAPE Jan 2-7 (%) | Runtime (s) |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for n in order:
        s = scores[n]
        rows.append(
            f"| {C.MODEL_LABELS[n]} | {s['mape_test_pct']:.2f} | {s['rmse_test_mw']:,.0f} | "
            f"{s['mae_test_mw']:,.0f} | {s['mape_jan1_pct']:.2f} | "
            f"{s['mape_jan2_to_jan7_pct']:.2f} | {s['runtime_seconds']:.1f} |"
        )
    return header + "\n".join(rows)


def write_transcript(scores: dict[str, dict], order: list[str], winner: str,
                     prob: dict, total_runtime: float, uname: str,
                     substitution_note: str) -> None:
    s = scores
    w = C.MODEL_LABELS[winner]
    naive_mape = s["naive"]["mape_test_pct"]
    win_mape = s[winner]["mape_test_pct"]
    improvement = (naive_mape - win_mape) / naive_mape * 100.0

    table = _results_table(s, order)

    text = f"""# German hourly load: a six-model forecasting bake-off

## 1. Data

The single input is `opsd_de_load.csv` from Open Power System Data, itself
derived from the ENTSO-E Transparency Platform. It holds one column of interest,
the actual German national load in megawatts, sampled every hour from
2015-01-01 00:00 UTC to 2020-09-30 23:00 UTC. That is exactly 50,400 hourly
rows. On loading we check the row count, confirm there are no missing values,
and confirm the hourly timestamps form an unbroken sequence with no gaps or
duplicates. All three checks pass, so no imputation or gap-filling is needed and
none is done: the study runs on the raw series exactly as delivered.

## 2. Why these six models

The six models are a deliberate complexity gradient, from "almost no model" to
"deep neural network", so we can see how much each extra layer of machinery
actually buys.

- **Seasonal-naive** repeats the load from the same hour one week earlier. It
  has no parameters and exists as the bar every other model must clear.
- **SARIMA** is the classical statistical workhorse: autoregression, moving
  averages, and a daily seasonal term after differencing.
- **Prophet** fits a smooth trend plus daily, weekly and yearly seasonal
  shapes, and is explicitly told about German public holidays.
- **LightGBM** abandons the time-series view and instead predicts each hour
  from engineered features: calendar fields, holiday flag, lags of the load
  (one day, one week, one year), and rolling means and spreads.
- **N-BEATS** is a deep fully-connected network that learns the forecast purely
  from the recent shape of the series.
- **PatchTST** (served here by TSMixer, see below) is a modern deep architecture
  aimed at longer horizons.

The gradient tests a real question: on a purely univariate problem, does
sophistication pay, or does a good feature set plus gradient boosting win?

Substitution note: {substitution_note}

## 3. Validation strategy

The data is cut into three non-overlapping blocks in time. Train is
2015-01-01 to 2019-09-30 (about 4.75 years, 41,616 hours). Validation is
2019-10-01 to 2019-12-31 (2,208 hours). Test is the 168 hours of
2020-01-01 to 2020-01-07, held out and scored exactly once.

Every model with settings to choose is fit on Train only, then used to forecast
the first week of the validation window (2019-10-01 to 2019-10-07); the setting
with the lowest validation MAPE is kept. This mirrors the real test task, which
is a single 168-hour-ahead forecast. Once a model's settings are locked, the
model is refit on Train and Validation combined (all data up to
2019-12-31 23:00) and only then asked to forecast the test week. Refitting on
all pre-test data is standard practice: there is no reason to withhold the
validation months from the final model once they have done their tuning job.
The test week is never seen during any fitting or selection step.

## 4. Results

{table}

Winner: **{w}**, with a test MAPE of {win_mape:.2f}%, which is {improvement:.0f}%
lower than the seasonal-naive baseline ({naive_mape:.2f}%).

For the winner, the prediction intervals were checked against the actual load:

- 80% interval: nominal 80%, actual coverage {prob['coverage_80'] * 100:.1f}%
- 95% interval: nominal 95%, actual coverage {prob['coverage_95'] * 100:.1f}%
- Pinball loss (MW): q0.1 = {prob['pinball_q10']:,.1f}, q0.5 = {prob['pinball_q50']:,.1f}, q0.9 = {prob['pinball_q90']:,.1f}

## 5. Discussion

{_discussion(scores, order, winner)}

## 6. Recommendation

For a JRC production short-term load forecasting pipeline, the model to put in
first is **{w}**. It gave the lowest error here, it is cheap to fit and fully
deterministic (a rerun gives the identical answer; the runtime in the table is
mostly the hyperparameter sweep and the hour-by-hour interval forecast, not a
single fit), and its feature-based design is the natural place to later add the
exogenous signals this study deliberately excluded. The first improvement to make would be
to add weather, above all temperature, since heating and cooling demand is the
single largest driver of load that a purely calendar-and-lag model cannot see.
After that: a proper holiday-and-bridge-day calendar (the day after New Year, the
days between Christmas and New Year), and rolling-origin backtesting across many
weeks and seasons rather than a single winter test week, so the error estimate
is not tied to one unusual holiday period. Prediction intervals would be
recalibrated on a validation set before anyone relied on them operationally.

## 7. Reproducibility

- Random seed: {C.SEED} (Python, NumPy, and PyTorch).
- The two deep models (N-BEATS, TSMixer) train on the Apple MPS GPU when
  available; MPS is not bit-for-bit deterministic across runs, so their numbers
  can move by a few tenths of a MAPE point between runs. The tree, statistical
  and baseline models are fully deterministic, and the winner is one of those,
  so the headline result is stable.
- Total model-fitting wall-clock this run: {total_runtime:.0f} seconds.
- Hardware / OS: {uname}
- To reproduce: `../../.venv/bin/python code/forecast.py` from this directory.
"""
    (C.ROOT / "transcript.md").write_text(text)


def _discussion(scores: dict[str, dict], order: list[str], winner: str) -> str:
    s = scores
    win = C.MODEL_LABELS[winner]
    worst_jan1 = max(order, key=lambda n: s[n]["mape_jan1_pct"])
    win_j1 = s[winner]["mape_jan1_pct"]

    naive_pd = s["naive"]["per_day_mape"]
    days = sorted(naive_pd)
    naive_j1 = naive_pd[days[0]]
    naive_worst_day = max(naive_pd, key=naive_pd.get)
    naive_worst = naive_pd[naive_worst_day]
    worst_dow = dt.date.fromisoformat(naive_worst_day).strftime("%a %-d %b")

    return f"""The ranking is close to what forecasting theory would predict for a
univariate problem with a strong, learnable calendar structure. {win} won because
its features hand the model exactly the things that drive load: what hour and
weekday it is, whether it is a holiday, and where the level has been over the last
day, week and year. Gradient boosting then only has to learn a fairly smooth map
from those features to demand.

The New Year holiday is a more subtle test than it first looks, and the per-day
breakdown (figure 4) overturns the obvious expectation that every model should
stumble on Jan 1. For the models that lean on the one-week lag it is the reverse:
the seasonal-naive baseline predicts the Jan 1 holiday well ({naive_j1:.1f}% MAPE)
and {win} is best of all on it ({win_j1:.1f}%). The reason is a calendar
coincidence. One week before Jan 1 2020 is Dec 25 2019, Christmas Day, itself a
depressed-demand holiday, so the one-week lag quietly hands these models a
holiday-shaped template for the New Year. Their real weakness is the mirror image:
the worst single day in the whole study is the post-holiday return to normal.
Seasonal-naive's largest error is {worst_dow} ({naive_worst:.1f}%), because the day
one week earlier (New Year's Eve, still inside the Christmas lull) is a poor
template for an ordinary working Tuesday. So the lag is a gift on the holiday and a
trap on the recovery. The one model clearly caught out on Jan 1 itself is Prophet
({s[worst_jan1]['mape_jan1_pct']:.1f}%): its additive holiday term over-corrects and
pushes the Jan 1 forecast well below the actual load, the deep dip visible in its
panel of figure 2.

Where each model fails is instructive. Seasonal-naive is exact whenever this week
resembles last week and badly wrong when it does not, so its error sits on the
recovery days, not the holiday. SARIMA, restricted to daily seasonality, never
really captures the weekly rhythm and lands essentially on top of the naive
baseline: the classical machinery buys almost nothing here without a weekly
seasonal term. Prophet has the right seasonal vocabulary and a holiday term, so it
is the runner-up, but it fits a smooth average shape, misses the sharper
hour-to-hour moves, and over-shoots the New Year dip. The two deep models are a
surprise only if one expected depth to win automatically: with a single week of
context, no calendar inputs, and under five years of hourly data, they have to
relearn structure the feature-based model is simply handed, and they finish
mid-pack, noticeably noisier hour to hour (figure 2). This matches the standard
finding that on tabular, feature-rich forecasting problems, gradient-boosted trees
are very hard to beat. The main caveat is that this is a single 168-hour window
over an unusual holiday week, so the exact ranking should not be over-read; a fair
operational verdict would repeat the exercise over many rolling windows across the
seasons."""


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    C.apply_figure_style()
    C.FIGURES_DIR.mkdir(exist_ok=True)
    bundle = C.build_bundle()
    actual = bundle.test.to_numpy()
    index = bundle.test.index

    modules = {name: _load_module(f"m_{name}", fn) for name, fn in MODEL_FILES.items()}

    results: dict[str, C.ForecastResult] = {}
    t0 = time.time()
    for name in C.MODEL_ORDER:
        print(f"[run] {name} ...", flush=True)
        results[name] = modules[name].run(bundle)
        print(f"[run] {name} done: test MAPE {C.mape(actual, results[name].point):.2f}%",
              flush=True)
    total_runtime = time.time() - t0

    scores = {name: score_model(results[name], actual, index) for name in C.MODEL_ORDER}
    order = sorted(C.MODEL_ORDER, key=lambda n: scores[n]["mape_test_pct"])
    winner = order[0]
    win_res = results[winner]

    # --- Winner probabilistic metrics ---------------------------------------
    q = win_res.quantiles
    prob = {
        "coverage_80": C.coverage(actual, q[0.1], q[0.9]),
        "coverage_95": C.coverage(actual, q[0.025], q[0.975]),
        "pinball_q10": C.pinball_loss(actual, q[0.1], 0.1),
        "pinball_q50": C.pinball_loss(actual, q[0.5], 0.5),
        "pinball_q90": C.pinball_loss(actual, q[0.9], 0.9),
    }

    # --- metrics.json / metrics.csv -----------------------------------------
    models_json = []
    for n in order:
        s = scores[n]
        models_json.append({
            "name": n,
            "mape_test_pct": round(s["mape_test_pct"], 4),
            "rmse_test_mw": round(s["rmse_test_mw"], 4),
            "mae_test_mw": round(s["mae_test_mw"], 4),
            "mape_jan1_pct": round(s["mape_jan1_pct"], 4),
            "mape_jan2_to_jan7_pct": round(s["mape_jan2_to_jan7_pct"], 4),
            "runtime_seconds": round(s["runtime_seconds"], 2),
            "hyperparameters": s["hyperparameters"],
        })

    metrics = {
        "test_start": "2020-01-01",
        "test_end": "2020-01-07",
        "n_test_observations": int(len(actual)),
        "models": models_json,
        "winner": winner,
        "winner_coverage_80pct": round(prob["coverage_80"], 4),
        "winner_coverage_95pct": round(prob["coverage_95"], 4),
        "winner_pinball_loss_q10": round(prob["pinball_q10"], 4),
        "winner_pinball_loss_q50": round(prob["pinball_q50"], 4),
        "winner_pinball_loss_q90": round(prob["pinball_q90"], 4),
        "total_runtime_seconds": round(total_runtime, 2),
    }
    (C.ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    csv_rows = []
    for n in order:
        s = scores[n]
        csv_rows.append({
            "name": n,
            "mape_test_pct": s["mape_test_pct"],
            "rmse_test_mw": s["rmse_test_mw"],
            "mae_test_mw": s["mae_test_mw"],
            "mape_jan1_pct": s["mape_jan1_pct"],
            "mape_jan2_to_jan7_pct": s["mape_jan2_to_jan7_pct"],
            "runtime_seconds": s["runtime_seconds"],
            "val_mape_pct": s["val_mape_pct"],
            "is_winner": n == winner,
            "hyperparameters": json.dumps(s["hyperparameters"]),
        })
    pd.DataFrame(csv_rows).to_csv(C.ROOT / "metrics.csv", index=False)

    # --- Figures ------------------------------------------------------------
    print("[fig] generating figures ...", flush=True)
    fig_overview(bundle.full)
    fig_forecast_comparison(actual, index, results, scores)
    fig_metric_comparison(scores, order)
    fig_per_day_mape(scores, order)
    fig_winner_intervals(actual, index, win_res, scores[winner],
                         prob["coverage_80"], prob["coverage_95"])
    fig_residuals(actual, index, win_res)

    top2 = order[:2]
    if "lightgbm" in top2:
        fig_feature_importance(results["lightgbm"])
    if "prophet" in top2:
        fig_prophet_decomposition(bundle, results["prophet"].hyperparameters)

    # --- transcript ---------------------------------------------------------
    uname = platform.platform()
    try:
        uname = subprocess.check_output(["uname", "-a"]).decode().strip()
    except Exception:
        pass
    substitution = results["patchtst"].hyperparameters.get(
        "substitution_reason", "none")
    substitution_note = (
        f"PatchTST is not available in the installed darts (0.41.0), so the "
        f"PatchTST slot is filled by darts TSMixerModel, the first substitute the "
        f"prompt allows. Reason recorded: {substitution}."
    )
    write_transcript(scores, order, winner, prob, total_runtime, uname, substitution_note)

    print(f"[done] winner: {winner} (MAPE {scores[winner]['mape_test_pct']:.2f}%)", flush=True)
    print(f"[done] total model-fitting runtime: {total_runtime:.0f}s", flush=True)


if __name__ == "__main__":
    main()
