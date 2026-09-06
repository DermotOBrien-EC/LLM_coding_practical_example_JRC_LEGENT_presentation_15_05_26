from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    TEST_END,
    TEST_START,
    MetricsRow,
    ModelOutput,
    coverage,
    evaluate_output,
    final_training_series,
    json_safe,
    load_data,
    load_series,
    per_day_mape,
    pinball_loss,
    residuals,
    test_series,
    train_series,
    validation_series,
)

ROOT = Path(".")
FIGURES = ROOT / "figures"
CODE = ROOT / "code"
ARTIFACTS = CODE / ".artifacts"
PYTHON = Path("../../.venv/bin/python")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: object) -> object:
        safe = json_safe(obj)
        if safe is obj:
            return super().default(obj)
        return safe


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": "#0b0b0b",
            "axes.titlecolor": "#0b0b0b",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "grid.color": "#e1e0d9",
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "legend.frameon": False,
        }
    )


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_model_subprocess(name: str) -> None:
    command = [str(PYTHON), str(CODE / "run_one.py"), name]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def load_model_output(name: str) -> ModelOutput:
    metadata = json.loads((ARTIFACTS / f"{name}_metadata.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(ARTIFACTS / f"{name}_forecast.csv")
    index = pd.to_datetime(frame["timestamp"])

    def optional_series(column: str) -> pd.Series | None:
        values = frame[column].to_numpy(dtype=float)
        if np.isnan(values).all():
            return None
        return pd.Series(values, index=index)

    feature_importance = None
    importance_path = ARTIFACTS / "lightgbm_feature_importance.csv"
    if name == "lightgbm" and importance_path.exists():
        importance_frame = pd.read_csv(importance_path)
        feature_importance = pd.Series(importance_frame["gain"].to_numpy(dtype=float), index=importance_frame["feature"], name="LightGBM gain").sort_values(ascending=False)

    return ModelOutput(
        name=str(metadata["name"]),
        forecast=pd.Series(frame["forecast"].to_numpy(dtype=float), index=index, name="forecast"),
        runtime_seconds=float(metadata["runtime_seconds"]),
        hyperparameters=dict(metadata["hyperparameters"]),
        lower_80=optional_series("lower_80"),
        upper_80=optional_series("upper_80"),
        lower_95=optional_series("lower_95"),
        upper_95=optional_series("upper_95"),
        q10=optional_series("q10"),
        q50=optional_series("q50"),
        q90=optional_series("q90"),
        feature_importance=feature_importance,
        notes=list(metadata.get("notes", [])),
    )


def plot_overview(series: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(series.index, series.values, color="#c3c2b7", linewidth=0.5, label="Observed load")
    test = series.loc[TEST_START:TEST_END]
    ax.plot(test.index, test.values, color="#d03b3b", linewidth=1.8, label="Held-out test week")
    ax.axvspan(TEST_START, TEST_END, color="#d03b3b", alpha=0.12)
    ax.set_title("German hourly electricity load with held-out test window")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="upper right")
    savefig(fig, FIGURES / "01_overview.png")


def plot_forecast_comparison(actual: pd.Series, outputs: dict[str, ModelOutput], metrics: dict[str, MetricsRow]) -> None:
    all_values = [actual.to_numpy(dtype=float)]
    all_values.extend(output.forecast.reindex(actual.index).to_numpy(dtype=float) for output in outputs.values())
    ymin = min(float(np.nanmin(v)) for v in all_values)
    ymax = max(float(np.nanmax(v)) for v in all_values)
    pad = (ymax - ymin) * 0.08
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, MODEL_ORDER):
        forecast = outputs[name].forecast.reindex(actual.index)
        ax.plot(actual.index, actual.values, color="#0b0b0b", linewidth=1.6, label="Observed")
        ax.plot(forecast.index, forecast.values, color=MODEL_COLORS[name], linewidth=1.6, label="Forecast")
        ax.set_title(f"{MODEL_LABELS[name]}: MAPE {metrics[name].mape_test_pct:.2f}%")
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Load (MW)")
        ax.tick_params(axis="x", rotation=30)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    savefig(fig, FIGURES / "02_forecast_comparison.png")


def plot_metric_comparison(metrics_rows: list[MetricsRow]) -> None:
    rows = sorted(metrics_rows, key=lambda row: row.mape_test_pct)
    names = [row.name for row in rows]
    labels = [MODEL_LABELS[name] for name in names]
    mape = np.array([row.mape_test_pct for row in rows])
    rmse_scaled = np.array([row.rmse_test_mw / 1000.0 for row in rows])
    mae_scaled = np.array([row.mae_test_mw / 1000.0 for row in rows])
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(6, 6))
    bars = [
        ax.bar(x - width, mape, width, label="MAPE (%)", color="#2a78d6"),
        ax.bar(x, rmse_scaled, width, label="RMSE (thousand MW)", color="#eb6834"),
        ax.bar(x + width, mae_scaled, width, label="MAE (thousand MW)", color="#1baf7a"),
    ]
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Test-week error metrics by model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(loc="upper left")
    savefig(fig, FIGURES / "03_metric_comparison.png")


def plot_per_day_mape(actual: pd.Series, outputs: dict[str, ModelOutput]) -> None:
    days = pd.date_range(TEST_START, TEST_END, freq="D").date
    matrix = pd.DataFrame(index=[str(day) for day in days])
    for name in MODEL_ORDER:
        matrix[MODEL_LABELS[name]] = per_day_mape(actual, outputs[name].forecast).reindex(days).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 6))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="Blues", aspect="auto")
    ax.set_title("Per-day MAPE on the held-out test week")
    ax.set_xlabel("Model")
    ax.set_ylabel("Date (UTC)")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color="#0b0b0b", fontsize=8)
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label("MAPE (%)")
    savefig(fig, FIGURES / "04_per_day_mape.png")


def plot_winner_intervals(actual: pd.Series, output: ModelOutput, metrics: MetricsRow) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    if output.lower_95 is not None and output.upper_95 is not None:
        ax.fill_between(actual.index, output.lower_95.reindex(actual.index).values, output.upper_95.reindex(actual.index).values, color=MODEL_COLORS[output.name], alpha=0.16, label="95% interval")
    if output.lower_80 is not None and output.upper_80 is not None:
        ax.fill_between(actual.index, output.lower_80.reindex(actual.index).values, output.upper_80.reindex(actual.index).values, color=MODEL_COLORS[output.name], alpha=0.28, label="80% interval")
    ax.plot(actual.index, actual.values, color="#0b0b0b", linewidth=1.7, label="Observed")
    ax.plot(actual.index, output.forecast.reindex(actual.index).values, color=MODEL_COLORS[output.name], linewidth=1.8, label="Point forecast")
    cov80 = coverage(actual, output.lower_80, output.upper_80) if output.lower_80 is not None and output.upper_80 is not None else float("nan")
    cov95 = coverage(actual, output.lower_95, output.upper_95) if output.lower_95 is not None and output.upper_95 is not None else float("nan")
    ax.set_title(f"{MODEL_LABELS[output.name]} winner: MAPE {metrics.mape_test_pct:.2f}%, coverage 80%={cov80:.2f}, 95%={cov95:.2f}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="best")
    savefig(fig, FIGURES / "05_winner_with_intervals.png")


def plot_residuals(actual: pd.Series, output: ModelOutput) -> None:
    res = residuals(actual, output.forecast)
    frame = pd.DataFrame({"residual": res, "hour": res.index.hour, "day": [ts.strftime("%a %d") for ts in res.index]})
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    hour_data = [frame.loc[frame["hour"] == hour, "residual"].to_numpy(dtype=float) for hour in range(24)]
    axes[0].boxplot(hour_data, positions=np.arange(24), widths=0.55, showfliers=False)
    axes[0].axhline(0, color="#0b0b0b", linewidth=1)
    axes[0].set_title("Winner residuals by hour of day")
    axes[0].set_xlabel("Hour of day (UTC)")
    axes[0].set_ylabel("Actual minus forecast (MW)")
    day_labels = list(dict.fromkeys(frame["day"].tolist()))
    day_data = [frame.loc[frame["day"] == day, "residual"].to_numpy(dtype=float) for day in day_labels]
    axes[1].boxplot(day_data, tick_labels=day_labels, showfliers=False)
    axes[1].axhline(0, color="#0b0b0b", linewidth=1)
    axes[1].set_title("Winner residuals by test day")
    axes[1].set_xlabel("Day in test week (UTC)")
    axes[1].set_ylabel("Actual minus forecast (MW)")
    axes[1].tick_params(axis="x", rotation=35)
    savefig(fig, FIGURES / "06_residuals.png")


def plot_feature_importance(output: ModelOutput) -> None:
    if output.feature_importance is None:
        return
    importance = output.feature_importance.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(importance.index, importance.values, color=MODEL_COLORS["lightgbm"])
    ax.set_title("LightGBM feature importance")
    ax.set_xlabel("Gain importance (split gain)")
    ax.set_ylabel("Feature")
    savefig(fig, FIGURES / "07_feature_importance.png")


def metrics_to_csv(metrics_rows: list[MetricsRow], path: Path) -> None:
    rows: list[dict[str, object]] = []
    for row in metrics_rows:
        item = row.as_dict()
        item["hyperparameters"] = json.dumps(json_safe(row.hyperparameters), sort_keys=True)
        rows.append(item)
    pd.DataFrame(rows).sort_values("mape_test_pct").to_csv(path, index=False)


def write_metrics_json(metrics_rows: list[MetricsRow], outputs: dict[str, ModelOutput], actual: pd.Series, total_runtime_seconds: float) -> dict[str, object]:
    rows = sorted(metrics_rows, key=lambda row: row.mape_test_pct)
    winner_name = rows[0].name
    winner = outputs[winner_name]
    cov80 = coverage(actual, winner.lower_80, winner.upper_80) if winner.lower_80 is not None and winner.upper_80 is not None else float("nan")
    cov95 = coverage(actual, winner.lower_95, winner.upper_95) if winner.lower_95 is not None and winner.upper_95 is not None else float("nan")
    q10_loss = pinball_loss(actual, winner.q10, 0.1) if winner.q10 is not None else float("nan")
    q50_loss = pinball_loss(actual, winner.q50, 0.5) if winner.q50 is not None else float("nan")
    q90_loss = pinball_loss(actual, winner.q90, 0.9) if winner.q90 is not None else float("nan")
    payload = {
        "test_start": "2020-01-01",
        "test_end": "2020-01-07",
        "n_test_observations": 168,
        "models": [row.as_dict() for row in rows],
        "winner": winner_name,
        "winner_coverage_80pct": cov80,
        "winner_coverage_95pct": cov95,
        "winner_pinball_loss_q10": q10_loss,
        "winner_pinball_loss_q50": q50_loss,
        "winner_pinball_loss_q90": q90_loss,
        "total_runtime_seconds": total_runtime_seconds,
    }
    (ROOT / "metrics.json").write_text(json.dumps(payload, cls=NumpyEncoder, indent=2) + "\n", encoding="utf-8")
    return payload


def discussion_text(rows: list[MetricsRow], winner: str) -> str:
    best = rows[0]
    second = rows[1]
    naive_row = next(row for row in rows if row.name == "naive")
    prophet_row = next(row for row in rows if row.name == "prophet")
    lgbm_row = next(row for row in rows if row.name == "lightgbm")
    neural_rows = [row for row in rows if row.name in {"nbeats", "patchtst"}]
    neural_sentence = " and ".join(f"{MODEL_LABELS[row.name]} ({row.mape_test_pct:.2f}%)" for row in neural_rows)
    return (
        f"{MODEL_LABELS[winner]} won on the primary metric, with test MAPE of {best.mape_test_pct:.2f}%. "
        f"The margin over {MODEL_LABELS[second.name]} was {second.mape_test_pct - best.mape_test_pct:.2f} percentage points, so the ranking should be read as a short-horizon test-week result rather than a universal law. "
        f"The seasonal-naive baseline reached {naive_row.mape_test_pct:.2f}% MAPE, which is a strong reminder that German load is highly weekly. "
        f"A model has to add holiday handling, local trend adjustment, or nonlinear lag interactions to justify its extra machinery. "
        f"LightGBM's result ({lgbm_row.mape_test_pct:.2f}%) shows the value of simple, knowable calendar variables plus direct lag features: the model can use the Jan 1 holiday flag while still leaning on the previous day, previous week, and aligned previous year. "
        f"Prophet's German holidays make it interpretable and useful for explaining the holiday effect, but its smooth components can miss the exact level of an unusual week; here it scored {prophet_row.mape_test_pct:.2f}%. "
        f"The neural sequence models, {neural_sentence}, had enough flexibility to learn repeated daily and weekly shape, but this small bake-off gives them only one univariate series and a short validation target. "
        f"That is not the setting where deep models usually dominate. "
        f"SARIMA is the most transparent stochastic benchmark, but daily seasonality is only a proxy for the true 168-hour structure, so weekly and holiday mismatches appear as residual structure. "
        f"Across models, Jan 1 is the stress test: it is a Wednesday public holiday, so models that mainly copy ordinary Wednesdays or recent weekdays over-predict morning and daytime load."
    )


def recommendation_text(winner: str) -> str:
    return (
        f"For a JRC production short-term load forecasting pipeline under the same univariate-input rule, I would start from {MODEL_LABELS[winner]} because it had the lowest held-out MAPE in this bake-off. "
        "Before production use, I would expand the evaluation to rolling weekly backtests across multiple years, keep the holiday-vs-working-day breakdown as a standing diagnostic, calibrate prediction intervals on recent residuals, and only then consider weather variables in a separate, clearly labelled multivariate study."
    )


def write_transcript(metrics_rows: list[MetricsRow], metrics_payload: dict[str, object], outputs: dict[str, ModelOutput], uname: str, series: pd.Series) -> None:
    rows = sorted(metrics_rows, key=lambda row: row.mape_test_pct)
    winner = str(metrics_payload["winner"])
    train_n = len(train_series(series))
    val_n = len(validation_series(series))
    final_n = len(final_training_series(series))
    table_lines = [
        "| Rank | Model | MAPE (%) | RMSE (MW) | MAE (MW) | Jan 1 MAPE (%) | Jan 2 to Jan 7 MAPE (%) | Runtime (s) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        table_lines.append(
            f"| {rank} | {MODEL_LABELS[row.name]} | {row.mape_test_pct:.2f} | {row.rmse_test_mw:.0f} | {row.mae_test_mw:.0f} | {row.mape_jan1_pct:.2f} | {row.mape_jan2_to_jan7_pct:.2f} | {row.runtime_seconds:.1f} |"
        )
    notes: list[str] = []
    for output in outputs.values():
        notes.extend(output.notes)
    note_text = "\n\n".join(f"Note: {note}" for note in notes)
    content = f"""# German hourly load forecasting bake-off

## Data

The study uses `opsd_de_load.csv`, an Open Power System Data extract derived from the ENTSO-E Transparency Platform. It contains German national hourly load from 2015-01-01 00:00 UTC to 2020-09-30 23:00 UTC, with 50,400 rows and one target column, `DE_load_actual_entsoe_transparency`, measured in megawatts. The timestamp index was rechecked as hourly, gap-free, duplicate-free, and without missing target values, so no imputation was needed.

## Why these six models

The six models form a complexity gradient. The seasonal-naive model asks how much skill comes from simply copying the same hour from the previous week. SARIMA adds a classical stochastic time-series structure. Prophet adds explicit daily, weekly, yearly, and German-holiday components. LightGBM uses timestamp features, public-holiday flags, lags, and rolling summaries in a supervised learning setup. N-BEATS and the transformer-family TSMixer substitute test whether modern neural sequence models extract more from the same univariate history.

## Validation strategy

The initial training window was 2015-01-01 to 2019-09-30 ({train_n:,} hourly rows), validation was 2019-10-01 to 2019-12-31 ({val_n:,} rows), and the held-out test was 2020-01-01 to 2020-01-07 (168 rows). Hyperparameters or model orders were selected only on the validation window. After selection, each non-naive model was refit on train plus validation ({final_n:,} rows) before the single final forecast of the test week.

## Results table

{chr(10).join(table_lines)}

## Discussion

{discussion_text(rows, winner)}

## Recommendation

{recommendation_text(winner)}

## Reproducibility note

Random seed 42 was used for LightGBM and the Darts neural models. The categorical figure palette is the validated six-slot light-mode subset from the data-visualization reference palette: blue, orange, aqua, yellow, magenta, and green. Total wall-clock runtime was {float(metrics_payload['total_runtime_seconds']):.1f} seconds. Hardware and OS from `uname -a`: `{uname}`.

{note_text}
"""
    (ROOT / "transcript.md").write_text(content.strip() + "\n", encoding="utf-8")


def run_all() -> None:
    total_start = perf_counter()
    FIGURES.mkdir(exist_ok=True)
    CODE.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    set_style()
    df = load_data()
    series = load_series(df)
    actual = test_series(series)

    outputs: dict[str, ModelOutput] = {}
    metrics_rows: list[MetricsRow] = []
    for name in MODEL_ORDER:
        run_model_subprocess(name)
        output = load_model_output(name)
        outputs[name] = output
        metrics_rows.append(evaluate_output(output, actual))

    metrics_to_csv(metrics_rows, ROOT / "metrics.csv")
    total_runtime = perf_counter() - total_start
    metrics_payload = write_metrics_json(metrics_rows, outputs, actual, total_runtime)

    plot_overview(series)
    metrics_by_name = {row.name: row for row in metrics_rows}
    plot_forecast_comparison(actual, outputs, metrics_by_name)
    plot_metric_comparison(metrics_rows)
    plot_per_day_mape(actual, outputs)
    winner_name = str(metrics_payload["winner"])
    plot_winner_intervals(actual, outputs[winner_name], metrics_by_name[winner_name])
    plot_residuals(actual, outputs[winner_name])
    sorted_names = [row.name for row in sorted(metrics_rows, key=lambda row: row.mape_test_pct)]
    if "lightgbm" in sorted_names[:2]:
        plot_feature_importance(outputs["lightgbm"])
    uname = platform.uname()
    uname_text = " ".join([uname.system, uname.node, uname.release, uname.version, uname.machine])
    write_transcript(metrics_rows, metrics_payload, outputs, uname_text, series)
    print(f"ALL_DONE winner={winner_name} total_runtime={total_runtime:.1f}", flush=True)


if __name__ == "__main__":
    run_all()
