from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    metadata = json.loads((ROOT / "metrics.json").read_text())
    manifest = json.loads((ROOT / "artifacts" / "run_manifest.json").read_text())
    source = pd.read_csv(ROOT / "opsd_de_load.csv")
    timestamps = pd.to_datetime(source.utc_timestamp, utc=True)
    mask = (timestamps >= "2020-01-01") & (timestamps < "2020-01-08")
    actual = source.loc[mask, "DE_load_actual_entsoe_transparency"].to_numpy()
    target_dates = pd.date_range("2020-01-01", periods=168, freq="h", tz="UTC")
    assert len(actual) == 168
    names = {"naive", "sarima", "prophet", "lightgbm", "nbeats", "patchtst"}
    assert {row["name"] for row in metadata["models"]} == names
    assert len(metadata["models"]) == 6
    flat = pd.read_csv(ROOT / "metrics.csv").set_index("name")
    results: dict[str, float] = {}
    quantile_arrays: dict[str, np.ndarray] = {}
    for row in metadata["models"]:
        name = row["name"]
        predictions = pd.read_csv(ROOT / "artifacts" / f"{name}_forecast.csv")
        assert pd.DatetimeIndex(pd.to_datetime(predictions.utc_timestamp, utc=True)).equals(
            target_dates
        )
        pred = predictions.point_mw.to_numpy()
        assert pred.shape == (168,) and np.isfinite(pred).all()
        error = actual - pred
        expected = {
            "mape_test_pct": float((np.abs(error) / actual).mean() * 100),
            "rmse_test_mw": float(np.sqrt((error * error).mean())),
            "mae_test_mw": float(np.abs(error).mean()),
            "mape_jan1_pct": float((np.abs(error[:24]) / actual[:24]).mean() * 100),
            "mape_jan2_to_jan7_pct": float((np.abs(error[24:]) / actual[24:]).mean() * 100),
        }
        for key, value in expected.items():
            np.testing.assert_allclose(row[key], value, rtol=1e-10, atol=1e-9)
            np.testing.assert_allclose(flat.loc[name, key], value, rtol=1e-10, atol=1e-9)
        assert json.loads(flat.loc[name, "hyperparameters"]) == row["hyperparameters"]
        results[name] = expected["mape_test_pct"]
        run = json.loads((ROOT / "artifacts" / f"{name}_run.json").read_text())
        assert run["validation"] and all(
            np.isfinite(item["mape_validation_pct"]) for item in run["validation"]
        )
        if name != "naive":
            q = predictions[
                [f"q{level:g}_mw" for level in [0.025, 0.1, 0.5, 0.9, 0.975]]
            ].to_numpy()
            assert q.shape == (168, 5) and np.isfinite(q).all()
            assert (np.diff(q, axis=1) >= 0).all()
            quantile_arrays[name] = q
        if name in {"nbeats", "patchtst"}:
            hp = run["hyperparameters"]
            assert 1 <= hp["selected_epochs"] <= hp["maximum_epochs"] <= 30
            assert hp["input_chunk_length"] == hp["output_chunk_length"] == 168
            best_epoch = min(run["validation"], key=lambda record: record["mape_validation_pct"])[
                "epoch"
            ]
            assert hp["selected_epochs"] == best_epoch
            assert run["diagnostics"]["public_api_validation_agreement"]
    winner = min(results, key=results.get)
    assert metadata["winner"] == winner
    if winner in quantile_arrays:
        q = quantile_arrays[winner]
        for level, lower, upper in [(80, 1, 3), (95, 0, 4)]:
            expected_coverage = float(
                np.count_nonzero((actual >= q[:, lower]) & (actual <= q[:, upper])) / 168
            )
            np.testing.assert_allclose(
                metadata[f"winner_coverage_{level}pct"], expected_coverage, atol=1e-12
            )
        for level, column, suffix in [(0.1, 1, "q10"), (0.5, 2, "q50"), (0.9, 3, "q90")]:
            delta = actual - q[:, column]
            expected_pinball = float(
                np.where(delta >= 0, level * delta, (1 - level) * -delta).mean()
            )
            np.testing.assert_allclose(
                metadata[f"winner_pinball_loss_{suffix}"], expected_pinball, rtol=1e-10
            )
    for key in [
        "winner_coverage_80pct",
        "winner_coverage_95pct",
        "winner_pinball_loss_q10",
        "winner_pinball_loss_q50",
        "winner_pinball_loss_q90",
        "total_runtime_seconds",
    ]:
        if metadata[key] is not None:
            np.testing.assert_allclose(flat[key].to_numpy(), metadata[key], rtol=1e-10)
    figure_names = [
        "01_overview.png",
        "02_forecast_comparison.png",
        "03_metric_comparison.png",
        "04_per_day_mape.png",
        "05_winner_with_intervals.png",
        "06_residuals.png",
    ]
    top_two = sorted(results, key=results.get)[:2]
    if "lightgbm" in top_two:
        figure_names.append("07_feature_importance.png")
    if "prophet" in top_two:
        figure_names.append("08_decomposition.png")
    for name in figure_names:
        with Image.open(ROOT / "figures" / name) as image:
            assert image.format == "PNG"
            expected_size = (1800, 1800) if name.startswith(("03", "04", "07")) else (3300, 1800)
            assert image.size == expected_size, (name, image.size)
            assert all(abs(dpi - 300) < 0.1 for dpi in image.info["dpi"])
    for name in [
        "common.py",
        "naive.py",
        "sarima.py",
        "prophet.py",
        "lightgbm_features.py",
        "nbeats.py",
        "patchtst.py",
        "forecast.py",
    ]:
        assert (ROOT / "code" / name).is_file()
    assert (
        hashlib.sha256((ROOT / "opsd_de_load.csv").read_bytes()).hexdigest()
        == manifest["data_sha256"]
    )
    report = (ROOT / "transcript.md").read_text()
    assert chr(0x2014) not in report
    discussion = report.split("## 5. Discussion\n\n")[1].split("\n## 6. Recommendation")[0]
    word_count = len(discussion.split())
    assert 200 <= word_count <= 400, word_count
    audit = {
        "status": "PASS",
        "metric_rows_independently_recomputed": 6,
        "quantile_forecasts_checked": 5,
        "figures_checked": figure_names,
        "discussion_words": word_count,
        "report_words": len(report.split()),
        "successful_runtime_under_30_minutes": metadata["total_runtime_seconds"] < 1800,
        "data_sha256_verified": True,
    }
    (ROOT / "artifacts" / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
