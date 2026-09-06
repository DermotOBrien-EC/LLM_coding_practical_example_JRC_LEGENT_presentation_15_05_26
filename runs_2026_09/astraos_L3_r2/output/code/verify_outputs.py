from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    report = json.loads((ROOT / "metrics.json").read_text())
    csv = pd.read_csv(ROOT / "metrics.csv")
    data = pd.read_csv(ROOT / "opsd_de_load.csv")
    timestamps = pd.to_datetime(data.utc_timestamp, utc=True)
    actual = data.loc[
        (timestamps >= "2020-01-01") & (timestamps < "2020-01-08"),
        "DE_load_actual_entsoe_transparency",
    ].to_numpy()
    assert len(actual) == 168
    assert set(csv.name) == {"naive", "sarima", "prophet", "lightgbm", "nbeats", "patchtst"}
    per_day = pd.read_csv(ROOT / "artifacts" / "per_day_mape.csv")
    checks: list[str] = []
    for row in report["models"]:
        name = row["name"]
        frame = pd.read_csv(ROOT / "artifacts" / f"{name}_forecast.csv")
        index = pd.DatetimeIndex(pd.to_datetime(frame.utc_timestamp, utc=True))
        assert index.equals(pd.date_range("2020-01-01", periods=168, freq="h", tz="UTC"))
        prediction = frame.point_mw.to_numpy()
        absolute = np.abs(actual - prediction)
        errors = 100 * absolute / actual
        expected = {
            "mape_test_pct": errors.mean(),
            "mae_test_mw": absolute.mean(),
            "rmse_test_mw": np.sqrt(np.square(actual - prediction).mean()),
            "mape_jan1_pct": errors[:24].mean(),
            "mape_jan2_to_jan7_pct": errors[24:].mean(),
        }
        for key, value in expected.items():
            np.testing.assert_allclose(row[key], value, rtol=1e-10)
            np.testing.assert_allclose(csv.loc[csv.name == name, key].iloc[0], value, rtol=1e-10)
        np.testing.assert_allclose(per_day[name], errors.reshape(7, 24).mean(axis=1))
        if name == "naive":
            reference = data.loc[
                (timestamps >= "2019-12-25") & (timestamps < "2020-01-01"),
                "DE_load_actual_entsoe_transparency",
            ].to_numpy()
            np.testing.assert_array_equal(prediction, reference)
        else:
            q = frame[[f"q{x:g}_mw" for x in [0.025, 0.1, 0.5, 0.9, 0.975]]].to_numpy()
            assert np.all(np.isfinite(q)) and np.all(np.diff(q, axis=1) >= 0)
            if name == report["winner"]:
                for nominal, lo, hi in [(80, 1, 3), (95, 0, 4)]:
                    coverage = np.count_nonzero((q[:, lo] <= actual) & (actual <= q[:, hi])) / 168
                    np.testing.assert_allclose(report[f"winner_coverage_{nominal}pct"], coverage)
                for j, level, label in [(1, 0.1, "q10"), (2, 0.5, "q50"), (3, 0.9, "q90")]:
                    deviation = actual - q[:, j]
                    pinball = np.where(
                        deviation >= 0, level * deviation, (level - 1) * deviation
                    ).mean()
                    np.testing.assert_allclose(report[f"winner_pinball_loss_{label}"], pinball)
        checks.append(f"{name}: independent score recomputation and forecast-grid checks passed")
    assert report["winner"] == min(report["models"], key=lambda row: row["mape_test_pct"])["name"]
    required = [
        "01_overview.png",
        "02_forecast_comparison.png",
        "03_metric_comparison.png",
        "04_per_day_mape.png",
        "05_winner_with_intervals.png",
        "06_residuals.png",
    ]
    top_two = {
        row["name"] for row in sorted(report["models"], key=lambda row: row["mape_test_pct"])[:2]
    }
    if "lightgbm" in top_two:
        required.append("07_feature_importance.png")
    if "prophet" in top_two:
        required.append("08_decomposition.png")
    for name in required:
        with Image.open(ROOT / "figures" / name) as image:
            expected_size = (1800, 1800) if name[:2] in ["03", "04", "07"] else (3300, 1800)
            assert image.size == expected_size, (name, image.size)
            assert abs(image.info["dpi"][0] - 300) < 0.1
    text = (ROOT / "transcript.md").read_text()
    assert "—" not in text
    discussion = text.split("## 5. Discussion\n", 1)[1].split("## 6. Recommendation", 1)[0]
    discussion_words = len(re.findall(r"\S+", discussion))
    assert 200 <= discussion_words <= 400, discussion_words
    provenance = json.loads((ROOT / "artifacts" / "provenance.json").read_text())
    for path, digest in provenance["artifact_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    output: dict[str, Any] = {
        "checks": checks,
        "required_figures_checked": required,
        "figure_dimensions_and_300dpi": True,
        "discussion_words": discussion_words,
        "transcript_words": len(re.findall(r"\S+", text)),
        "artifact_digests_match": True,
    }
    (ROOT / "artifacts" / "verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
