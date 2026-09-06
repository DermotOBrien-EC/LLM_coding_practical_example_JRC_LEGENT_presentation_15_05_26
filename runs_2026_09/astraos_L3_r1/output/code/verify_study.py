from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"naive", "sarima", "prophet", "lightgbm", "nbeats", "patchtst"}
QUANTILES = [0.025, 0.1, 0.5, 0.9, 0.975]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-9, atol=1e-7):
        raise AssertionError(f"{name}: {actual} differs from independently computed {expected}")


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def main() -> None:
    data = pd.read_csv(ROOT / "opsd_de_load.csv")
    dates = pd.to_datetime(data.utc_timestamp, utc=True).dt.tz_localize(None)
    expected = pd.date_range("2015-01-01", "2020-09-30 23:00", freq="h")
    check(pd.DatetimeIndex(dates).equals(expected), "Raw timestamp grid changed")
    observed = data.DE_load_actual_entsoe_transparency.to_numpy(float)
    check(bool(np.isfinite(observed).all() and (observed > 0).all()), "Invalid raw load")
    mask = (dates >= "2020-01-01") & (dates < "2020-01-08")
    actual = observed[mask]
    test_dates = pd.DatetimeIndex(dates[mask])
    validation_dates = pd.date_range("2019-10-01", "2019-12-31 23:00", freq="h")
    validation_actual = observed[(dates >= "2019-10-01") & (dates < "2020-01-01")]
    result: dict[str, Any] = json.loads((ROOT / "metrics.json").read_text(), parse_constant=reject_constant)
    run = json.loads((ROOT / "artifacts" / "run.json").read_text())
    check(hashlib.sha256((ROOT / "opsd_de_load.csv").read_bytes()).hexdigest() == run["data_sha256"], "Input digest mismatch")
    for relative, expected_digest in run["code_sha256"].items():
        check(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected_digest,
              f"Code changed since fitting: {relative}")
    check(result["n_test_observations"] == len(actual) == 168, "Incorrect test count")
    check(result["test_start"] == "2020-01-01" and result["test_end"] == "2020-01-07", "Wrong test dates")
    check(len(result["models"]) == 6 and {r["name"] for r in result["models"]} == MODELS, "Model set mismatch")
    csv_rows = {r["name"]: r for r in csv.DictReader((ROOT / "metrics.csv").open())}
    check(set(csv_rows) == MODELS, "CSV models mismatch")
    recalculated = []
    coverage: dict[str, float] = {}
    for row in result["models"]:
        name = row["name"]
        f = pd.read_csv(ROOT / "artifacts" / f"{name}_forecast.csv", index_col=0, parse_dates=True)
        v = pd.read_csv(ROOT / "artifacts" / f"{name}_validation.csv", index_col=0, parse_dates=True)
        metadata = json.loads((ROOT / "artifacts" / f"{name}.json").read_text())
        check(f.index.equals(test_dates), f"{name}: forecast alignment")
        check(v.index.equals(validation_dates), f"{name}: validation alignment")
        check(bool(np.isfinite(f.to_numpy()).all()), f"{name}: non-finite forecast")
        error = actual - f.point.to_numpy()
        percentages = 100 * np.abs(error) / actual
        computed = {"mape_test_pct": percentages.mean(), "rmse_test_mw": np.sqrt(np.mean(error * error)),
                    "mae_test_mw": np.abs(error).mean(), "mape_jan1_pct": percentages[:24].mean(),
                    "mape_jan2_to_jan7_pct": percentages[24:].mean()}
        for key, value in computed.items():
            assert_close(row[key], float(value), f"{name}.{key}")
            assert_close(float(csv_rows[name][key]), float(value), f"CSV {name}.{key}")
        check(json.loads(csv_rows[name]["hyperparameters"]) == row["hyperparameters"], f"{name}: CSV hyperparameters differ")
        check(row["hyperparameters"] == metadata["hyperparameters"], f"{name}: selected configuration differs")
        for key, value in result.items():
            if key == "models":
                continue
            if isinstance(value, (float, int)):
                assert_close(float(csv_rows[name][key]), value, f"CSV {name}.{key}")
            else:
                check(csv_rows[name][key] == value, f"CSV {name}.{key} differs")
        validation_score = float(np.mean(np.abs(validation_actual - v.point.to_numpy()) / validation_actual) * 100)
        selected_score = min(c["validation_mape_pct"] for c in metadata["validation_candidates"])
        assert_close(validation_score, selected_score, f"{name}: validation selection")
        if name in {"nbeats", "patchtst"}:
            chosen = min(metadata["validation_candidates"], key=lambda c: c["validation_mape_pct"])
            check(chosen["epoch"] == row["hyperparameters"]["selected_epochs"], f"{name}: wrong refit epoch count")
            check(row["hyperparameters"]["selected_epochs"] <= 30, f"{name}: excessive epochs")
            check(row["hyperparameters"]["training_stride_hours"] == 168, f"{name}: unexpected training stride")
        if name == "naive":
            naive_expected = observed[np.flatnonzero(mask) - 168]
            np.testing.assert_array_equal(f.point.to_numpy(), naive_expected)
            check(list(f.columns) == ["point"], "Naive incorrectly claims native intervals")
        else:
            q = f[[f"q{x:g}" for x in QUANTILES]].to_numpy()
            check(bool((np.diff(q, axis=1) >= 0).all()), f"{name}: crossed quantiles")
            if name != "prophet":
                np.testing.assert_allclose(q[:, 2], f.point, atol=1e-7)
            if name == result["winner"]:
                for nominal, lower, upper in [(80, 1, 3), (95, 0, 4)]:
                    fraction = float(np.count_nonzero((actual >= q[:, lower]) & (actual <= q[:, upper])) / 168)
                    assert_close(result[f"winner_coverage_{nominal}pct"], fraction, f"coverage {nominal}")
                    coverage[str(nominal)] = fraction
                for index, probability, label in [(1, 0.1, "q10"), (2, 0.5, "q50"), (3, 0.9, "q90")]:
                    residual = actual - q[:, index]
                    loss = np.where(residual >= 0, probability * residual, (probability - 1) * residual).mean()
                    assert_close(result[f"winner_pinball_loss_{label}"], float(loss), f"pinball {label}")
        recalculated.append((name, float(percentages.mean())))
    check(result["winner"] == min(recalculated, key=lambda x: x[1])[0], "Incorrect winner")
    check([x[1] for x in recalculated] == sorted(x[1] for x in recalculated), "Table not ranked by MAPE")
    required_figures = ["01_overview.png", "02_forecast_comparison.png", "03_metric_comparison.png",
                        "04_per_day_mape.png", "05_winner_with_intervals.png", "06_residuals.png"]
    top_two = {r["name"] for r in result["models"][:2]}
    if "lightgbm" in top_two:
        required_figures.append("07_feature_importance.png")
    if "prophet" in top_two:
        required_figures.append("08_decomposition.png")
    for filename in required_figures:
        with Image.open(ROOT / "figures" / filename) as image:
            square = filename[:2] in {"03", "04", "07"}
            check(image.size == ((1800, 1800) if square else (3300, 1800)), f"{filename}: wrong dimensions")
            check(all(abs(dpi - 300) < 0.1 for dpi in image.info["dpi"]), f"{filename}: wrong resolution")
            check(float(np.asarray(image.convert("L")).std()) > 10, f"{filename}: apparently blank image")
    transcript = (ROOT / "transcript.md").read_text()
    discussion = transcript.split("## 5. Discussion\n\n")[1].split("\n\n## 6.")[0]
    check(200 <= len(discussion.split()) <= 400, "Discussion word count outside specified range")
    check("TransformerModel" in transcript and "PatchTST" in transcript, "Missing substitution disclosure")
    for path in list((ROOT / "code").glob("*.py")) + list(ROOT.glob("*.md")):
        if path.name != "AGENTS.md":
            check(chr(0x2014) not in path.read_text(), f"Forbidden em dash in {path.name}")
    report = {"status": "passed", "checks": ["raw grid and source digest", "all six metric rows",
              "CSV/JSON agreement", "validation argmin and epoch selection", "forecast alignment",
              "naive lag identity", "monotone quantiles and winner coverage/pinball",
              "figure dimensions and DPI", "discussion length and substitution disclosure", "prose dash scan"],
              "n_figures": len(required_figures), "discussion_words": len(discussion.split()),
              "winner": result["winner"], "winner_coverage": coverage}
    (ROOT / "artifacts" / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
