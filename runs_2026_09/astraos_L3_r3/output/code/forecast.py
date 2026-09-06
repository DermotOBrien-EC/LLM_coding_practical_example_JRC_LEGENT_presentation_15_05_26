from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import types
from pathlib import Path
from time import perf_counter

# Keep the required local prophet.py from shadowing the third-party prophet package.
DIRECTORY = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != DIRECTORY]
PACKAGE = types.ModuleType("study")
PACKAGE.__path__ = [str(DIRECTORY)]
sys.modules.setdefault("study", PACKAGE)

from study.common import (  # noqa: E402
    NAMES,
    ROOT,
    TEST_INDEX,
    TEST_START,
    load_data,
    load_result,
    mape,
    quantile_diagnostics,
    save_result,
    score,
    source_digest,
    write_json,
)


def render(total_runtime: float) -> None:
    import pandas as pd
    from study.figures import make_figures
    from study.report import write_report

    manifest = json.loads((ROOT / "artifacts" / "run_manifest.json").read_text())
    if source_digest() != manifest["data_sha256"]:
        raise ValueError("Input data changed since fitting; refusing to rescore saved forecasts")
    data = load_data()
    actual = data.loc[TEST_INDEX].to_numpy()
    results = [load_result(name) for name in NAMES]
    rows = []
    for result in results:
        metrics = score(actual, result.point)
        rows.append(
            {
                "name": result.name,
                **metrics,
                "mape_jan1_pct": mape(actual[:24], result.point[:24]),
                "mape_jan2_to_jan7_pct": mape(actual[24:], result.point[24:]),
                "runtime_seconds": result.runtime_seconds,
                "hyperparameters": result.hyperparameters,
            }
        )
    rows.sort(key=lambda row: row["mape_test_pct"])
    winner = next(result for result in results if result.name == rows[0]["name"])
    if winner.quantiles is None:
        diagnostics = {
            key: None
            for key in [
                "winner_coverage_80pct",
                "winner_coverage_95pct",
                "winner_pinball_loss_q10",
                "winner_pinball_loss_q50",
                "winner_pinball_loss_q90",
            ]
        }
    else:
        diagnostics = quantile_diagnostics(actual, winner.quantiles)
    summary = {
        "test_start": "2020-01-01",
        "test_end": "2020-01-07",
        "n_test_observations": 168,
        "models": rows,
        "winner": winner.name,
        **diagnostics,
        "total_runtime_seconds": total_runtime,
    }
    write_json(ROOT / "metrics.json", summary)
    flattened = []
    for row in rows:
        flattened.append(
            {
                **{key: value for key, value in summary.items() if key != "models"},
                **row,
                "hyperparameters": json.dumps(row["hyperparameters"], sort_keys=True),
            }
        )
    pd.DataFrame(flattened).to_csv(ROOT / "metrics.csv", index=False)
    joined = pd.DataFrame({"utc_timestamp": TEST_INDEX, "actual_mw": actual})
    for result in results:
        joined[result.name] = result.point
    joined.to_csv(ROOT / "artifacts" / "test_predictions.csv", index=False)
    make_figures(data, results, summary)
    write_report(results, summary)
    print(
        json.dumps(
            {"winner": winner.name, "mape_test_pct": rows[0]["mape_test_pct"], **diagnostics}
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-free German load forecasting bake-off")
    parser.add_argument("--models", nargs="+", choices=NAMES, default=NAMES)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Regenerate tables, figures and report from saved forecasts",
    )
    args = parser.parse_args()
    if args.render_only:
        manifest = json.loads((ROOT / "artifacts" / "run_manifest.json").read_text())
        render(manifest["total_runtime_seconds"])
        return
    start = perf_counter()
    data = load_data()
    pretest = data.loc[data.index < TEST_START].copy()
    for name in args.models:
        print(f"START {name}", flush=True)
        module_name = "lightgbm_features" if name == "lightgbm" else name
        module = importlib.import_module(f"study.{module_name}")
        result = module.run(pretest)
        save_result(result)
        print(f"DONE {name} runtime_seconds={result.runtime_seconds:.3f}", flush=True)
    elapsed = perf_counter() - start
    if args.models == NAMES:
        manifest = {
            "total_runtime_seconds": elapsed,
            "seed": 2026,
            "data_sha256": source_digest(),
            "uname": subprocess.check_output(["uname", "-a"], text=True).strip(),
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in [
                    "pandas",
                    "numpy",
                    "matplotlib",
                    "statsmodels",
                    "pmdarima",
                    "darts",
                    "lightgbm",
                    "holidays",
                    "torch",
                    "pytorch-lightning",
                    "prophet",
                    "scipy",
                ]
            },
            "train_rows": 41616,
            "validation_rows": 2208,
            "refit_rows": 43824,
            "test_rows": 168,
            "validation_origins": 14,
            "training_complete_before_test_scoring": True,
        }
        write_json(ROOT / "artifacts" / "run_manifest.json", manifest)
        rendering_start = perf_counter()
        render(elapsed)
        # Include initial figure rendering and reporting, then refresh only timing-bearing outputs.
        manifest["total_runtime_seconds"] = perf_counter() - start
        manifest["rendering_seconds"] = perf_counter() - rendering_start
        write_json(ROOT / "artifacts" / "run_manifest.json", manifest)
        summary = json.loads((ROOT / "metrics.json").read_text())
        summary["total_runtime_seconds"] = manifest["total_runtime_seconds"]
        write_json(ROOT / "metrics.json", summary)
        import pandas as pd

        table = pd.read_csv(ROOT / "metrics.csv")
        table["total_runtime_seconds"] = manifest["total_runtime_seconds"]
        table.to_csv(ROOT / "metrics.csv", index=False)
        from study.report import write_report

        write_report([load_result(name) for name in NAMES], summary)
        print(f"ALL_DONE total_runtime_seconds={manifest['total_runtime_seconds']:.3f}", flush=True)


if __name__ == "__main__":
    main()
