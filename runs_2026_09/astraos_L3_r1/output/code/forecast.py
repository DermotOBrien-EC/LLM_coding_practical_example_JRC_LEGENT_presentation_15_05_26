from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
# The required prophet.py filename must not shadow the installed Prophet package.
sys.path = [entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != CODE]
sys.dont_write_bytecode = True
for key, folder in {"MPLCONFIGDIR": "matplotlib", "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
                    "XDG_CACHE_HOME": "xdg", "TORCH_HOME": "torch", "HF_HOME": "huggingface"}.items():
    local_directory = ROOT / ".cache" / folder
    local_directory.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(local_directory)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"


def load_module(name: str, filename: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load_module("common", "common.py")

import numpy as np
import pandas as pd


def save_result(result: Any, future: pd.DatetimeIndex, validation_index: pd.DatetimeIndex) -> None:
    target = ROOT / "artifacts"
    target.mkdir(exist_ok=True)
    forecast = pd.DataFrame({"point": result.point}, index=future)
    if result.quantiles is not None:
        forecast = pd.concat([forecast, common.quantile_frame(result.quantiles, future)], axis=1)
    forecast.index.name = "utc_timestamp"
    forecast.to_csv(target / f"{result.name}_forecast.csv", float_format="%.12g")
    pd.DataFrame({"point": result.validation_point}, index=validation_index).to_csv(
        target / f"{result.name}_validation.csv", index_label="utc_timestamp", float_format="%.12g")
    extras: dict[str, Any] = {}
    for key, value in result.extras.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(target / f"{result.name}_{key}.csv", index=False, float_format="%.12g")
        elif key != "fitted_prophet":
            extras[key] = value
    common.dump_json(target / f"{result.name}.json", {
        "name": result.name, "runtime_seconds": result.runtime_seconds,
        "hyperparameters": result.hyperparameters,
        "validation_candidates": result.validation_candidates, "extras": extras,
    })


def read_results() -> list[Any]:
    results = []
    for name in common.NAMES:
        metadata = json.loads((ROOT / "artifacts" / f"{name}.json").read_text())
        forecast = pd.read_csv(ROOT / "artifacts" / f"{name}_forecast.csv", index_col=0, parse_dates=True)
        validation = pd.read_csv(ROOT / "artifacts" / f"{name}_validation.csv", index_col=0, parse_dates=True)
        q = forecast.iloc[:, 1:].to_numpy() if len(forecast.columns) > 1 else None
        results.append(common.ModelResult(name, forecast["point"].to_numpy(), q,
                                          validation["point"].to_numpy(), metadata["hyperparameters"],
                                          metadata["validation_candidates"], metadata["runtime_seconds"],
                                          metadata["extras"]))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the six-model fixed-origin load study.")
    parser.add_argument("--train-only", action="store_true", help="Write model artifacts without reports.")
    parser.add_argument("--report-only", action="store_true", help="Rebuild reports from saved model artifacts.")
    args = parser.parse_args()
    if args.train_only and args.report_only:
        parser.error("Choose at most one mode.")
    started = time.perf_counter()
    data = common.load_data()
    train = data.loc[:"2019-09-30 23:00"].copy()
    pretest = data.loc[:"2019-12-31 23:00"].copy()
    future = pd.date_range("2020-01-01", periods=168, freq="h")
    if not args.report_only:
        common.seed_everything()
        for name, filename in zip(common.NAMES, ["naive.py", "sarima.py", "prophet.py", "lightgbm_features.py", "nbeats.py", "patchtst.py"]):
            print(f"START {name}", flush=True)
            module = load_module(f"study_{name}", filename)
            result = module.run(train.copy(), pretest.copy(), future.copy())
            if result.point.shape != (168,) or not np.isfinite(result.point).all():
                raise ValueError(f"Invalid forecast for {name}")
            save_result(result, future, pretest.index[len(train):])
            print(f"DONE {name} runtime_seconds={result.runtime_seconds:.2f}", flush=True)
        versions = {name: importlib.metadata.version(name) for name in
                    ["numpy", "pandas", "matplotlib", "statsmodels", "darts", "lightgbm", "holidays", "torch", "pytorch-lightning", "prophet"]}
        common.dump_json(ROOT / "artifacts" / "run.json", {
            "training_wall_seconds": time.perf_counter() - started, "seed": common.SEED,
            "python": platform.python_version(), "packages": versions,
            "uname": subprocess.check_output(["uname", "-a"], text=True).strip(),
            "machine": platform.machine(), "logical_cpu_count": os.cpu_count(),
            "data_sha256": common.source_digest(), "train_rows": len(train),
            "validation_rows": len(pretest) - len(train), "refit_rows": len(pretest),
            "test_rows": len(future),
            "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(CODE.glob("*.py"))},
        })
    if not args.train_only:
        reporting = load_module("study_reporting", "reporting.py")
        reporting.generate(data, read_results())
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
