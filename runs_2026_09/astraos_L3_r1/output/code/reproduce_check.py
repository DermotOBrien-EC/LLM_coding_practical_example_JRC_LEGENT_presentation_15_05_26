from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    paths = sorted((ROOT / "artifacts").glob("*_forecast.csv")) + sorted((ROOT / "artifacts").glob("*_validation.csv"))
    if len(paths) != 12:
        raise ValueError("Run forecast.py first; expected six test and six validation forecast files.")
    before = {path.name: digest(path) for path in paths}
    reference_path = ROOT / "artifacts" / "reproduction_reference.json"
    reference_path.write_text(json.dumps(before, indent=2) + "\n")
    started = time.perf_counter()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with (ROOT / "reproduction.log").open("w") as log:
        subprocess.run([sys.executable, str(ROOT / "code" / "forecast.py")],
                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    after = {path.name: digest(path) for path in paths}
    mismatches = [name for name in before if before[name] != after[name]]
    report = {"status": "passed" if not mismatches else "failed", "forecast_files_checked": len(paths),
              "byte_identical_forecast_files": len(paths) - len(mismatches), "mismatches": mismatches,
              "full_command_wall_seconds": time.perf_counter() - started,
              "scope": "Two independent full fits, identical saved test points/quantiles and validation points; runtimes may differ."}
    (ROOT / "artifacts" / "reproducibility.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if mismatches:
        raise AssertionError("Forecast files changed during deterministic reproduction")


if __name__ == "__main__":
    main()
