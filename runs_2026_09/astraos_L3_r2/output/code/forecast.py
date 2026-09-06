from __future__ import annotations

# ruff: noqa: E402
# Package bootstrapping must precede imports to prevent Prophet name shadowing.

import argparse
import importlib
import importlib.util
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "code"
# Local prophet.py must not shadow the third-party Prophet package.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != CODE]
spec = importlib.util.spec_from_file_location("load_bakeoff", CODE / "__init__.py")
package = importlib.util.module_from_spec(spec)
sys.modules["load_bakeoff"] = package
spec.loader.exec_module(package)

from load_bakeoff.common import NAMES, TEST_START, load_data, save_result, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the six-model German load forecasting study"
    )
    parser.add_argument(
        "--model", choices=NAMES, help="Run one model without scoring test observations"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild reports from saved forecasts, without fitting",
    )
    args = parser.parse_args()
    start = perf_counter()
    if not args.report_only:
        data = load_data()
        pretest = data.loc[data.index < TEST_START].copy()
        del data
        for name in [args.model] if args.model else NAMES:
            module_name = "lightgbm_features" if name == "lightgbm" else name
            print(f"START {name}", flush=True)
            module = importlib.import_module(f"load_bakeoff.{module_name}")
            result = module.run(pretest)
            save_result(result)
            print(f"DONE {name} seconds={result.runtime_seconds:.2f}", flush=True)
        if args.model:
            return
        write_json(
            ROOT / "artifacts" / "execution.json",
            {
                "fit_and_forecast_wall_seconds": perf_counter() - start,
                "mode": "sequential full reproduction",
            },
        )
    from load_bakeoff.report import render_report

    render_report()
    print(f"STUDY_COMPLETE command_seconds={perf_counter() - start:.2f}", flush=True)


if __name__ == "__main__":
    main()
