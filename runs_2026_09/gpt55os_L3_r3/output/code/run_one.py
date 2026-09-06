from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

from common import ModelOutput, evaluate_output, json_safe, load_data, load_series, test_series, timed

ROOT = Path(".")
CODE = ROOT / "code"
ARTIFACTS = CODE / ".artifacts"


def import_model_module(name: str) -> object:
    if name != "prophet":
        return importlib.import_module(name if name != "lightgbm" else "lightgbm_features")
    local_path = CODE / "prophet.py"
    spec = importlib.util.spec_from_file_location("forecast_prophet_model", local_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {local_path}")
    module = importlib.util.module_from_spec(spec)
    saved_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if Path(p or ".").resolve() != CODE.resolve()]
        spec.loader.exec_module(module)
    finally:
        sys.path = saved_path
    return module


def series_or_nan(series: pd.Series | None, index: pd.DatetimeIndex) -> pd.Series:
    if series is None:
        return pd.Series([float("nan")] * len(index), index=index)
    return series.reindex(index)


def write_output(output: ModelOutput) -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    index = output.forecast.index
    forecast_frame = pd.DataFrame(
        {
            "timestamp": index.astype(str),
            "forecast": output.forecast.reindex(index).to_numpy(dtype=float),
            "lower_80": series_or_nan(output.lower_80, index).to_numpy(dtype=float),
            "upper_80": series_or_nan(output.upper_80, index).to_numpy(dtype=float),
            "lower_95": series_or_nan(output.lower_95, index).to_numpy(dtype=float),
            "upper_95": series_or_nan(output.upper_95, index).to_numpy(dtype=float),
            "q10": series_or_nan(output.q10, index).to_numpy(dtype=float),
            "q50": series_or_nan(output.q50, index).to_numpy(dtype=float),
            "q90": series_or_nan(output.q90, index).to_numpy(dtype=float),
        }
    )
    forecast_frame.to_csv(ARTIFACTS / f"{output.name}_forecast.csv", index=False)
    if output.feature_importance is not None:
        output.feature_importance.rename_axis("feature").reset_index(name="gain").to_csv(ARTIFACTS / "lightgbm_feature_importance.csv", index=False)
    metadata = {
        "name": output.name,
        "runtime_seconds": output.runtime_seconds,
        "hyperparameters": json_safe(output.hyperparameters),
        "notes": output.notes,
    }
    (ARTIFACTS / f"{output.name}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_one.py <model>")
    requested = sys.argv[1]
    df = load_data()
    series = load_series(df)
    actual = test_series(series)
    module = import_model_module(requested)
    print(f"RUN_START {requested}", flush=True)
    output = timed(lambda: module.run(series))
    row = evaluate_output(output, actual)
    write_output(output)
    if output.name == "prophet" and hasattr(module, "plot_prophet_decomposition"):
        module.plot_prophet_decomposition(series, output.hyperparameters, Path("figures") / "08_decomposition.png")
    print(f"RUN_DONE {output.name} mape={row.mape_test_pct:.4f} runtime={row.runtime_seconds:.1f}", flush=True)


if __name__ == "__main__":
    main()
