from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != CODE_DIR]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from code.common import DataSplits, ForecastResult, MODEL_ORDER, load_splits  # noqa: E402
from code.reporting import write_outputs  # noqa: E402

ModelRunner = Callable[[DataSplits], ForecastResult]


def _cache_signature(model_file: Path, csv_path: Path) -> str:
    digest = hashlib.sha256()
    for path in (CODE_DIR / "common.py", model_file, csv_path):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_or_run(
    name: str,
    runner: ModelRunner,
    model_file: Path,
    splits: DataSplits,
    csv_path: Path,
    cache_dir: Path,
    force: bool,
    use_cache: bool,
) -> ForecastResult:
    cache_path = cache_dir / f"{name}.pkl"
    signature = _cache_signature(model_file, csv_path)
    if use_cache and not force and cache_path.exists():
        with cache_path.open("rb") as handle:
            cached = pickle.load(handle)
        if cached.get("signature") == signature:
            result = cached["result"]
            print(
                f"[{name}] loaded validated cache, runtime {result.runtime_seconds:.1f} s",
                flush=True,
            )
            return result

    print(f"[{name}] fitting and forecasting", flush=True)
    result = runner(splits)
    if use_cache:
        cache_dir.mkdir(exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump({"signature": signature, "result": result}, handle)
    print(f"[{name}] complete in {result.runtime_seconds:.1f} s", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the six-model German load forecasting bake-off"
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore model caches and refit every model"
    )
    parser.add_argument("--no-cache", action="store_true", help="Do not read or write model caches")
    arguments = parser.parse_args()

    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    from code.lightgbm_features import run as run_lightgbm
    from code.naive import run as run_naive
    from code.nbeats import run as run_nbeats
    from code.patchtst import run as run_patchtst
    from code.prophet import run as run_prophet
    from code.sarima import run as run_sarima

    csv_path = ROOT / "opsd_de_load.csv"
    splits = load_splits(csv_path)
    runners: dict[str, tuple[ModelRunner, Path]] = {
        "naive": (run_naive, CODE_DIR / "naive.py"),
        "sarima": (run_sarima, CODE_DIR / "sarima.py"),
        "prophet": (run_prophet, CODE_DIR / "prophet.py"),
        "lightgbm": (run_lightgbm, CODE_DIR / "lightgbm_features.py"),
        "nbeats": (run_nbeats, CODE_DIR / "nbeats.py"),
        "patchtst": (run_patchtst, CODE_DIR / "patchtst.py"),
    }
    cache_dir = ROOT / ".forecast_cache"
    results: list[ForecastResult] = []
    for name in MODEL_ORDER:
        runner, model_file = runners[name]
        results.append(
            _load_or_run(
                name,
                runner,
                model_file,
                splits,
                csv_path,
                cache_dir,
                arguments.force,
                not arguments.no_cache,
            )
        )

    total_runtime_seconds = sum(result.runtime_seconds for result in results)
    payload = write_outputs(ROOT, splits, results, total_runtime_seconds)
    print(
        f"Study complete: winner={payload['winner']}, "
        f"total_runtime={payload['total_runtime_seconds']:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
