"""Write a run's pickled point (and quantile) forecast out as a CSV.

Some L3 sessions keep their test-window forecast only inside the result
pickles their own ``code/common.py`` defines (``artifacts/<model>.pkl``), with
no forecast CSV beside ``metrics.json``. The scorer reads CSVs, so this
script loads the pickle with the run's own class definitions, checks the
series is the 168 test hours, and writes
``runs_2026_09/<run>/derived/<model>_test_forecast.csv`` (UTC timestamps,
``point_forecast`` and one column per quantile). The CSV is derived by the
orchestrator, not written by the agent; ``scoring.json`` says so where it is
used.

Usage:
    uv run python scripts/extract_pickle_forecast.py <run> <model>
    e.g.  uv run python scripts/extract_pickle_forecast.py opus5_L3_r1 lightgbm
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs_2026_09"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    run, model = argv
    code_dir = RUNS / run / "output" / "code"
    pkl = RUNS / run / "output" / "artifacts" / f"{model}.pkl"
    if not pkl.exists():
        print(f"no pickle at {pkl}", file=sys.stderr)
        return 1
    # the pickle references classes from the run's own common.py
    os.chdir(code_dir)
    sys.path.insert(0, str(code_dir))
    with pkl.open("rb") as fh:
        result = pickle.load(fh)
    point = getattr(result, "point_forecast", None)
    if not isinstance(point, pd.Series) or len(point) != 168:
        print(f"{pkl}: no 168-hour point_forecast series on the object", file=sys.stderr)
        return 1
    idx = pd.DatetimeIndex(point.index)
    if idx.tz is None:
        print(f"{pkl}: naive index; refusing to guess the zone", file=sys.stderr)
        return 1
    frame = pd.DataFrame({"utc_timestamp": idx.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "point_forecast": point.to_numpy(dtype=float)})
    quant = getattr(result, "quantile_forecast", None)
    if isinstance(quant, pd.DataFrame) and len(quant) == 168:
        for col in quant.columns:
            frame[f"q{col}"] = quant[col].to_numpy(dtype=float)
    out = RUNS / run / "derived" / f"{model}_test_forecast.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out.relative_to(ROOT)}: {len(frame)} rows, {list(frame.columns)}; "
          f"first {frame['utc_timestamp'].iloc[0]} last {frame['utc_timestamp'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
