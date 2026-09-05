"""Write a run's pickled point (and quantile) forecast out as a CSV.

Some L3 sessions keep their test-window forecast only inside the result
pickles their own ``code/common.py`` defines (``<model>.pkl`` somewhere under
``output/``), with no forecast CSV beside ``metrics.json``. The scorer reads
CSVs, so this script loads the pickle with the run's own class definitions,
checks the series is the 168 test hours, and writes
``runs_2026_09/<run>/derived/<model>_test_forecast.csv`` (UTC timestamps,
``point_forecast`` and one column per quantile). The CSV is derived by the
orchestrator, not written by the agent; ``scoring.json`` says so where it is
used.

Usage:
    uv run python scripts/extract_pickle_forecast.py <run> <model> [--naive-tz <zone>]
    e.g.  uv run python scripts/extract_pickle_forecast.py opus5_L3_r1 lightgbm
          uv run python scripts/extract_pickle_forecast.py fable51_L3_r2 lightgbm --naive-tz UTC
The zone flag is required, and recorded in scoring.json, when the pickled
series carries a naive index (the run's own code says which zone it is).
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
    naive_tz: str | None = None
    if len(argv) == 4 and argv[2] == "--naive-tz":
        argv, naive_tz = argv[:2], argv[3]
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    run, model = argv
    code_dir = RUNS / run / "output" / "code"
    found = sorted((RUNS / run / "output").rglob(f"{model}.pkl"))
    if len(found) != 1:
        print(f"expected exactly one {model}.pkl under output/, found {found}", file=sys.stderr)
        return 1
    pkl = found[0]
    # the pickle references classes from the run's own common.py
    os.chdir(code_dir)
    sys.path.insert(0, str(code_dir))
    with pkl.open("rb") as fh:
        result = pickle.load(fh)
    # the agents name the fields differently: point_forecast / point
    point = next((getattr(result, a) for a in ("point_forecast", "point") if isinstance(getattr(result, a, None), pd.Series)), None)
    if point is None or len(point) != 168:
        print(f"{pkl}: no 168-hour point-forecast series on the object", file=sys.stderr)
        return 1
    idx = pd.DatetimeIndex(point.index)
    if idx.tz is None:
        # a naive index needs the zone the run's own code documents, passed
        # explicitly as --naive-tz <zone>; the choice is recorded in scoring.json
        if not naive_tz:
            print(f"{pkl}: naive index; refusing to guess the zone (pass --naive-tz <zone>)", file=sys.stderr)
            return 1
        idx = idx.tz_localize(naive_tz)
    frame = pd.DataFrame({"utc_timestamp": idx.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "point_forecast": point.to_numpy(dtype=float)})
    quant = next((getattr(result, a) for a in ("quantile_forecast", "quantiles")
                  if getattr(result, a, None) is not None), None)
    if isinstance(quant, pd.DataFrame) and len(quant) == 168:
        for col in quant.columns:
            frame[f"q{col}"] = quant[col].to_numpy(dtype=float)
    elif isinstance(quant, dict):
        for q, series in sorted(quant.items(), key=lambda kv: float(kv[0])):
            if isinstance(series, pd.Series) and len(series) == 168:
                frame[f"q{q}"] = series.to_numpy(dtype=float)
    out = RUNS / run / "derived" / f"{model}_test_forecast.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"wrote {out.relative_to(ROOT)}: {len(frame)} rows, {list(frame.columns)}; "
          f"first {frame['utc_timestamp'].iloc[0]} last {frame['utc_timestamp'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
