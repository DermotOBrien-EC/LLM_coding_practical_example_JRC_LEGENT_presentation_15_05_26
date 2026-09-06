"""Recompute the test-window accuracy of every run from what it wrote to disk.

Two modes.

``discover`` (default): for every run directory under ``runs_2026_09/`` list
every CSV column that can be scored against the 168 held-out hours
(2020-01-01 00:00 to 2020-01-07 23:00 UTC) and print its MAPE / RMSE / MAE.
Also reads an L3-style ``metrics.json`` if present. Writes
``runs_2026_09/score_candidates.json``. Nothing is chosen here.

``final``: read ``runs_2026_09/scoring.json``, a hand-written file that names,
per run, which CSV and column is the agent's headline forecast (with a
reason), recompute that column's MAPE, and write ``runs_2026_09/results.csv``
with one row per run. If a run has no scorable forecast the row carries the
agent-reported number and ``source = agent_reported``.

Usage:
    uv run python scripts/score_runs.py            # discover
    uv run python scripts/score_runs.py final      # apply scoring.json
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs_2026_09"
DATA = ROOT / "data" / "opsd_de_load.csv"
TEST_START = pd.Timestamp("2020-01-01 00:00", tz="UTC")
TEST_END = pd.Timestamp("2020-01-07 23:00", tz="UTC")
SKIP_DIRS = {".claude", "__pycache__", "lightning_logs", "darts_logs", "cache", ".venv"}
# Extension B agents built their own environments in the working directory; a
# CSV inside a downloaded package is not a forecast (DESIGN.md section 12).
VENDOR_DIR_RE = re.compile(r"^\.?[\w-]*venv[\w.-]*$", re.I)
# Only downloaded environments and package trees; tool caches such as
# .mypy_cache stay in the manifest, as they were for every earlier run.
VENDOR_DIR_NAMES = {"site-packages", ".packages", ".python_packages", ".uvcache",
                    ".uv-cache", ".uv_cache", ".conda"}


def actuals() -> pd.Series:
    """The 168 UTC test hours (the reference window)."""
    s = full_series().loc[TEST_START:TEST_END]
    assert len(s) == 168, len(s)
    return s


_FULL: dict[str, pd.Series] = {}


def full_series() -> pd.Series:
    """The whole hourly series, UTC-indexed, so a forecast stamped in local
    time (CET) can be scored on the hours it was actually made for."""
    if "s" not in _FULL:
        df = pd.read_csv(DATA, parse_dates=["utc_timestamp"])
        s = df.set_index("utc_timestamp")["DE_load_actual_entsoe_transparency"]
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        _FULL["s"] = s
    return _FULL["s"]


def metrics(y: np.ndarray, f: np.ndarray) -> dict[str, float]:
    err = y - f
    return {
        "mape_pct": float(np.mean(np.abs(err) / y) * 100.0),
        "rmse_mw": float(np.sqrt(np.mean(err**2))),
        "mae_mw": float(np.mean(np.abs(err))),
    }


def _rel(path: Path) -> str:
    """Path relative to the runs folder when inside it, else to the repo root."""
    for base in (RUNS, ROOT):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def _find_time_index(df: pd.DataFrame, naive_tz: str = "UTC",
                     time_col: str | None = None) -> pd.DatetimeIndex | None:
    """Return a UTC DatetimeIndex aligned to df's rows, or None.

    Timestamps that carry an offset are converted to UTC. Naive timestamps
    are interpreted in ``naive_tz`` (UTC by default; ``scoring.json`` can
    override per run when the agent documented local time). Without
    ``time_col`` the first column that parses is used; a file that carries
    both a local and a UTC stamp (Extension C, opus5 L1 r3) names the one
    the agent meant in ``scoring.json``.
    """
    candidates = list(df.columns)
    if df.index.name is not None:
        df = df.reset_index()
        candidates = list(df.columns)
    if time_col is not None:
        if time_col not in candidates:
            return None
        candidates = [time_col]
    lo, hi = TEST_START - pd.Timedelta(hours=48), TEST_END + pd.Timedelta(hours=48)
    for col in candidates:
        if not (df[col].dtype == object or "datetime" in str(df[col].dtype)):
            continue
        try:
            raw = pd.to_datetime(df[col], errors="coerce")
        except (ValueError, TypeError):
            continue
        if raw.notna().sum() < 168:
            continue
        try:
            if getattr(raw.dt, "tz", None) is None:
                ts = raw.dt.tz_localize(naive_tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
            else:
                ts = raw.dt.tz_convert("UTC")
        except (TypeError, ValueError, AttributeError):
            try:
                ts = pd.to_datetime(df[col], utc=True, errors="coerce")
            except (ValueError, TypeError):
                continue
        idx = pd.DatetimeIndex(ts)
        if ((idx >= lo) & (idx <= hi)).sum() >= 168:
            return idx
    return None


def score_csv(path: Path, y: pd.Series, base: Path | None = None,
              naive_tz: str = "UTC", time_col: str | None = None) -> list[dict[str, object]]:
    """Score every numeric column of one CSV against the actual load.

    With a usable timestamp column the forecast is scored on the 168 hours
    it is stamped with (converted to UTC), which may be the UTC test window
    or a local-time week one hour off; the candidate records which. Without
    timestamps a 168-row file is assumed to be the UTC window in order.
    """
    out: list[dict[str, object]] = []
    label = str(path.relative_to(base)) if base and base in path.parents else _rel(path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - any unreadable CSV is simply skipped
        return [{"file": label, "error": f"{type(exc).__name__}: {exc}"}]
    if df.empty:
        return out
    full = full_series()
    idx = _find_time_index(df, naive_tz, time_col)
    if idx is not None:
        df = df.copy()
        df.index = idx
        lo, hi = TEST_START - pd.Timedelta(hours=48), TEST_END + pd.Timedelta(hours=48)
        near = df.loc[(df.index >= lo) & (df.index <= hi)]
        near = near[~near.index.duplicated(keep="first")].sort_index()
        # the forecast's own window: the first 168 stamped hours that
        # overlap the test week
        window = near.iloc[:168] if len(near) >= 168 else near
        if len(window) != 168:
            return [{"file": label, "note": f"{len(window)} stamped rows near the test window"}]
        truth = full.reindex(window.index)
        if truth.isna().any():
            return [{"file": label, "note": "stamped hours not all present in the data"}]
        shift_h = int((window.index[0] - TEST_START) / pd.Timedelta(hours=1))
        alignment = "timestamp" if shift_h == 0 else f"timestamp (window shifted {shift_h:+d} h from UTC week)"
    elif len(df) == 168:
        window = df.copy()
        window.index = y.index
        truth = y
        alignment = "row_order_168"
    else:
        return out
    ytrue = truth.to_numpy(dtype=float)
    for col in window.columns:
        vals = pd.to_numeric(window[col], errors="coerce")
        if vals.notna().sum() != 168:
            continue
        arr = vals.to_numpy(dtype=float)
        if np.allclose(arr, ytrue, atol=1e-6):
            out.append({"file": label, "column": col,
                        "is_actual": True, "alignment": alignment})
            continue
        if np.nanmax(np.abs(arr)) < 1000:  # not a load in MW; a flag or a small feature
            continue
        m = metrics(ytrue, arr)
        out.append({"file": label, "column": col,
                    "is_actual": False, "alignment": alignment, **m})
    return out


def discover() -> int:
    y = actuals()
    report: dict[str, dict[str, object]] = {}
    for run_dir in sorted(p for p in RUNS.iterdir()
                          if p.is_dir() and not p.name.startswith("_") and (p / "run_meta.json").exists()):
        entry: dict[str, object] = {"candidates": [], "metrics_json": None}
        out_dir = run_dir / "output"
        if not out_dir.exists():
            report[run_dir.name] = {"candidates": [], "metrics_json": None, "note": "no output/ dir"}
            continue
        for csv_path in sorted(run_dir.rglob("*.csv")):
            rel_parts = csv_path.relative_to(run_dir).parts
            if rel_parts[0] not in ("output", "output_outside"):
                continue
            if any(part in SKIP_DIRS for part in rel_parts) or csv_path.name == "opsd_de_load.csv":
                continue
            if any(VENDOR_DIR_RE.match(part) or part in VENDOR_DIR_NAMES for part in rel_parts):
                continue
            entry["candidates"].extend(score_csv(csv_path, y, run_dir))  # type: ignore[union-attr]
        mj = out_dir / "metrics.json"
        if mj.exists():
            try:
                data = json.loads(mj.read_text())
                if isinstance(data, dict) and "models" in data:
                    entry["metrics_json"] = {
                        "winner": data.get("winner"),
                        "models": {m.get("name"): m.get("mape_test_pct") for m in data["models"]},
                        "coverage_80": data.get("winner_coverage_80pct"),
                        "coverage_95": data.get("winner_coverage_95pct"),
                    }
                else:
                    entry["metrics_json"] = {"raw": data}
            except Exception as exc:  # noqa: BLE001
                entry["metrics_json"] = {"error": str(exc)}
        report[run_dir.name] = entry
    (RUNS / "score_candidates.json").write_text(json.dumps(report, indent=2))
    for run, entry in report.items():
        print(f"== {run}")
        if entry["metrics_json"]:
            print("   metrics.json:", json.dumps(entry["metrics_json"]))
        for c in entry["candidates"]:  # type: ignore[union-attr]
            if c.get("is_actual"):
                print(f"   [actual] {c['file']}::{c['column']}")
            elif "mape_pct" in c:
                print(f"   {c['file']}::{c['column']}  MAPE {c['mape_pct']:.3f}  RMSE {c['rmse_mw']:.0f}  MAE {c['mae_mw']:.0f}  ({c['alignment']})")
            else:
                print("   ", c)
    return 0


def final() -> int:
    y = actuals()
    decisions = json.loads((RUNS / "scoring.json").read_text())
    rows: list[dict[str, object]] = []
    for run, d in decisions.items():
        row: dict[str, object] = {"run": run, "model_tag": d.get("model_tag"), "level": d.get("level"),
                                  "rep": d.get("rep"), "headline_model": d.get("headline_model"),
                                  "agent_reported_mape": d.get("agent_reported_mape"),
                                  "reason": d.get("reason")}
        if d.get("file") and d.get("column"):
            cands = score_csv(RUNS / run / d["file"], y, RUNS / run, d.get("naive_tz", "UTC"),
                              d.get("time_col"))
            match = [c for c in cands if c.get("column") == d["column"] and "mape_pct" in c]
            if not match:
                raise SystemExit(f"{run}: column {d['column']} in {d['file']} is not scorable")
            row.update({"mape_pct": round(match[0]["mape_pct"], 3),  # type: ignore[index]
                        "rmse_mw": round(match[0]["rmse_mw"], 1),  # type: ignore[index]
                        "mae_mw": round(match[0]["mae_mw"], 1),  # type: ignore[index]
                        "alignment": match[0]["alignment"],
                        "source": "recomputed"})
        else:
            row.update({"mape_pct": d.get("agent_reported_mape"), "rmse_mw": None, "mae_mw": None,
                        "source": "agent_reported" if d.get("agent_reported_mape") is not None else "none"})
        rows.append(row)
    cols = ["run", "model_tag", "level", "rep", "headline_model", "mape_pct", "rmse_mw", "mae_mw",
            "agent_reported_mape", "source", "alignment", "reason"]
    with (RUNS / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    for r in rows:
        print(f"{r['run']:<22} {str(r['headline_model']):<28} MAPE {r['mape_pct']}  ({r['source']})")
    return 0


if __name__ == "__main__":
    sys.exit(final() if len(sys.argv) > 1 and sys.argv[1] == "final" else discover())
