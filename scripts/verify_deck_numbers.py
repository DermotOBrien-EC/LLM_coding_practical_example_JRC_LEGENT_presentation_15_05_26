"""Cross-check the re-run slides in the seminar deck against the run results.

Reads the seminar deck's ``index.qmd`` (path given on the command line) and
compares every number and count it quotes about the September 2026 re-run
with ``runs_2026_09/results.csv``, ``scoring.json`` and ``summary.json``.
Prints one OK/FAIL line per check and exits non-zero on any FAIL, or if a
``L3PENDING`` placeholder is still in the deck.

Usage:
    uv run python scripts/verify_deck_numbers.py /path/to/ai_seminar_jrc/index.qmd
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs_2026_09"

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, ok, detail))


def load() -> tuple[dict[str, dict[str, str]], dict[str, dict], dict[str, dict]]:
    results = {r["run"]: r for r in csv.DictReader((RUNS / "results.csv").open())}
    scoring = json.loads((RUNS / "scoring.json").read_text())
    summary = {s["run"]: s for s in json.loads((RUNS / "summary.json").read_text())}
    return results, scoring, summary


def cell(results: dict[str, dict[str, str]], tag: str, level: str) -> list[float]:
    vals = []
    for run, r in results.items():
        if r["model_tag"] == tag and r["level"] == level and r.get("mape_pct") not in (None, ""):
            vals.append(float(r["mape_pct"]))
    return sorted(vals)


def fmt1(v: float) -> str:
    return f"{v:.1f}"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    deck = Path(argv[0]).read_text(encoding="utf-8")
    results, scoring, summary = load()

    check("no L3PENDING placeholder left", "L3PENDING" not in deck,
          f"{deck.count('L3PENDING')} left")

    # Accuracy slide: a range per Fable cell that has three scored runs; a
    # cell with none must not be quoted as a September result at all.
    for level in ("L1", "L2", "L3"):
        vals = cell(results, "fable51", level)
        if len(vals) == 3:
            expect = f"{fmt1(vals[0])} to {fmt1(vals[-1])}"
            check(f"Fable {level} range quoted as '{expect}'", expect in deck, f"values {vals}")
        elif not vals:
            check(f"Fable {level} has no scored run, so the deck quotes no range for it",
                  f"On {level}: **" not in deck, "deck still quotes a range")
        else:
            check(f"Fable {level} has 3 or 0 scored runs", False, f"found {len(vals)}")

    # L1 example slide: run 2 of 3 MAPE 3.3%
    r = results.get("fable51_L1_r2")
    if r:
        expect = f"MAPE {fmt1(float(r['mape_pct']))}%"
        check(f"L1 example slide quotes '{expect}'", expect in deck)

    # Discipline table: counts per Fable cell
    def count(level: str, key: str, want: object = True) -> int:
        n = 0
        for run, sc in scoring.items():
            if sc.get("level") == level and sc.get("model_tag") == "fable51":
                if key == "leaked":
                    # the deck column is "target week entered the headline"; a run
                    # whose leak is confined to a labelled supplement counts as clean
                    n += (sc.get("test_selection") == "leaked"
                          and not str(sc.get("leak_scope", "")).startswith("supplement"))
                elif key == "finished":
                    n += not sc.get("status_note")
                else:
                    n += bool(sc.get(key)) == want
        return n

    # A cell is reportable only if it has scored runs.
    scored_levels = [lvl for lvl in ("L1", "L2", "L3") if cell(results, "fable51", lvl)]
    for level in ("L1", "L2", "L3"):
        if level not in scored_levels:
            check(f"no Fable {level} row in the discipline table (no scored run)",
                  f"| Fable 5.1, {level} (" not in deck)
            continue
        expect_row = (f"| Fable 5.1, {level} (3) | {count(level, 'validation')} of 3 | {count(level, 'intervals')} of 3 "
                      f"| {count(level, 'methods_doc')} of 3 | {count(level, 'leaked')} of 3 | {count(level, 'finished')} of 3 |")
        check(f"discipline table row Fable {level}", expect_row in deck, expect_row)

    def yn(v: object) -> str:
        return "yes" if v else "no"

    opus_levels = [lvl for lvl in ("L1", "L2", "L3")
                   if next((s for s in scoring.values()
                            if s.get("model_tag") == "opus5" and s.get("level") == lvl
                            and results.get(f"opus5_{lvl}_r1", {}).get("mape_pct") not in (None, "")), None)]
    o = [next(s for s in scoring.values() if s.get("model_tag") == "opus5" and s.get("level") == lvl)
         for lvl in opus_levels]
    if o:
        label = " / ".join(opus_levels)
        expect_row = (f"| Opus 5, {label} | " + " / ".join(yn(s.get("validation")) for s in o) + " | "
                      + " / ".join(yn(s.get("intervals")) for s in o) + " | "
                      + " / ".join(yn(s.get("methods_doc")) for s in o) + " | "
                      + " / ".join(yn(s.get("test_selection") == "leaked") for s in o) + " | "
                      + " / ".join(yn(not s.get("status_note")) for s in o) + " |")
        check("discipline table row Opus 5", expect_row in deck, expect_row)

    # Wall-clock claims on the card: Opus minutes and Fable range
    mins = {run: round(int(s.get("wallclock_s", 0)) / 60) for run, s in summary.items()}
    fable_l12 = sorted(m for run, m in mins.items() if run.startswith("fable51_L1") or run.startswith("fable51_L2"))
    if fable_l12:
        check(f"Fable L1/L2 wall-clock range '{fable_l12[0]} to {fable_l12[-1]}' quoted",
              f"{fable_l12[0]} to {fable_l12[-1]}" in deck, str(fable_l12))
    if "opus5_L1_r1" in mins and "opus5_L2_r1" in mins:
        a, b = sorted([mins["opus5_L2_r1"], mins["opus5_L1_r1"]])
        check(f"Opus wall-clock '{a} and {b} minutes' quoted", f"{a} and {b} minutes" in deck)

    # AGENTS.md read in every L2 run
    l2 = [s for run, s in summary.items() if run.endswith("_L2_r1") or run.endswith("_L2_r2") or run.endswith("_L2_r3")]
    check("every L2 run read AGENTS.md (summary)", all(s.get("agents_md_read") == "yes" for s in l2),
          str({s['run']: s.get('agents_md_read') for s in l2}))

    # Figures referenced exist
    for fig in re.findall(r"!\[\]\((figures/exp-rerun-[^)]+)\)", deck):
        check(f"figure exists {fig}", (Path(argv[0]).parent / fig).exists())

    # No em-dashes in the re-run slides (author rule)
    m = re.search(r"## Four months later.*?## Seven tips for working with AI \(cont\.\)", deck, re.S)
    check("no em-dash in the re-run slides", bool(m) and "—" not in m.group(0))

    bad = 0
    for name, ok, detail in checks:
        print(f"[{'OK ' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
        bad += not ok
    print(f"{len(checks) - bad} OK, {bad} FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
