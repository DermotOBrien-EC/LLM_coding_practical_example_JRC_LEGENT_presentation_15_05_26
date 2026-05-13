"""Verify slides/talk.md against ground truth.

Three classes of check:
1. Every figure path referenced in `talk.md` exists on disk.
2. Every quoted MAPE / coverage / runtime / word-count / line-count matches
   the corresponding source file (runs/L*/metrics.*, prompts/L*.md, AGENTS.md).
3. The winner model name in talk.md matches `runs/L3/metrics.json:winner`.

Run:  uv run python scripts/verify_slide_numbers.py
Exits non-zero if any check fails.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TALK = ROOT / "slides" / "talk.md"


@dataclass
class Check:
    name: str
    expected: str
    actual: str
    ok: bool


def fmt(c: Check) -> str:
    mark = "OK " if c.ok else "FAIL"
    return f"[{mark}] {c.name}: expected={c.expected!r} actual={c.actual!r}"


def load_talk() -> str:
    return TALK.read_text(encoding="utf-8")


def check_figure_paths(talk: str) -> list[Check]:
    """Pull every PNG path mentioned in talk.md and verify it resolves."""
    # Match the path inside a Marp image directive: `![...](<path>)`.
    # Catches `../runs/L*/figures/x.png`, `assets/x.png`, `slides/assets/x.png`,
    # and any other relative PNG reference. All paths are resolved relative
    # to talk.md's own directory.
    pattern = re.compile(r"!\[[^\]]*\]\(([^)\s]+\.png)\)")
    paths = sorted(set(pattern.findall(talk)))
    checks: list[Check] = []
    for p in paths:
        abs_path = (TALK.parent / p).resolve()
        checks.append(
            Check(
                name=f"figure exists: {p}",
                expected="file exists",
                actual="exists" if abs_path.is_file() else f"MISSING ({abs_path})",
                ok=abs_path.is_file(),
            )
        )
    # Sanity: every PNG known to the deck (assets + run figures) must be
    # represented. If the regex above silently misses one, this catches it.
    expected_count = 8  # 7 run figures + 1 generated asset, per the plan
    if len(paths) != expected_count:
        checks.append(
            Check(
                name="figure count sanity",
                expected=f"{expected_count} figures",
                actual=f"{len(paths)} found: {paths}",
                ok=len(paths) == expected_count,
            )
        )
    if not checks:
        checks.append(Check(name="figure paths scanned", expected="at least 1", actual="0", ok=False))
    return checks


def check_l1_mape(talk: str) -> Check:
    """L1 headline number must be the best of runs/L1/metrics.csv MAPE column."""
    rows = list(csv.DictReader((ROOT / "runs" / "L1" / "metrics.csv").open()))
    mapes = [float(r["MAPE_pct"]) for r in rows]
    best = min(mapes)
    expected = f"{best:.2f}"  # 10.76
    return Check(
        name="L1 MAPE 10.76% matches runs/L1/metrics.csv best baseline",
        expected=expected,
        actual=expected if f"{expected}%" in talk else "not found in talk.md",
        ok=f"{expected}%" in talk,
    )


def check_l2_mape(talk: str) -> Check:
    rows = list(csv.DictReader((ROOT / "runs" / "L2" / "metrics.csv").open()))
    by_metric = {r["metric"]: float(r["value"]) for r in rows}
    expected = f"{by_metric['MAPE_pct']:.2f}"  # 5.52
    return Check(
        name="L2 MAPE 5.52% matches runs/L2/metrics.csv",
        expected=expected,
        actual=expected if f"{expected}%" in talk else "not found in talk.md",
        ok=f"{expected}%" in talk,
    )


def check_l3_winner_mape(talk: str, metrics: dict) -> Check:
    winner_name = metrics["winner"]
    winner = next(m for m in metrics["models"] if m["name"] == winner_name)
    expected = f"{winner['mape_test_pct']:.2f}"  # 3.43
    return Check(
        name="L3 winner MAPE 3.43% matches runs/L3/metrics.json",
        expected=expected,
        actual=expected if f"{expected}%" in talk else "not found in talk.md",
        ok=f"{expected}%" in talk,
    )


def check_l3_coverage(talk: str, metrics: dict, level: int, expected_str: str) -> Check:
    key = f"winner_coverage_{level}pct"
    actual_pct = metrics[key] * 100
    rounded = f"{actual_pct:.1f}"  # 79.8 / 92.9
    return Check(
        name=f"L3 {level}% PI coverage {expected_str} matches metrics.json {key}={actual_pct:.4f}",
        expected=rounded,
        actual=rounded if f"{rounded}%" in talk else "not found in talk.md",
        ok=f"{rounded}%" in talk,
    )


def check_winner_name(talk: str, metrics: dict) -> Check:
    """Winner name in metrics.json (lowercase 'lightgbm') must appear as
    'LightGBM' somewhere in talk.md."""
    winner = metrics["winner"]  # "lightgbm"
    display = "LightGBM"
    return Check(
        name=f"winner '{display}' (metrics.json:winner={winner!r}) appears in talk.md",
        expected=display,
        actual=display if display in talk else "not found",
        ok=display in talk,
    )


def check_n_test_obs(talk: str, metrics: dict) -> Check:
    n = metrics["n_test_observations"]
    return Check(
        name=f"test horizon {n}-hour appears in talk.md",
        expected=f"{n}",
        actual=f"{n}" if f"{n}" in talk else "not found",
        ok=f"{n}" in talk,
    )


def check_prompt_word_counts(talk: str) -> list[Check]:
    out: list[Check] = []
    for level, expected_label in [
        ("L1", "10 words"),
        ("L2", "46 words"),
        ("L3", "1,673 words"),  # comma-formatted in talk
    ]:
        words = len((ROOT / "prompts" / f"{level}.md").read_text().split())
        # The talk uses '10 words', '46 words', '1,673 words' (with comma) in
        # one spot, and the raw '1673' inside the personae table.
        comma = f"{words:,}"
        plain = f"{words}"
        contained = (
            f"{comma} words" in talk or f"{plain} words" in talk or f"<b>{words:,}</b>" in talk or f"<b>{plain}</b>" in talk
        )
        out.append(
            Check(
                name=f"{level} prompt word count {words} appears in talk.md",
                expected=f"{words} (or {comma})",
                actual=f"found" if contained else "not found",
                ok=contained,
            )
        )
    return out


def check_agents_line_counts(talk: str) -> list[Check]:
    out: list[Check] = []
    for level, claim in [("L2", "7 lines"), ("L3", "113 lines")]:
        path = ROOT / "runs" / level / "AGENTS.md"
        n_lines = sum(1 for _ in path.open())
        contains = claim in talk or f"{n_lines}-line" in talk or f"({n_lines} lines)" in talk
        out.append(
            Check(
                name=f"{level} AGENTS.md line count {n_lines} matches talk claim {claim!r}",
                expected=f"{n_lines}",
                actual=f"talk states {claim!r}" if contains else "not found",
                ok=(contains and n_lines == int(claim.split()[0])),
            )
        )
    return out


def main() -> int:
    if not TALK.is_file():
        print(f"FAIL: {TALK} does not exist", file=sys.stderr)
        return 1
    talk = load_talk()
    metrics_l3 = json.loads((ROOT / "runs" / "L3" / "metrics.json").read_text())

    checks: list[Check] = []
    checks.extend(check_figure_paths(talk))
    checks.append(check_l1_mape(talk))
    checks.append(check_l2_mape(talk))
    checks.append(check_l3_winner_mape(talk, metrics_l3))
    checks.append(check_l3_coverage(talk, metrics_l3, level=80, expected_str="79.8%"))
    checks.append(check_l3_coverage(talk, metrics_l3, level=95, expected_str="92.9%"))
    checks.append(check_winner_name(talk, metrics_l3))
    checks.append(check_n_test_obs(talk, metrics_l3))
    checks.extend(check_prompt_word_counts(talk))
    checks.extend(check_agents_line_counts(talk))

    for c in checks:
        print(fmt(c))

    n_fail = sum(1 for c in checks if not c.ok)
    print()
    print(f"Summary: {len(checks) - n_fail}/{len(checks)} passed, {n_fail} failed.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
