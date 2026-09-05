"""Build the deck figure and the results table for the September 2026 re-run.

Inputs (all under runs_2026_09/):
- results.csv     from `scripts/score_runs.py final` (recomputed MAPE per run)
- scoring.json    the hand-written per-run assessment (validation, intervals,
                  write-up, models fitted, headline model)
- summary.json    from `scripts/summarize_runs.py` (turns, cost, files, AGENTS.md)

Reference points come from the May 2026 runs on disk
(runs/L1/metrics.csv, runs/L2/metrics.csv, runs/L3/metrics.json).

Outputs:
- runs_2026_09/figures/exp-rerun-mape.png   test MAPE by level, every run
- runs_2026_09/RESULTS_table.md             markdown table for RESULTS.md / deck

Usage:
    uv run python scripts/build_rerun_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs_2026_09"
OUT_FIG = RUNS / "figures" / "exp-rerun-mape.png"
OUT_TABLE = RUNS / "RESULTS_table.md"

LEVELS = ["L1", "L2", "L3"]
LEVEL_LABEL = {"L1": "L1, beginner\n10 words", "L2": "L2, average\n46 words + 7-line AGENTS.md",
               "L3": "L3, research-grade\n1,673 words + 113-line AGENTS.md"}
SERIES = {
    "may2026": ("Opus 4.7, May 2026 (1 run)", "#7f7f7f", "D"),
    "fable51": ("Fable 5.1, Sep 2026 (3 runs)", "#ff7f0e", "o"),
    "opus5": ("Opus 5, Sep 2026 (1 run)", "#1f77b4", "s"),
}
X_OFFSET = {"may2026": -0.22, "fable51": 0.0, "opus5": 0.22}


def may_reference() -> dict[str, float]:
    ref: dict[str, float] = {}
    with (ROOT / "runs" / "L1" / "metrics.csv").open() as fh:
        for row in csv.DictReader(fh):
            if row[""] == "naive_y":
                ref["L1"] = float(row["MAPE_pct"])
    with (ROOT / "runs" / "L2" / "metrics.csv").open() as fh:
        for row in csv.DictReader(fh):
            if row["metric"] == "MAPE_pct":
                ref["L2"] = float(row["value"])
    l3 = json.loads((ROOT / "runs" / "L3" / "metrics.json").read_text())
    winner = l3["winner"]
    ref["L3"] = next(m["mape_test_pct"] for m in l3["models"] if m["name"] == winner)
    return ref


def load_results() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with (RUNS / "results.csv").open() as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def build_figure(results: list[dict[str, object]], ref: dict[str, float]) -> None:
    scoring = json.loads((RUNS / "scoring.json").read_text())
    fig, ax = plt.subplots(figsize=(11, 5.2))
    xs = {lvl: i for i, lvl in enumerate(LEVELS)}
    # May 2026 reference; the L3 winner fed the test week's own actual loads into
    # its 24 h lag and rolling features (DESIGN.md 5.2, correction of 2026-09-05),
    # so it is drawn hollow with a dagger
    for lvl, v in ref.items():
        lab, col, mk = SERIES["may2026"]
        day_ahead = lvl == "L3"
        ax.scatter(xs[lvl] + X_OFFSET["may2026"], v, marker=mk, s=110, zorder=3,
                   facecolors="white" if day_ahead else col, edgecolors=col, linewidths=1.8,
                   label=lab if lvl == "L1" else None)
        ax.annotate(f"{v:.2f}" + ("†" if day_ahead else ""), (xs[lvl] + X_OFFSET["may2026"], v),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=col)
    seen: set[str] = set()
    for r in results:
        tag = str(r["model_tag"])
        lvl = str(r["level"])
        if tag not in SERIES or lvl not in xs or r.get("mape_pct") in (None, ""):
            continue
        v = float(r["mape_pct"])  # type: ignore[arg-type]
        lab, col, mk = SERIES[tag]
        rep = int(r.get("rep") or 1)
        jitter = (rep - 2) * 0.06 if tag == "fable51" else 0.0
        x = xs[lvl] + X_OFFSET[tag] + jitter
        hollow = str(r.get("source")) != "recomputed"
        incomplete = bool(scoring.get(str(r["run"]), {}).get("status_note"))
        ax.scatter(x, v, marker=mk, s=110, zorder=3,
                   facecolors="white" if (hollow or incomplete) else col, edgecolors=col, linewidths=1.8,
                   label=lab if tag not in seen else None)
        seen.add(tag)
        ax.annotate(f"{v:.2f}" + ("*" if incomplete else ""), (x, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=col)
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels([LEVEL_LABEL[l] for l in LEVELS], fontsize=10)
    ax.set_ylabel("Test-window MAPE (%), lower is better")
    ax.set_title("Same three prompts, same data: May 2026 and September 2026", fontsize=13)
    ax.set_ylim(0, max([*ref.values(), *[float(r["mape_pct"]) for r in results  # type: ignore[arg-type]
                                        if r.get("mape_pct") not in (None, "")]] or [12]) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.text(0.01, 0.012,
             "Hollow marker with * = session ended before the agent's final step, scored on the forecast it had written.\n"
             "† = May's L3 winner read the test week's own actual loads through its 24 h lag (day-ahead for 6 of 7 days); "
             "the September L3 forecasts are recursive week-ahead forecasts.",
             fontsize=7.5, color="#555555", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300)
    print(f"wrote {OUT_FIG}")


def yes(v: object) -> str:
    return "yes" if v in (True, "true", "True", "yes", 1) else "no"


def build_table(results: list[dict[str, object]]) -> None:
    scoring = json.loads((RUNS / "scoring.json").read_text())
    summary = {s["run"]: s for s in json.loads((RUNS / "summary.json").read_text())}
    lines = [
        "| Run | Model | Headline model class | Test MAPE % | Test-window audit | Models fitted "
        "| Held-out validation | Intervals | Write-up | Figures | .py files | Read AGENTS.md | Turns "
        "| Wall min | Status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        run = str(r["run"])
        sc = scoring.get(run, {})
        sm = summary.get(run, {})
        mape = r.get("mape_pct")
        mape_s = "n/a" if mape in (None, "") else (f"{float(mape):.2f}" if r.get("source") == "recomputed" else f"*{float(mape):.2f}*")  # type: ignore[arg-type]
        audit = str(sc.get("test_selection", "?"))
        if audit == "test_selected":
            audit += f" ({sc.get('n_candidates', '?')})"
        agents = str(sm.get("agents_md_read", "?"))
        if agents.startswith("n/a"):
            agents = "n/a"
        status = "incomplete" if sc.get("status_note") else "complete"
        lines.append(
            f"| {run} | {sc.get('model_label', r.get('model_tag'))} | {r.get('headline_model')} | {mape_s} "
            f"| {audit} | {sc.get('n_models_fitted', '?')} | {yes(sc.get('validation'))} | {yes(sc.get('intervals'))} "
            f"| {yes(sc.get('methods_doc'))} | {int(sm.get('n_png', 0)) + int(sm.get('n_img_other', 0))} | {sm.get('n_python', '?')} "
            f"| {agents} | {sm.get('num_turns', '?')} | {round(int(sm.get('wallclock_s', 0)) / 60)} | {status} |"
        )
    OUT_TABLE.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_TABLE}")
    print("\n".join(lines))


def main() -> int:
    ref = may_reference()
    results = load_results()
    build_figure(results, ref)
    build_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
