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
- runs_2026_09/figures/exp-extension-outcomes.png  how each session ended (forecast / question / plan)
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
OUT_FIG_OPENAI = RUNS / "figures" / "exp-openai-mape.png"
OUT_TABLE = RUNS / "RESULTS_table.md"

LEVELS = ["L1", "L2", "L3"]
LEVEL_LABEL = {"L1": "L1, beginner\n10 words", "L2": "L2, average\n46 words + 7-line AGENTS.md",
               "L3": "L3, research-grade\n1,673 words + 113-line AGENTS.md"}
SERIES = {
    "may2026": ("Opus 4.7, May 2026 (1 run)", "#7f7f7f", "D"),
    # the Claude arm (sections 1 to 7) and Extension C (section 13)
    "opus47": ("Opus 4.7, Sep 2026", "#d62728", "*"),
    "opus48": ("Opus 4.8, Sep 2026", "#17becf", "X"),
    "opus5": ("Opus 5, Sep 2026", "#1f77b4", "s"),
    "fable51": ("Fable 5.1, Sep 2026", "#ff7f0e", "o"),
    # Extension A (section 11): the OpenAI models on the same harness
    "astra": ("GPT-6-Astra, frozen prompt", "#2ca02c", "^"),
    "sol": ("GPT-5.6-Sol, frozen prompt", "#9467bd", "v"),
    "gpt55": ("GPT-5.5, frozen prompt", "#8c564b", "P"),
    # Extension B (section 12): the same prompt plus the one-shot note
    "astraos": ("GPT-6-Astra, + one-shot note", "#2ca02c", "^"),
    "solos": ("GPT-5.6-Sol, + one-shot note", "#9467bd", "v"),
    "gpt55os": ("GPT-5.5, + one-shot note", "#8c564b", "P"),
}
# one figure per arm, so a panel never carries more than five series
CLAUDE_TAGS = ["may2026", "opus47", "opus48", "opus5", "fable51"]
OPENAI_TAGS = ["astra", "astraos", "sol", "solos", "gpt55", "gpt55os"]
# Extension B is drawn open-faced so the frozen-prompt and one-shot runs of the
# same model are told apart at a glance
ONESHOT_TAGS = {"astraos", "solos", "gpt55os"}
THREE_RUN_TAGS = {"fable51", "astra", "sol", "gpt55", "astraos", "solos", "gpt55os",
                  "opus47", "opus48", "opus5"}


def offsets(tags: list[str]) -> dict[str, float]:
    """Spread the series of one panel evenly around each level's tick."""
    n = len(tags)
    span = 0.74
    step = span / max(n - 1, 1)
    return {t: -span / 2 + i * step for i, t in enumerate(tags)}


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


def build_figure(results: list[dict[str, object]], ref: dict[str, float], tags: list[str],
                 out: Path, title: str, footnote: str) -> None:
    scoring = json.loads((RUNS / "scoring.json").read_text())
    X_OFFSET = offsets(tags)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    xs = {lvl: i for i, lvl in enumerate(LEVELS)}
    # May 2026 reference; the L3 winner fed the test week's own actual loads into
    # its 24 h lag and rolling features (DESIGN.md 5.2, correction of 2026-09-05),
    # so it is drawn hollow with a dagger
    for lvl, v in (ref.items() if "may2026" in tags else []):
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
        if tag not in tags or lvl not in xs or r.get("mape_pct") in (None, ""):
            continue
        v = float(r["mape_pct"])  # type: ignore[arg-type]
        lab, col, mk = SERIES[tag]
        rep = int(r.get("rep") or 1)
        jitter = (rep - 2) * 0.035 if tag in THREE_RUN_TAGS else 0.0
        x = xs[lvl] + X_OFFSET[tag] + jitter
        sc = scoring.get(str(r["run"]), {})
        hollow = str(r.get("source")) != "recomputed"
        incomplete = bool(sc.get("status_note"))
        # a headline that read the test week's own loads (DESIGN.md 5.2) is drawn like May's L3
        leaked = sc.get("test_selection") == "leaked" and str(sc.get("leak_scope", "")).startswith("headline")
        open_faced = hollow or incomplete or leaked or tag in ONESHOT_TAGS
        ax.scatter(x, v, marker=mk, s=110, zorder=3,
                   facecolors="white" if open_faced else col, edgecolors=col, linewidths=1.8,
                   label=lab if tag not in seen else None)
        seen.add(tag)
        ax.annotate(f"{v:.2f}" + ("*" if incomplete else "") + ("†" if leaked else ""), (x, v),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=col)
    ax.set_xticks(list(xs.values()))
    ax.set_xticklabels([LEVEL_LABEL[l] for l in LEVELS], fontsize=10)
    ax.set_ylabel("Test-window MAPE (%), lower is better")
    ax.set_title(title, fontsize=13)
    shown = [float(r["mape_pct"]) for r in results  # type: ignore[arg-type]
             if str(r["model_tag"]) in tags and r.get("mape_pct") not in (None, "")]
    ax.set_ylim(0, max([*(ref.values() if "may2026" in tags else []), *shown] or [12]) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, ncol=2 if len(tags) > 5 else 1)
    fig.text(0.01, 0.012, footnote, fontsize=7.5, color="#555555", va="bottom")
    fig.tight_layout(rect=(0, 0.02 + 0.023 * footnote.count("\n"), 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


OUT_FIG2 = RUNS / "figures" / "exp-extension-outcomes.png"
OUTCOME_ORDER = ["fable51", "opus5", "astra", "sol", "gpt55"]


def outcome_of(run: str, scoring: dict[str, dict[str, object]], results: dict[str, dict[str, object]]) -> str:
    sc = scoring.get(run, {})
    head = str(sc.get("headline_model", ""))
    if head.startswith("none (asked"):
        return "asked"
    if head.startswith("none (presented"):
        return "plan"
    if sc.get("status_note"):
        return "incomplete"
    if str(results.get(run, {}).get("source")) == "agent_reported":
        return "reported"
    return "forecast"


def build_outcomes_figure(results: list[dict[str, object]]) -> None:
    """One cell per run: did the session end with a forecast, a question, or a plan?"""
    scoring = json.loads((RUNS / "scoring.json").read_text())
    by_run = {str(r["run"]): r for r in results}
    colours = {"forecast": "#2b8a3e", "incomplete": "#94d82d", "reported": "#1c7ed6",
               "asked": "#e8590c", "plan": "#fab005"}
    labels = {"forecast": "wrote a forecast file", "incomplete": "forecast file written, session ended early",
              "reported": "finished; accuracy as the agent reported it (no forecast file)",
              "asked": "asked a clarifying question and stopped", "plan": "presented a plan and stopped"}
    rows = [t for t in OUTCOME_ORDER if any(str(r["model_tag"]) == t for r in results)]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    for yi, tag in enumerate(rows):
        for xi, lvl in enumerate(LEVELS):
            reps = sorted(int(r.get("rep") or 1) for r in results if str(r["model_tag"]) == tag and str(r["level"]) == lvl)
            for k, rep in enumerate(reps):
                run = f"{tag}_{lvl}_r{rep}"
                oc = outcome_of(run, scoring, by_run)
                x = xi * 3.6 + k
                ax.add_patch(plt.Rectangle((x, yi), 0.9, 0.9, color=colours[oc]))
                mape = by_run[run].get("mape_pct")
                txt = f"{float(mape):.1f}" if mape not in (None, "") else ("?" if oc == "asked" else "plan")  # type: ignore[arg-type]
                if oc == "forecast" or oc == "incomplete":
                    sc = scoring.get(run, {})
                    if sc.get("test_selection") == "leaked" and str(sc.get("leak_scope", "")).startswith("headline"):
                        txt += "†"
                ax.text(x + 0.45, yi + 0.45, txt, ha="center", va="center", fontsize=9,
                        color="white" if oc in ("forecast", "asked", "reported") else "#333333",
                        fontstyle="italic" if oc == "reported" else "normal")
    ax.set_xlim(-0.3, 3 * 3.6 - 0.4)
    ax.set_ylim(-0.2, len(rows) + 0.1)
    ax.set_xticks([xi * 3.6 + 1.4 for xi in range(3)])
    ax.set_xticklabels([LEVEL_LABEL[l] for l in LEVELS], fontsize=9)
    ax.set_yticks([yi + 0.45 for yi in range(len(rows))])
    ax.set_yticklabels([SERIES[t][0].split(",")[0] for t in rows], fontsize=9)
    ax.invert_yaxis()
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    order = ("forecast", "incomplete", "reported", "asked", "plan")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[k]) for k in order]
    ax.legend(handles, [labels[k] for k in order], loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8, frameon=False)
    ax.set_title("How each headless session ended (one cell per run; number = test MAPE %, † = headline read test-week loads)",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT_FIG2, dpi=300)
    print(f"wrote {OUT_FIG2}")


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
        elif audit == "leaked":
            scope = str(sc.get("leak_scope", ""))
            if scope.startswith("supplement"):
                audit += " (supplement only; headline clean)"
            elif scope.startswith("selection"):
                audit += " (selection stage; headline features clean)"
            elif scope.startswith("intermediate"):
                audit += " (intermediate file only; final forecast clean)"
            elif scope.startswith("headline"):
                audit += " (headline)"
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


CLAUDE_FOOT = (
    "Hollow marker with * = session ended before the agent's final step, scored on the forecast it had written.\n"
    "† = the LightGBM winner read the test week's own actual loads through its rolling features from the 2nd hour and its\n"
    "24 h lag from the 2nd day; the other September L3 forecasts are recursive week-ahead forecasts. Opus 4.7, 4.8 and\n"
    "Opus 5 runs 2 and 3 are Extension C (DESIGN.md section 13); one May 2026 run per level is the reference point.")
OPENAI_FOOT = (
    "Filled = the frozen prompt (Extension A). Open = the same prompt plus a one-paragraph note that nobody can answer a\n"
    "question and the job must be finished in the session (Extension B, DESIGN.md section 12). A missing marker is a run\n"
    "that wrote no forecast: under the frozen prompt most sessions stopped to ask or to propose a plan. Hollow with * =\n"
    "session ended before the agent's final step; † = the winner read the test week's own loads through its lag and\n"
    "rolling features. These models ran on Claude Code through a local gateway: model plus translation layer, described.")


def main() -> int:
    ref = may_reference()
    results = load_results()
    build_figure(results, ref, CLAUDE_TAGS, OUT_FIG,
                 "Same three prompts, same data: May 2026 and September 2026", CLAUDE_FOOT)
    build_figure(results, ref, OPENAI_TAGS, OUT_FIG_OPENAI,
                 "The same prompts given to three OpenAI models, with and without a one-shot note", OPENAI_FOOT)
    build_outcomes_figure(results)
    build_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
