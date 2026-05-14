"""Build the bespoke figures that the slide deck needs.

Two figures are bespoke for the deck (the rest are lifted from
runs/L*/figures/ unchanged):
- prompt-length bar chart (Tip 2 word-count visual)
- feature-importance bar chart, normalised to percentages — the L3 agent
  emitted raw "gain" values which read as 1e13 on the x-axis and are
  presentationally awful even though they're mathematically correct.

Usage:
    uv run python scripts/build_slide_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "slides" / "assets"


# Persona colours — chosen so L1 is grey ("undifferentiated beginner"),
# L2 is light orange ("warming up"), L3 is saturated tab10 orange
# ("the colour LightGBM uses everywhere else in the deck — the winner").
PERSONAS = [
    ("L1 — beginner", 10, "#7f7f7f"),
    ("L2 — average", 46, "#ffbb78"),
    ("L3 — pro", 1673, "#ff7f0e"),
]


def build_prompt_lengths_figure() -> Path:
    out = OUT_DIR / "prompt_lengths.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)

    labels = [p[0] for p in PERSONAS]
    counts = [p[1] for p in PERSONAS]
    colours = [p[2] for p in PERSONAS]

    bars = ax.barh(labels, counts, color=colours, edgecolor="black", linewidth=0.6)
    ax.invert_yaxis()  # L1 on top

    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count} words",
            va="center",
            ha="left",
            fontsize=18,
            fontweight="bold",
        )

    ax.set_xlabel("Prompt length (words)", fontsize=16)
    ax.set_xlim(0, max(counts) * 1.18)
    ax.tick_params(axis="y", labelsize=18)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_title(
        "Same task, three prompts — the words grow two orders of magnitude",
        fontsize=18,
        pad=14,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def build_feature_importance_normalised_figure() -> Path:
    """Normalised LightGBM feature importance from runs/L3/metrics.json.

    The L3 agent saved raw "gain" values (cumulative reduction in squared
    error contributed by each feature across every split in every tree).
    With targets in MW (~50,000 mean) and ~600 trees, the totals naturally
    land at O(1e13) — mathematically valid but unreadable. Renormalise
    each feature's gain as a percentage of the total.
    """
    out = OUT_DIR / "07_feature_importance_normalised.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((ROOT / "runs" / "L3" / "metrics.json").read_text())
    lgbm = next(m for m in metrics["models"] if m["name"] == "lightgbm")
    gains = lgbm["hyperparameters"]["feature_importance_gain"]

    total = sum(gains.values())
    items = sorted(((k, v / total * 100) for k, v in gains.items()), key=lambda p: p[1], reverse=True)
    names = [k for k, _ in items]
    pcts = [p for _, p in items]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    bars = ax.barh(names, pcts, color="#ff7f0e", edgecolor="black", linewidth=0.4)
    ax.invert_yaxis()  # most important on top

    for bar, pct in zip(bars, pcts, strict=True):
        ax.text(
            bar.get_width() + max(pcts) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            ha="left",
            fontsize=11,
        )

    ax.set_xlabel("Relative importance (% of total gain)", fontsize=12)
    ax.set_xlim(0, max(pcts) * 1.15)
    ax.set_title("LightGBM feature importance — normalised", fontsize=14, pad=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    for builder in (build_prompt_lengths_figure, build_feature_importance_normalised_figure):
        out = builder()
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
