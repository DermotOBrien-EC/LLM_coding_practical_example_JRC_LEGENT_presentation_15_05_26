"""Build the bespoke figures that the slide deck needs.

Right now there is exactly one: the prompt-length bar chart for the
"what changed: the prompt itself" slide. The other deck figures are
lifted from runs/L*/figures/ unchanged.

Usage:
    uv run python scripts/build_slide_figures.py
"""

from __future__ import annotations

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


def main() -> int:
    out = build_prompt_lengths_figure()
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
