# JRC C4 — practical AI usage, illustrated with a forecasting experiment

A three-level prompt-engineering demo for a JRC C4 talk on practical AI for
research code. The same forecasting task — Germany hourly electricity load
for the first week of January 2020 — is run through Claude Code (Opus 4.7)
at three levels of *user sophistication*: prompt and workspace context
scale together to model a beginner (L1), an average user (L2), and a pro
(L3). Result: MAPE 10.76 → 5.52 → 3.43, same agent, same data.

## Layout

- `prompts/{L1,L2,L3}.md` — the three frozen prompts. Inputs to the
  experiment.
- `data/opsd_de_load.csv` — canonical copy of the German hourly load data
  (Open Power System Data, 2015–2020, ~50k rows).
- `scripts/fetch_data.py` — re-runnable producer of `data/opsd_de_load.csv`.
- `scripts/build_slide_figures.py` — generates the bespoke deck figures
  (`prompt_lengths.png`, `07_feature_importance_normalised.png`).
- `scripts/verify_slide_numbers.py` — cross-checks every quoted MAPE,
  coverage, word count, and figure path in the deck against the source
  metrics. Runs in CI-style: prints 21 OK/FAIL lines.
- `runs/L1/` — beginner workspace: just the data file. No `AGENTS.md`.
- `runs/L2/` — average-user workspace: data + 7-line `AGENTS.md`.
- `runs/L3/` — pro workspace: data + 113-line `AGENTS.md` with output
  schema, do-nots, style. Includes the bake-off code (`runs/L3/code/`)
  and full results (`runs/L3/figures/`, `metrics.json`, `transcript.md`).
- `slides/` — see next section.
- `RUNBOOK.md` — step-by-step for the human operator running Phase 1.

## Slide deck

| File | Purpose |
|---|---|
| `slides/talk_v2.md` | Marp source of the 24-slide standalone deck (the full version). |
| `slides/talk_v2.pdf` | Rendered standalone deck. View directly on GitHub. |
| `slides/talk_v2.pptx` | PowerPoint export of the same. Editable. |
| `slides/jrc_section_v2_beamer.tex` | XeLaTeX/Beamer source of the 8-slide drop-in for the JRC C4 parent deck. JRC blue + EC emblem footer. |
| `slides/jrc_section_v2_beamer.pdf` | Rendered Beamer drop-in. |
| `slides/speaker_companion.tex` | One-page-per-slide speaker companion (LaTeX article). Same JRC palette as the deck. "On screen" + "Say this" + anchor facts + transition cue, designed to be read on a second screen or phone while presenting. |
| `slides/speaker_companion.pdf` | Rendered speaker companion (9 pages: cover + 8 slide-pages). |
| `slides/assets/` | Bespoke figures used by the decks (prompt-length bar, normalised feature-importance, extracted EC emblem). |

### Regenerate decks

```sh
# Marp standalone (PDF + PPTX)
CHROME_PATH="/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" \
    marp slides/talk_v2.md -o slides/talk_v2.pdf --allow-local-files
CHROME_PATH="..." marp slides/talk_v2.md -o slides/talk_v2.pptx --allow-local-files

# Beamer drop-in (PDF only; run twice for refs)
cd slides && xelatex jrc_section_v2_beamer.tex && xelatex jrc_section_v2_beamer.tex

# Speaker companion (PDF only; run twice for refs)
cd slides && xelatex speaker_companion.tex && xelatex speaker_companion.tex
```

### Per-slide PNG export (drag-and-drop into PowerPoint)

```sh
mkdir -p slides/jrc_png && rm -f slides/jrc_png/*.png
pdftoppm -r 900 -png slides/jrc_section_v2_beamer.pdf slides/jrc_png/slide
```

The `slides/jrc_png/` folder is gitignored — regenerate locally when
needed. 900 dpi gives ~5670×3189 px per slide (~1 MB each). 600 dpi is
the conservative middle ground; 400 dpi is too soft for projector use.

### PowerPoint Online (Teams) note

PowerPoint Online (the web version, used inside Teams) does **not**
expose the desktop app's "Do not compress images" setting. Whatever
PNG you drag in is what's stored; on display the browser handles the
scaling. If inserted images look soft, the cause is almost always one
of:

1. You inserted lower-DPI PNGs earlier and never replaced them.
2. The deck is being previewed in the small edit pane, not the
   full-screen slideshow view.
3. Browser zoom level / monitor pixel density.

If onscreen quality during projection is unsatisfactory, share the
PDF directly via Teams screen-share instead of routing through
PowerPoint at all — it avoids both the PowerPoint image pipeline and
the browser scaler.

## Quickstart (reproduce the experiment)

```sh
uv sync                                    # exact versions from uv.lock
uv run python scripts/fetch_data.py        # produces data/opsd_de_load.csv
```

Then follow `RUNBOOK.md` for the Phase-1 manual runs (L1 → L2 → L3).
Each run lands in `runs/Lx/` with code, figures, metrics, and a
methods write-up.

The deliberate point of the three personae: workspace context scales
*with* the prompt. A real beginner doesn't have `AGENTS.md` set up; a
pro does. The three rows of the headline aren't just three different
prompts, they're three different user personae.

## Tools used

The forecasting experiment, the analytical write-ups in
`runs/L*/transcript.md`, the deck source (`slides/talk_v2.md`,
`slides/jrc_section_v2_beamer.tex`), and the supporting scripts were all
produced in collaboration with **Claude Code (Opus 4.7)** as the agent
under test and as the build assistant. Where the experiment runs the
agent at three increasingly specific prompt levels, the deck and code
around it were built with the L3 workflow throughout: detailed prompts,
a project-level `AGENTS.md`, and the `/review-loop` skill for plan
review and code review.
