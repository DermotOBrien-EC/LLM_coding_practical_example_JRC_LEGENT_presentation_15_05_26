// "Seven tips for working with AI" — 24-slide standalone deck.
// Typst port of slides/talk_v2.md (Marp). Compile: typst compile talk_v2.typ
#import "jrc-theme.typ": *

#let primary = rgb("#1A3A6A")

// --- document setup ----------------------------------------------------
#set page(
  paper: "presentation-16-9",
  margin: (x: 1.7cm, top: 1.25cm, bottom: 1.3cm),
  footer: context {
    let n = counter(page).get().first()
    let total = counter(page).final().first()
    if n != 1 and n != total {
      align(right, text(size: 12pt, fill: faint)[#n])
    }
  },
)
#set text(font: sans, size: 20pt, fill: ink)
#set par(leading: 0.72em, justify: false)
#set list(marker: text(fill: primary)[•], spacing: 1.15em, indent: 4pt)
#set enum(spacing: 1.15em, indent: 4pt)
#show raw: set text(font: mono)
#show raw.where(block: true): it => box(
  width: 100%, fill: rgb("#F4F4F4"), inset: (x: 13pt, y: 11pt), radius: 5pt,
  text(size: 14pt, it),
)
#show raw.where(block: false): it => box(
  fill: rgb("#F4F4F4"), inset: (x: 4pt, y: 1pt), radius: 3pt,
  text(size: 0.92em, it),
)
#show strong: set text(fill: primary)

// --- slide scaffold ----------------------------------------------------
#let slide(title: none, tag: none, body) = {
  pagebreak(weak: true)
  if tag != none { tag; v(6pt) }
  if title != none {
    block(text(size: 28pt, weight: 700, fill: primary)[#title])
    v(9pt)
  }
  body
}

// ======================================================================
// Slide 1 — title
// ======================================================================
#align(horizon)[
  #text(size: 46pt, weight: 700, fill: primary)[Seven tips for working with AI]
  #v(16pt)
  #text(size: 26pt, fill: muted)[
    A field-tested playbook, \
    illustrated with a forecasting experiment we ran in this unit.
  ]
  #v(64pt)
  #text(size: 18pt, fill: faint)[JRC C4 · Dermot O'Brien · 2026]
]

// ======================================================================
// Slide 2 — TIP 1
// ======================================================================
#slide(tag: tip-tag("TIP 1", accent: primary), title: "Learn how to program first")[
  #text(size: 22pt)[
    - AI is a *multiplier of what you already know.*
    - It cannot replace the domain knowledge that an economist or engineer in C4 brings to a forecasting model, an impact assessment, or a policy briefing.
    - If you don't know what "good" looks like for the task, no prompt will rescue you — you won't be able to tell when the output is wrong.
  ]
  #v(22pt)
  #caption[The rest of this talk is about how to convert your domain knowledge into AI output that's actually usable.]
]

// ======================================================================
// Slide 3 — TIP 2
// ======================================================================
#slide(tag: tip-tag("TIP 2", accent: primary), title: "Be as specific as humanly possible")[
  #bquote[#text(size: 23pt)[AI is only as good as the context you give it.]]
  #v(20pt)
  #text(size: 22pt)[
    - Most people don't give it enough.
    - The next slides show *an experiment we ran in this unit* to quantify how much it matters.
  ]
]

// ======================================================================
// Slide 4 — The experiment
// ======================================================================
#slide(title: "The experiment")[
  #text(size: 18pt)[
    - *Task*: forecast 168 hours (one week) of hourly German electricity load, 2020-01-01 to 2020-01-07.
    - *Data*: Open Power System Data (OPSD) / ENTSO-E, 2015–2020, hourly.
    - *Agent*: Claude Code, Opus 4.7 — same agent in every run.
    - *Three prompts*, three workspace setups:
  ]
  #v(10pt)
  #grid(columns: (1fr, 1fr, 1fr), gutter: 15pt,
    level-card(l1, body-size: 13pt, title-size: 16pt, height: 5.7cm)[Level 1 — beginner][
      "Forecast first week of January 2020 German hourly electricity load."
      #v(5pt) #text(size: 12pt, fill: muted)[10 words · no AGENTS.md]
    ],
    level-card(l2, body-size: 13pt, title-size: 16pt, height: 5.7cm)[Level 2 — average][
      Names the data file, the horizon, the libraries. Asks for a plot and an accuracy number.
      #v(5pt) #text(size: 12pt, fill: muted)[46 words · 7-line AGENTS.md]
    ],
    level-card(l3, body-size: 13pt, title-size: 16pt, height: 5.7cm)[Level 3 — research-grade][
      Structured prompt (TASK / BACKGROUND / DO NOT). 6-model bake-off. Validation splits. Calibrated intervals. Written methods.
      #v(5pt) #text(size: 12pt, fill: muted)[1,673 words · 113-line AGENTS.md]
    ],
  )
]

// ======================================================================
// Slide 5 — Level 1 result
// ======================================================================
#slide(title: "Level 1 — what 10 words gets you")[
  #bquote[#text(size: 21pt)[Forecast first week of January 2020 German hourly electricity load.]]
  #v(12pt)
  #align(center, image("assets/L1_forecast.png", width: 55%))
  #v(10pt)
  #caption(size: 17pt)[
    Three baselines compared; L1 picked the best of three: *seasonal-naive (lag-364d), MAPE 10.76%*. \
    No validation set. No prediction intervals. No methodology document.
  ]
]

// ======================================================================
// Slide 6 — Level 2 prompt + workspace
// ======================================================================
#slide(title: "Level 2 — the prompt and the workspace")[
  #twocol(
    [
      #text(size: 20pt, weight: 700, fill: primary)[The prompt (46 words)]
      #v(8pt)
      ```
      Build me a Python script that
      forecasts German hourly electricity
      load for the first week of January
      2020. Train on the historical data,
      plot it against the actual values,
      and tell me how accurate it was.
      ```
    ],
    [
      #text(size: 20pt, weight: 700, fill: primary)[AGENTS.md (7 lines)]
      #v(8pt)
      ```
      # Notes for the AI
      Python project. The file
      opsd_de_load.csv has hourly
      German load (MW) from OPSD.
      Use pandas. Plot with matplotlib.
      ```
    ],
  )
]

// ======================================================================
// Slide 7 — Level 2 result
// ======================================================================
#slide(title: "Level 2 — the result")[
  #align(center, image("assets/L2_forecast.png", width: 66%))
  #v(12pt)
  #caption(size: 17pt)[
    The agent picked *GradientBoostingRegressor* with hand-engineered features (hour, day-of-week, lag-168h, lag-8760h). \
    *MAPE 5.52%.* Still no held-out validation, no prediction intervals, no methodology document.
  ]
]

// ======================================================================
// Slide 8 — Level 3 prompt structure
// ======================================================================
#slide(title: "Level 3 — the prompt structure (1,673 words)")[
  #stack(spacing: 12pt,
    pattern-box(task, "TASK")[Forecast 168 hours. Six-model bake-off. Report MAPE, RMSE, MAE.],
    pattern-box(bg, "BACKGROUND")[Data · Allowed inputs · Train/validate/test splits · Metrics · Required figures · Output structure · AGENTS.md schema (113 lines)],
    pattern-box(dont, "DO NOT")[Use the test window for fitting. Pull external data. Skip refit-on-train+val. Silently drop missing hours.],
  )
  #v(16pt)
  #caption[Three sections. Same pattern for forecasting, paper drafting, literature search, debugging.]
]

// ======================================================================
// Slide 9 — Level 3 result
// ======================================================================
#slide(title: "Level 3 — the result")[
  #align(center, image("assets/L3_05_winner_with_intervals.png", height: 66%))
  #v(10pt)
  #caption(size: 17pt)[
    LightGBM won a six-model bake-off. *MAPE 3.43%.* \
    80% prediction interval covered *79.8%* of points; 95% covered *92.9%* — essentially nominal.
  ]
]

// ======================================================================
// Slide 10 — the headline
// ======================================================================
#slide(title: "The headline isn't the MAPE — it's everything around it")[
  #grid(columns: (1fr, 1fr, 1fr), gutter: 15pt,
    level-card(l1, body-size: 13pt, title-size: 15pt, height: 7.7cm)[L1 — beginner][
      *MAPE 10.76%* \
      3 baselines, picked best \
      No validation set \
      No prediction intervals \
      No methods document \
      1 figure \
      Single script
    ],
    level-card(l2, body-size: 13pt, title-size: 15pt, height: 7.7cm)[L2 — average][
      *MAPE 5.52%* _(one model — got lucky)_ \
      1 model: GradientBoostingRegressor \
      No validation set \
      No prediction intervals \
      No methods document \
      1 figure \
      Single script
    ],
    level-card(l3, body-size: 13pt, title-size: 15pt, height: 7.7cm)[L3 — research-grade][
      *MAPE 3.43%* _(winner of a 6-model search)_ \
      6 models compared systematically \
      Held-out validation set (2019 Q4) \
      80%/95% intervals — 79.8%/92.9% coverage \
      Methods document (transcript.md) \
      8 figures \
      Orchestrator + per-model modules
    ],
  )
  #v(10pt)
  #caption[
    L2 might have got lucky picking a decent model. \
    L3 didn't have to be lucky — it searched, validated, calibrated, and documented.
  ]
]

// ======================================================================
// Slide 11 — what L3 produced
// ======================================================================
#slide(title: "Some of what L3 actually produced")[
  #fig-grid(img-height: 3.5cm, (
    (path: "assets/L3_02_forecast_comparison.png", cap: "Six-model forecast comparison"),
    (path: "assets/L3_03_metric_comparison.png", cap: "Scoreboard — MAPE, RMSE, MAE"),
    (path: "assets/L3_04_per_day_mape.png", cap: "Per-day MAPE heatmap"),
    (path: "assets/L3_06_residuals.png", cap: "Winner residuals by hour and day"),
    (path: "assets/07_feature_importance_normalised.png", cap: "LightGBM feature importance (normalised)"),
    (path: "assets/L3_08_decomposition.png", cap: "Prophet trend / seasonal decomposition"),
  ))
  #v(9pt)
  #caption(size: 14pt)[Each figure was generated because the L3 prompt asked for it. L1 and L2 produced one figure each.]
]

// ======================================================================
// Slide 12 — TIP 2 sub-tip: feed what you'd Google
// ======================================================================
#slide(tag: subtip-tag("TIP 2 · SUB-TIP", accent: primary), title: "Feed the AI what you'd Google")[
  #text(size: 20pt)[
    - AI assistants can read the web. Don't make them guess what you already know how to look up.
    - *Paste the docs.* API references, methodology papers, code examples — drop them into the prompt or attach them as files.
    - *Use `llms.txt` pages.* Some libraries publish their docs in a format designed for LLMs; reference the URL.
    - *Use screenshots.* Easier than describing a chart you want or a UI layout you have in mind.
  ]
  #v(13pt)
  #caption[This is the easiest 20-minute upgrade you can make to your prompts.]
]

// ======================================================================
// Slide 13 — TIP 2 sub-tip: improve your own prompt
// ======================================================================
#slide(tag: subtip-tag("TIP 2 · SUB-TIP", accent: primary), title: "Have the AI improve your own prompt")[
  #text(size: 22pt)[
    + Write a rough draft prompt with all the technical information you have.
    + Paste it back to the AI and ask: _"Rewrite this prompt using LLM best practices for a structured task description."_
    + Review the rewrite. Keep what improves it, discard what's wrong.
  ]
  #v(24pt)
  #caption[A 30-second meta-step. Catches the structure-and-edge-case gaps you didn't think of.]
]

// ======================================================================
// Slide 14 — TIP 3
// ======================================================================
#slide(tag: tip-tag("TIP 3", accent: primary), title: "The smaller the task, the better the results")[
  #text(size: 22pt)[
    - AI is good at small tasks. Worse at big, complex ones.
    - Break a big task into smaller ones.
    - *If you can't break it down, you don't understand the problem well enough yet.*
    - This isn't an AI trick — it's fundamental engineering: plan the solution, decompose it, then code (or have AI code) each piece.
  ]
  #v(20pt)
  #caption[
    We did exactly this in our experiment — split into Phase 1 (run the three levels), \
    Phase 2 (write the deck), with a plan-and-review step at every boundary.
  ]
]

// ======================================================================
// Slide 15 — TIP 3 sub-tip: type, don't think
// ======================================================================
#slide(tag: subtip-tag("TIP 3 · SUB-TIP", accent: primary), title: "Let AI type for you — not think for you")[
  #text(size: 23pt)[
    - Letting AI *type* the code or the prose for you is fine, even excellent.
    - Letting AI *think* for you means you stop applying the skills that justify your role.
    - The moment you outsource the decisions, the model's mistakes become invisible to you — because you've stopped having an opinion.
  ]
  #v(22pt)
  #caption[Plan the solution yourself. Read the output critically. Push back when it's wrong.]
]

// ======================================================================
// Slide 16 — TIP 4
// ======================================================================
#slide(tag: tip-tag("TIP 4", accent: primary), title: "Tell the AI what you don't want")[
  #text(size: 16pt)[
    The single highest-leverage structural change to any prompt is to add a "DO NOT" block. The pattern:
  ]
  #v(7pt)
  #let pb(accent, label, body) = pattern-box(
    accent, label, body, label-size: 15pt, body-size: 13pt, pad: 4pt,
  )
  #stack(spacing: 6pt,
    pb(task, "TASK")[What to do, precisely — the objective, the metric, the output shape.],
    pb(bg, "BACKGROUND")[The data, the files, the constraints, the references the model needs.],
    pb(dont, "DO NOT")[What to avoid — common mistakes, things that look right but aren't.],
  )
  #v(7pt)
  #caption(size: 14pt)[
    Our Level 3 prompt had nine specific don'ts. The one that mattered most: \
    _do not use the test window for fitting._
  ]
]

// ======================================================================
// Slide 17 — TIP 5
// ======================================================================
#slide(tag: tip-tag("TIP 5", accent: primary), title: "Tell the AI to remember — AGENTS.md")[
  #text(size: 18pt)[
    - Every Claude Code session reads an `AGENTS.md` file at startup, if present.
    - Put project standing context there *once*, not in every prompt: tech stack, file layout, data sources, conventions, things the agent should never do.
    - Our experiment quantified the gap:
  ]
  #v(8pt)
  #grid(columns: (1fr, 1fr, 1fr), gutter: 15pt,
    level-card(l1, body-size: 11pt, title-size: 15pt, height: 3.4cm)[L1][No AGENTS.md.],
    level-card(l2, body-size: 11pt, title-size: 15pt, height: 3.4cm)[L2 — 7 lines][Data file location, libraries.],
    level-card(l3, body-size: 11pt, title-size: 15pt, height: 3.4cm)[L3 — 113 lines][Allowed/forbidden inputs · output schema · `metrics.json` contract · DO-NOT block · style rules.],
  )
  #v(8pt)
  #caption[Single biggest leverage point most users miss.]
]

// ======================================================================
// Slide 18 — TIP 5 sub-tip: draft AGENTS.md itself
// ======================================================================
#slide(tag: subtip-tag("TIP 5 · SUB-TIP", accent: primary), title: "Have the AI draft AGENTS.md itself")[
  #text(size: 22pt)[
    + Point the agent at your existing codebase: _"Analyse this repository and propose an AGENTS.md."_
    + The agent reads the structure, infers conventions, drafts a v1.
    + You edit. Add the things it missed, remove the things it inferred wrongly.
  ]
  #v(24pt)
  #caption[Ten minutes' work. Saves hours of re-explaining your project for the rest of the year.]
]

// ======================================================================
// Slide 19 — TIP 5 sub-tip: skills
// ======================================================================
#slide(tag: subtip-tag("TIP 5 · SUB-TIP", accent: primary), title: "Reusable workflows — Skills")[
  #text(size: 19pt)[
    - A *skill* is a named, reusable workflow you can invoke from the agent — a markdown file the agent reads when triggered.
    - Lives in `~/.claude/skills/<name>/SKILL.md` (global) or `.claude/skills/<name>/SKILL.md` (project).
    - We used one to build this deck: `/review-loop plan` / `/review-loop code` — codifies the plan-then-implement-then-review pattern.
    - Heuristic: any workflow you do more than twice should be a skill. Sharable — drop a skill folder into a colleague's `.claude/skills/`.
  ]
  #v(14pt)
  #caption(size: 15pt)[AGENTS.md gives the agent persistent _context_. Skills give it persistent _procedures_.]
]

// ======================================================================
// Slide 20 — TIP 6
// ======================================================================
#slide(tag: tip-tag("TIP 6", accent: primary), title: "MCPs can extend what AI can do")[
  #text(size: 20pt)[
    *MCP = Model Context Protocol.* Plug-in tools the agent can invoke at runtime.

    A handful that are immediately useful for research code in C4:

    - *Context7* — fetches up-to-date library documentation on demand. No more pasting docs into prompts.
    - *Hugging Face MCP* — search models, datasets, papers; pull metadata.
    - *GitHub MCP* — read repos, issues, PRs; useful when working across projects.
  ]
  #v(16pt)
  #caption[Find the MCPs that match your tech stack — there's probably one for every external system you touch.]
]

// ======================================================================
// Slide 21 — TIP 7
// ======================================================================
#slide(tag: tip-tag("TIP 7", accent: primary), title: "Always give the AI a way to verify its work")[
  #text(size: 20pt)[
    AI should never just write code. It also needs a way to *prove the code works.*

    - *Tests* — unit, integration, regression.
    - *A held-out validation set* for any model fitting.
    - *Calibration metrics* — prediction interval coverage, residual diagnostics.
    - *A written discussion of failure modes* — where it expects the model to break.
  ]
  #v(14pt)
  #caption[
    Our Level 3 prompt asked for all four. The 80%/95% prediction intervals came back \
    covering *79.8%* and *92.9%* of observed points — essentially nominal coverage.
  ]
]

// ======================================================================
// Slide 22 — TIP 7 sub-tip: generate the tests
// ======================================================================
#slide(tag: subtip-tag("TIP 7 · SUB-TIP", accent: primary), title: "Have the AI generate the tests too — but verify them")[
  #text(size: 22pt)[
    - The fastest way to get test coverage is to ask the agent to write the tests.
    - The risk: AI-written tests can be tautological — they "pass" because they test the wrong thing.
    - *Read every test it writes.* A failing test that catches a real bug is worth ten passing tests that don't.
  ]
  #v(22pt)
  #caption[
    Same principle for AI-written validation scripts, AI-written sensitivity analyses, AI-written everything. \
    Trust, then verify the trust.
  ]
]

// ======================================================================
// Slide 23 — closing thesis
// ======================================================================
#slide(title: "Closing thesis — AI amplifies the habits you already have")[
  #text(size: 22pt)[
    The seven tips are not really AI tips. They are *good engineering habits*:

    - Be specific. Break the problem down. Tell collaborators what not to do. Write things down. Build verification in.

    If you bring those habits, AI multiplies your throughput substantially.

    If you don't — if you skip tests, skip docs, skip the "what could go wrong" question — AI amplifies *that* instead, faster than you can audit.
  ]
  #v(16pt)
  #caption[Be the kind of researcher whose habits AI is worth amplifying.]
]

// ======================================================================
// Slide 24 — closing
// ======================================================================
#pagebreak(weak: true)
#align(center + horizon)[
  #text(size: 58pt, weight: 700, fill: l3)[Your goal is leverage.]
  #v(20pt)
  #block(width: 72%, text(size: 28pt, fill: muted)[
    AI scales execution. You bring the judgement.
  ])
  #v(54pt)
  #text(size: 15pt, fill: faint, style: "italic")[
    code & data: github.com/DermotOBrien-EC/LLM_coding_practical_example_JRC_LEGENT_presentation_15_05_26
  ]
]
