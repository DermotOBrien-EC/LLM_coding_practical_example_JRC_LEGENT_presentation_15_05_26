// "Practical Use Cases" — 8-slide JRC C4 drop-in section.
// Typst port of slides/jrc_section_v2_beamer.tex (XeLaTeX/Beamer).
// Compile: typst compile jrc_section_v2.typ
//
// Slide 1 is a plain section divider (no footer). The seven content frames
// carry footer numbers 40–46, matching the slot in the JRC C4 parent deck.
#import "jrc-theme.typ": sans, mono, card, muted, faint, hair, l1, l2, fig-grid

// --- JRC palette (Beamer source) --------------------------------------
#let jrcblue = rgb("#3B3B8C")
#let l1grey  = rgb("#7F7F7F")
#let l2amber = rgb("#D97706")
#let l3orange = rgb("#EA580C")
#let darkgrey = rgb("#3A3A3A")

// --- document setup ----------------------------------------------------
#set page(
  paper: "presentation-16-9",
  margin: (x: 1.5cm, top: 1.15cm, bottom: 1.15cm),
  footer: context {
    let n = counter(page).get().first()
    if n > 1 {
      grid(columns: (1fr, 1fr),
        align(horizon + left, text(size: 9pt, fill: faint)[#(n + 38)]),
        align(horizon + right, image("assets/ec_logo.png", height: 15pt)),
      )
    }
  },
)
#set text(font: sans, size: 16pt, fill: black)
#set par(leading: 0.6em)
#set list(marker: text(fill: jrcblue)[•], spacing: 0.9em, indent: 3pt)
#show raw: set text(font: mono)
#show link: set text(fill: jrcblue)

// --- components --------------------------------------------------------
// Beamer \tipcard / \tipcardtight — a jrcblue-framed card.
#let tip-card(label, title, body, height: auto, body-size: 13pt, frame: jrcblue) = box(
  width: 100%, height: height, fill: card, stroke: 1pt + frame, radius: 3pt,
  inset: (x: 9pt, y: 7pt),
)[
  #text(weight: 700, size: body-size)[#label]
  #v(1pt)
  #text(fill: jrcblue, weight: 700, size: body-size + 1pt)[#title]
  #v(3pt)
  #text(size: body-size, fill: darkgrey)[#body]
]

// Beamer \levelcard — coloured-border card, fixed height for the row.
#let level-card(accent, title, body, height: 3.4cm, body-size: 12pt) = box(
  width: 100%, height: height, fill: card, stroke: 1.2pt + accent, radius: 3pt,
  inset: (x: 8pt, y: 7pt),
)[
  #text(fill: accent, weight: 700, size: 14pt)[#title]
  #v(2pt)
  #text(size: body-size, fill: darkgrey)[#body]
]

// Slide scaffold for the seven content frames.
#let frame(title, body) = {
  pagebreak(weak: true)
  block(text(size: 25pt, weight: 600, fill: jrcblue)[#title])
  v(10pt)
  body
}

#let cap(body, size: 13pt, fill: darkgrey) = align(
  center, text(size: size, fill: fill)[#body],
)

// ======================================================================
// Slide 1 (frame 39, plain) — section divider
// ======================================================================
#align(center + horizon)[
  #text(size: 42pt, weight: 700, fill: jrcblue)[Practical Use Cases]
  #v(14pt)
  #text(size: 22pt, fill: darkgrey)[Seven tips, one forecasting experiment]
  #v(80pt)
  #text(size: 14pt, fill: faint)[Speaker: Dermot O'Brien]
]

// ======================================================================
// Slide 2 (40) — Tips 1–5
// ======================================================================
#frame("Seven tips for working with AI")[
  #grid(columns: (1fr, 1fr), gutter: 18pt,
    tip-card("TIP 1", "Learn how to program, use your domain knowledge",
      [AI multiplies what you already know. Bring expertise; AI brings throughput.],
      height: 4.5cm),
    tip-card("TIP 2", "Be as specific as humanly possible",
      [Most prompts don't give enough context. Paste in docs, references, `llms.txt`, screenshots. Or draft a rough prompt and ask AI to rewrite it _"using LLM best practices."_],
      height: 4.5cm),
  )
  #v(13pt)
  #grid(columns: (1fr, 1fr, 1fr), gutter: 14pt,
    tip-card("TIP 3", "Smaller tasks beat big ones",
      [Break problems down. If you can't break it down, you don't understand it yet.],
      height: 4.2cm),
    tip-card("TIP 4", "Tell the AI what you don't want",
      [Three-section pattern: *TASK / BACKGROUND / DO NOT*.],
      height: 4.2cm),
    tip-card("TIP 5", "Tell AI to remember",
      [`AGENTS.md` (context) + skills (procedures) in `.claude/`. Ask the agent to draft v1.],
      height: 4.2cm),
  )
]

// ======================================================================
// Slide 3 (41) — The experiment
// ======================================================================
#frame("The experiment")[
  #text(size: 14pt)[
    - *Task*: forecast 168 hours of hourly German electricity load, 2020-01-01 to 2020-01-07.
    - *Data*: #link("https://data.open-power-system-data.org/time_series/")[Open Power System Data] (OPSD) / ENTSO-E, 2015–2020, hourly.
    - *Agent*: Claude Code, Opus 4.7 — same agent in every run.
    - *Code, prompts, and full results*: #link("https://github.com/DermotOBrien-EC/LLM_coding_practical_example_JRC_LEGENT_presentation_15_05_26")[github.com/DermotOBrien-EC/LLM_coding_practical_example_JRC_LEGENT_presentation_15_05_26]
    - *Three prompts, three workspace setups:*
  ]
  #v(9pt)
  #grid(columns: (1fr, 1fr, 1fr), gutter: 14pt,
    level-card(l1grey, "Level 1 — beginner", height: 4.3cm,
      [_"Forecast first week of January 2020 German hourly electricity load."_
       #v(4pt) *10 words · no AGENTS.md*]),
    level-card(l2amber, "Level 2 — average", height: 4.3cm,
      [Names data, horizon, libraries. Asks for plot + accuracy.
       #v(4pt) *46 words · 7-line AGENTS.md*]),
    level-card(l3orange, "Level 3 — research-grade", height: 4.3cm,
      [Structured TASK / BACKGROUND / DO NOT. 6-model bake-off, validation, intervals.
       #v(4pt) *1,673 words · 113-line AGENTS.md*]),
  )
  #v(7pt)
  #text(size: 9pt, fill: faint)[
    OPSD time series: #link("https://data.open-power-system-data.org/time_series/")
  ]
]

// ======================================================================
// Slide 4 (42) — Level 1 result
// ======================================================================
#frame("Level 1: what 10 words gets you")[
  #text(size: 14pt, style: "italic")[
    "Forecast first week of January 2020 German hourly electricity load."
  ]
  #v(8pt)
  #align(center, image("assets/L1_forecast.png", height: 56%))
  #v(8pt)
  #cap[
    Three baselines compared; L1 picked the best of three: *seasonal-naive (lag-364d), MAPE 10.76%*. \
    No validation set. No prediction intervals. No methodology document.
  ]
]

// ======================================================================
// Slide 5 (43) — Level 2 result
// ======================================================================
#frame("Level 2: average prompt + thin workspace")[
  #text(size: 13.5pt, style: "italic")[
    "Build me a Python script that forecasts German hourly electricity load for the first week of January 2020. Train on the historical data, plot it, and tell me how accurate it was."
  ]
  #v(3pt)
  #text(size: 11pt, fill: faint)[
    AGENTS.md: _Python project. opsd_de_load.csv has hourly German load. Use pandas. Plot with matplotlib._
  ]
  #v(7pt)
  #align(center, image("assets/L2_forecast.png", height: 49%))
  #v(7pt)
  #cap[
    *GradientBoostingRegressor*, engineered features. *MAPE 5.52%.*
    Still no held-out validation, no intervals, no methods document.
  ]
]

// ======================================================================
// Slide 6 (44) — Level 3 result
// ======================================================================
#frame("Level 3: research-grade prompt")[
  #text(size: 12pt, fill: darkgrey)[
    Prompt followed the TASK / BACKGROUND / DO NOT pattern (Tip 4).
    1,673 words · 113-line AGENTS.md · 6-model bake-off with held-out validation and calibrated intervals.
  ]
  #v(6pt)
  #align(center, image("assets/L3_05_winner_with_intervals.png", height: 50%))
  #v(6pt)
  #cap(size: 12pt)[
    *LightGBM* won a six-model bake-off. *MAPE 3.43%.* #h(10pt)
    80% PI covered *79.8%* · 95% PI covered *92.9%* — essentially nominal.
  ]
  #v(4pt)
  #cap(size: 12pt, fill: faint)[
    L2 might have got lucky picking a decent model. \
    L3 didn't have to be lucky — it searched, validated, calibrated, and documented.
  ]
]

// ======================================================================
// Slide 7 (45) — Some of what L3 produced
// ======================================================================
#frame("Some of what L3 actually produced")[
  #fig-grid(img-height: 3.6cm, caption-size: 11pt, (
    (path: "assets/L3_02_forecast_comparison.png", cap: "Six-model comparison"),
    (path: "assets/L3_03_metric_comparison.png", cap: "Scoreboard — MAPE, RMSE, MAE"),
    (path: "assets/L3_04_per_day_mape.png", cap: "Per-day MAPE heatmap"),
    (path: "assets/L3_06_residuals.png", cap: "Winner residuals by hour and day"),
    (path: "assets/07_feature_importance_normalised.png", cap: "Feature importance (normalised)"),
    (path: "assets/L3_08_decomposition.png", cap: "Prophet trend / seasonal decomposition"),
  ))
  #v(10pt)
  #cap(size: 13pt, fill: faint)[L1 produced 1 figure. L3 produced 8.]
]

// ======================================================================
// Slide 8 (46) — Tips 6, 7 + closing thesis
// ======================================================================
#frame("Seven tips for working with AI (cont.)")[
  #grid(columns: (1fr, 1fr), gutter: 18pt,
    stack(spacing: 13pt,
      tip-card("TIP 6", "MCPs extend what AI can do",
        [Plug-in tools at runtime. Context7 (docs), Hugging Face (papers), GitHub (repos).],
        height: 3.5cm),
      tip-card("TIP 7", "Always give AI a way to verify its work",
        [Tests, held-out validation, calibrated intervals. Have AI generate them _and_ read what it wrote — AI-written tests can "pass" while testing the wrong thing.],
        height: 4.6cm),
    ),
    box(
      width: 100%, height: 8.4cm, fill: rgb("#ECECF5"),
      stroke: 1.5pt + jrcblue, radius: 3pt, inset: 13pt,
    )[
      #text(fill: jrcblue, weight: 700, size: 19pt)[Closing thesis]
      #v(8pt)
      #text(size: 15pt)[
        *AI amplifies the habits you already have.*

        Let it type for you — not think for you.

        Bring good engineering habits and AI multiplies them.

        Skip tests, skip docs, skip the "what could go wrong" question — and AI amplifies _that_ instead.
      ]
    ],
  )
]
