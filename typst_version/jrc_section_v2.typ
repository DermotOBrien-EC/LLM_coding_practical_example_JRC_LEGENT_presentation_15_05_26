// "Practical Use Cases" — 8-slide JRC C4 drop-in section.
// Frankfurt theme. Footer keeps the parent-deck slot numbers 40–46.
// Compile: typst compile jrc_section_v2.typ
#import "jrc-theme.typ": *

#show: frankfurt-deck.with(
  sections: (
    ("Tips", 2, 2),
    ("The experiment", 3, 7),
    ("Takeaways", 8, 8),
  ),
  author: "Dermot O'Brien",
  doc-title: "Practical Use Cases",
  date: "JRC C4",
  total: 46,
  page-offset: 38,
)

// Frankfurt-blue framed tip card (label · title · body).
#let tip-card(label, title, body, height: auto, body-size: 12pt) = box(
  width: 100%, height: height, fill: card, stroke: 1.2pt + fblue, radius: 4pt,
  inset: (x: 9pt, y: 7pt),
)[
  #text(weight: 700, size: 12pt, fill: fmid)[#label]
  #v(1pt)
  #text(fill: fblue, weight: 700, size: 13.5pt)[#title]
  #v(3pt)
  #text(size: body-size, fill: rgb("#333333"))[#body]
]

// ======================================================================
// Slide 1 — cover / section divider
// ======================================================================
#cover(
  title: "Practical Use Cases",
  subtitle: [Seven tips, one forecasting experiment],
  meta: [Speaker: Dermot O'Brien],
  logo: image("assets/ec_logo.png", height: 13mm),
)

// ======================================================================
// Slide 2 (40) — Tips 1–5
// ======================================================================
#slide("Seven tips for working with AI")[
  #grid(columns: (1fr, 1fr), gutter: 16pt,
    tip-card("TIP 1", "Learn how to program, use your domain knowledge",
      [AI multiplies what you already know. Bring expertise; AI brings throughput.],
      height: 5.3cm),
    tip-card("TIP 2", "Be as specific as humanly possible",
      [Most prompts don't give enough context. Paste in docs, references, `llms.txt`, screenshots. Or draft a rough prompt and ask AI to rewrite it _"using LLM best practices."_],
      height: 5.3cm),
  )
  #v(11pt)
  #grid(columns: (1fr, 1fr, 1fr), gutter: 13pt,
    tip-card("TIP 3", "Smaller tasks beat big ones",
      [Break problems down. If you can't break it down, you don't understand it yet.],
      height: 4.9cm),
    tip-card("TIP 4", "Tell the AI what you don't want",
      [Three-section pattern: *TASK / BACKGROUND / DO NOT*.],
      height: 4.9cm),
    tip-card("TIP 5", "Tell AI to remember",
      [`AGENTS.md` (context) + skills (procedures) in `.claude/`. Ask the agent to draft v1.],
      height: 4.9cm),
  )
]

// ======================================================================
// Slide 3 (41) — The experiment
// ======================================================================
#slide("The experiment")[
  #text(size: 13.5pt)[
    - *Task*: forecast 168 hours of hourly German electricity load, 2020-01-01 to 2020-01-07.
    - *Data*: #link("https://data.open-power-system-data.org/time_series/")[Open Power System Data] (OPSD) / ENTSO-E, 2015–2020, hourly.
    - *Agent*: Claude Code, Opus 4.7 — same agent in every run.
    - *Code, prompts, and full results*: #link("https://github.com/DermotOBrien-EC/LLM_coding_practical_example_JRC_LEGENT_presentation_15_05_26")[github.com/DermotOBrien-EC/LLM_coding_practical_example_JRC_LEGENT_presentation_15_05_26]
    - *Three prompts, three workspace setups:*
  ]
  #v(8pt)
  #grid(columns: (1fr, 1fr, 1fr), gutter: 13pt,
    level-card(l1, "Level 1 — beginner", body-size: 12pt, title-size: 14pt, height: 4.2cm)[
      _"Forecast first week of January 2020 German hourly electricity load."_
      #v(4pt) *10 words · no AGENTS.md*
    ],
    level-card(l2, "Level 2 — average", body-size: 12pt, title-size: 14pt, height: 4.2cm)[
      Names data, horizon, libraries. Asks for plot + accuracy.
      #v(4pt) *46 words · 7-line AGENTS.md*
    ],
    level-card(l3, "Level 3 — research-grade", body-size: 12pt, title-size: 14pt, height: 4.2cm)[
      Structured TASK / BACKGROUND / DO NOT. 6-model bake-off, validation, intervals.
      #v(4pt) *1,673 words · 113-line AGENTS.md*
    ],
  )
]

// ======================================================================
// Slide 4 (42) — Level 1 result
// ======================================================================
#slide("Level 1: what 10 words gets you")[
  #text(size: 14pt, style: "italic")[
    "Forecast first week of January 2020 German hourly electricity load."
  ]
  #v(8pt)
  #align(center, image("assets/L1_forecast.png", width: 50%))
  #v(8pt)
  #caption[
    Three baselines compared; L1 picked the best of three: *seasonal-naive (lag-364d), MAPE 10.76%*. \
    No validation set. No prediction intervals. No methodology document.
  ]
]

// ======================================================================
// Slide 5 (43) — Level 2 result
// ======================================================================
#slide("Level 2: average prompt + thin workspace")[
  #text(size: 13pt, style: "italic")[
    "Build me a Python script that forecasts German hourly electricity load for the first week of January 2020. Train on the historical data, plot it, and tell me how accurate it was."
  ]
  #v(3pt)
  #text(size: 11pt, fill: faint)[
    AGENTS.md: _Python project. opsd_de_load.csv has hourly German load. Use pandas. Plot with matplotlib._
  ]
  #v(7pt)
  #align(center, image("assets/L2_forecast.png", width: 47%))
  #v(6pt)
  #caption(size: 14pt)[
    *GradientBoostingRegressor*, engineered features. *MAPE 5.52%.*
    Still no held-out validation, no intervals, no methods document.
  ]
]

// ======================================================================
// Slide 6 (44) — Level 3 result
// ======================================================================
#slide("Level 3: research-grade prompt")[
  #text(size: 12pt, fill: muted)[
    Prompt followed the TASK / BACKGROUND / DO NOT pattern (Tip 4).
    1,673 words · 113-line AGENTS.md · 6-model bake-off with held-out validation and calibrated intervals.
  ]
  #v(6pt)
  #align(center, image("assets/L3_05_winner_with_intervals.png", width: 42%))
  #v(6pt)
  #caption(size: 13pt)[
    *LightGBM* won a six-model bake-off. *MAPE 3.43%.* #h(8pt)
    80% PI covered *79.8%* · 95% PI covered *92.9%* — essentially nominal.
  ]
  #v(3pt)
  #caption(size: 12pt)[
    L2 might have got lucky picking a decent model. \
    L3 didn't have to be lucky — it searched, validated, calibrated, and documented.
  ]
]

// ======================================================================
// Slide 7 (45) — Some of what L3 produced
// ======================================================================
#slide("Some of what L3 actually produced")[
  #fig-grid(img-height: 3.05cm, caption-size: 11pt, (
    (path: "assets/L3_02_forecast_comparison.png", cap: "Six-model comparison"),
    (path: "assets/L3_03_metric_comparison.png", cap: "Scoreboard — MAPE, RMSE, MAE"),
    (path: "assets/L3_04_per_day_mape.png", cap: "Per-day MAPE heatmap"),
    (path: "assets/L3_06_residuals.png", cap: "Winner residuals by hour and day"),
    (path: "assets/07_feature_importance_normalised.png", cap: "Feature importance (normalised)"),
    (path: "assets/L3_08_decomposition.png", cap: "Prophet trend / seasonal decomposition"),
  ))
  #v(9pt)
  #caption(size: 13pt)[L1 produced 1 figure. L3 produced 8.]
]

// ======================================================================
// Slide 8 (46) — Tips 6, 7 + closing thesis
// ======================================================================
#slide("Seven tips for working with AI (cont.)")[
  #grid(columns: (1fr, 1fr), gutter: 16pt,
    stack(spacing: 12pt,
      tip-card("TIP 6", "MCPs extend what AI can do",
        [Plug-in tools at runtime. Context7 (docs), Hugging Face (papers), GitHub (repos).],
        height: 3.7cm),
      tip-card("TIP 7", "Always give AI a way to verify its work",
        [Tests, held-out validation, calibrated intervals. Have AI generate them _and_ read what it wrote — AI-written tests can "pass" while testing the wrong thing.],
        height: 5.1cm),
    ),
    box(
      width: 100%, height: 9cm, fill: fitem,
      stroke: 1.5pt + fblue, radius: 5pt, inset: 14pt,
    )[
      #text(fill: fblue, weight: 700, size: 18pt)[Closing thesis]
      #v(9pt)
      #text(size: 14.5pt)[
        *AI amplifies the habits you already have.*

        Let it type for you — not think for you.

        Bring good engineering habits and AI multiplies them.

        Skip tests, skip docs, skip the "what could go wrong" question — and AI amplifies _that_ instead.
      ]
    ],
  )
]
