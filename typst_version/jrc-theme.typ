// Shared theme for the JRC C4 "Practical AI" decks.
// Re-implements the Marp/Beamer styled components as plain Typst functions.
// No external packages — `typst compile` works offline once fonts resolve.

// --- fonts -------------------------------------------------------------
// Fallback lists so a recompile on Linux degrades gracefully. The compiled
// PDF/PNG/SVG shipped alongside are the canonical artifacts (see README).
#let sans = ("Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans")
#let mono = ("Menlo", "SF Mono", "DejaVu Sans Mono", "Liberation Mono")

// --- shared palette ----------------------------------------------------
#let l1     = rgb("#7F7F7F")  // beginner   — grey
#let l2     = rgb("#D97706")  // average    — amber
#let l3     = rgb("#FF7F0E")  // research   — orange
#let task   = rgb("#1F77B4")  // TASK       — blue
#let bg     = rgb("#2CA02C")  // BACKGROUND — green
#let dont   = rgb("#D62728")  // DO NOT     — red
#let card   = rgb("#FAFAFA")  // card fill
#let ink    = rgb("#1A1A1A")  // body text
#let muted  = rgb("#555555")  // captions / secondary
#let faint  = rgb("#999999")  // footer / credits
#let hair   = rgb("#E0E0E0")  // thin borders

// --- chips -------------------------------------------------------------
// Filled "TIP n" chip placed above a slide title.
#let tip-tag(label, accent: rgb("#1A3A6A")) = box(
  fill: accent, inset: (x: 11pt, y: 5pt), radius: 3pt,
  text(fill: white, weight: 600, size: 15pt, tracking: 0.5pt, upper(label)),
)

// Lighter "TIP n · SUB-TIP" chip.
#let subtip-tag(label, accent: rgb("#1A3A6A")) = box(
  fill: rgb("#E7EEF7"), inset: (x: 10pt, y: 4pt), radius: 3pt,
  text(fill: accent, weight: 600, size: 13pt, tracking: 0.3pt, upper(label)),
)

// --- cards -------------------------------------------------------------
// Coloured-border card for the L1/L2/L3 experiment slides.
// `height` lets call sites equalise card heights across a row.
#let level-card(accent, title, body, body-size: 15pt, height: auto, title-size: 17pt) = box(
  width: 100%, height: height, fill: card,
  stroke: 2pt + accent, radius: 6pt, inset: 11pt,
)[
  #text(fill: accent, weight: 700, size: title-size)[#title]
  #v(4pt)
  #text(size: body-size, fill: rgb("#333333"))[#body]
]

// TASK / BACKGROUND / DO NOT box. Sizes are tunable so a dense slide can
// pass a compact variant.
#let pattern-box(accent, label, body, label-size: 18pt, body-size: 16pt, pad: 9pt) = box(
  width: 100%, fill: card,
  stroke: 2pt + accent, radius: 6pt, inset: (x: 14pt, y: pad),
)[
  #text(fill: accent, weight: 700, size: label-size)[#label]
  #v(2pt)
  #text(size: body-size, fill: rgb("#333333"))[#body]
]

// Generic outlined card (closing thesis, etc.).
#let panel(body, accent: rgb("#1A3A6A"), fill: card) = box(
  width: 100%, fill: fill, stroke: 1.5pt + accent, radius: 6pt, inset: 14pt,
)[#body]

// --- layout helpers ----------------------------------------------------
#let twocol(left, right, gutter: 26pt) = grid(
  columns: (1fr, 1fr), gutter: gutter, left, right,
)

#let threecol(a, b, c, gutter: 16pt) = grid(
  columns: (1fr, 1fr, 1fr), gutter: gutter, a, b, c,
)

// Centred secondary caption under a figure or below body text.
#let caption(body, size: 16pt) = align(
  center, text(size: size, fill: muted)[#body],
)

// Left-bordered blockquote.
#let bquote(body) = box(
  width: 100%, fill: card, inset: (left: 22pt, rest: 11pt),
  stroke: (left: 5pt + rgb("#BDBDBD")),
)[#text(fill: rgb("#333333"))[#body]]

// 3-column figure grid. `items` is an array of (path, cap) dicts.
// Each thumbnail sits in a fixed-height bordered cell so rows stay aligned.
#let fig-grid(items, img-height: 3.5cm, caption-size: 12pt) = grid(
  columns: (1fr, 1fr, 1fr), gutter: 13pt, row-gutter: 9pt,
  ..items.map(it => box(width: 100%)[
    #box(width: 100%, height: img-height, stroke: 0.5pt + hair, radius: 2pt,
         clip: true, inset: 3pt)[
      #align(center + horizon, image(it.path, height: 100%))
    ]
    #v(3pt)
    #align(center, text(size: caption-size, fill: muted)[#it.cap])
  ]),
)
