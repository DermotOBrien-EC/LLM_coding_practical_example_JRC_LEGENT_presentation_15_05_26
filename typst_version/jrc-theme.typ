// Frankfurt theme for the JRC C4 "Practical AI" decks.
// A Typst re-creation of the Beamer/Slidev "Frankfurt" theme:
//   - black section-navigation bar with per-slide progress dots
//   - full-bleed blue title bar
//   - three-tone footer info line (author / title / date + page)
// Reference: github.com/MuTsunTsai/slidev-theme-frankfurt
// No external packages — compiles offline.

// --- fonts -------------------------------------------------------------
#let sans = ("Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans")
#let mono = ("Menlo", "SF Mono", "DejaVu Sans Mono", "Liberation Mono")

// --- Frankfurt palette -------------------------------------------------
#let fblue   = rgb("#3333B3")  // primary  — title bar, footer col 3, cover box
#let fmid    = rgb("#262686")  // mid      — Item header, footer col 2, emphasis
#let fdark   = rgb("#191959")  // darkest  — footer col 1
#let fitem   = rgb("#E9E9F3")  // light lavender — Item body
#let navbg   = black
#let navoff  = rgb("#A8A29E")  // inactive section text (stone)

// --- semantic palette (kept — these colours carry meaning) -------------
#let l1     = rgb("#7F7F7F")  // beginner   — grey
#let l2     = rgb("#D97706")  // average    — amber
#let l3     = rgb("#FF7F0E")  // research   — orange
#let task   = rgb("#1F77B4")  // TASK       — blue
#let bg     = rgb("#2CA02C")  // BACKGROUND — green
#let dont   = rgb("#D62728")  // DO NOT     — red
#let card   = rgb("#FAFAFA")  // card fill
#let ink    = rgb("#1A1A1A")  // body text
#let muted  = rgb("#555555")  // captions / secondary
#let faint  = rgb("#8A8A8A")  // tertiary
#let hair   = rgb("#E0E0E0")  // thin borders

// --- layout metrics ----------------------------------------------------
#let nav-h   = 12.5mm   // top navigation bar
#let foot-h  = 7mm      // bottom info line
#let title-h = 16mm     // blue title bar
#let pad-x   = 15mm     // content side padding

// --- navigation bar ----------------------------------------------------
// `specs` is an array of (name, first-page, last-page) triples.
#let nav-bar(specs) = context {
  let p = counter(page).get().first()
  block(width: 100%, height: nav-h, fill: navbg, inset: (x: 9mm))[
    #grid(
      columns: (1fr,) * specs.len(),
      align: center + horizon,
      ..specs.map(s => {
        let (name, first, last) = s
        let active = p >= first and p <= last
        let col = if active { white } else { navoff }
        stack(
          spacing: 3.5pt,
          text(size: 11.5pt, fill: col, weight: if active { 600 } else { 400 })[#name],
          text(size: 10pt, fill: col)[
            #range(first, last + 1).map(pg => if pg == p { "●" } else { "○" }).join("  ")
          ],
        )
      }),
    )
  ]
}

// --- footer info line --------------------------------------------------
// Displayed page number = physical page + `page-offset`; hidden on the cover.
#let info-line(author, doc-title, date, total, page-offset) = context {
  let p = counter(page).get().first()
  let cell(bgcol, body) = box(width: 100%, height: 100%, fill: bgcol, inset: (x: 8pt))[
    #set text(fill: white, size: 11pt)
    #set align(center + horizon)
    #body
  ]
  grid(
    columns: (1fr, 1fr, 1fr), rows: foot-h,
    cell(fdark)[#author],
    cell(fmid)[#doc-title],
    box(width: 100%, height: 100%, fill: fblue, inset: (x: 11pt))[
      #set text(fill: white, size: 11pt)
      #place(center + horizon)[#date]
      #if p > 1 { place(right + horizon)[#(p + page-offset) / #total] }
    ],
  )
}

// --- slide scaffold ----------------------------------------------------
// Full-bleed blue title bar (optional wayfinding chip) + soft shadow +
// padded content body.
#let slide(title, body, tag: none, title-size: 24pt) = {
  pagebreak(weak: true)
  block(width: 100%, height: title-h, fill: fblue, inset: (left: pad-x, right: 12mm))[
    #set align(horizon)
    #if tag != none {
      box(fill: white, inset: (x: 8pt, y: 4pt), radius: 3pt,
        text(fill: fblue, weight: 700, size: 12pt, tracking: 0.4pt, upper(tag)))
      h(12pt)
    }
    #text(size: title-size, fill: white, weight: 600)[#title]
  ]
  block(width: 100%, height: 2.2mm, fill: gradient.linear(
    fblue.transparentize(45%), white, angle: 90deg))
  pad(left: pad-x, right: pad-x, top: 9pt, bottom: 4pt)[#body]
}

// Cover / section-divider layout: rounded blue title box, centred.
#let cover(title: "", subtitle: none, meta: none, logo: none) = align(center + horizon)[
  #box(fill: fblue, inset: (x: 40pt, y: 22pt), radius: 12pt, stroke: none)[
    #text(size: 38pt, fill: white, weight: 700)[#title]
  ]
  #if subtitle != none {
    v(20pt)
    text(size: 21pt, fill: muted)[#subtitle]
  }
  #if meta != none {
    v(38pt)
    text(size: 15pt, fill: faint)[#meta]
  }
  #if logo != none {
    v(26pt)
    logo
  }
]

// --- document wrapper --------------------------------------------------
#let frankfurt-deck(
  sections: (), author: "", doc-title: "", date: "",
  total: 1, page-offset: 0, body,
) = {
  set page(
    paper: "presentation-16-9",
    margin: (left: 0pt, right: 0pt, top: nav-h, bottom: foot-h),
    header-ascent: 0pt,
    footer-descent: 0pt,
    header: nav-bar(sections),
    footer: info-line(author, doc-title, date, total, page-offset),
  )
  set text(font: sans, size: 18pt, fill: ink)
  set par(leading: 0.62em, spacing: 0.9em, justify: false)
  // Zero automatic block spacing — every gap in a slide is an explicit v().
  // Keeps slide heights predictable so the page/section mapping holds.
  set block(spacing: 0pt)
  set list(marker: text(fill: fblue)[▪], spacing: 0.85em, indent: 3pt, tight: false)
  set enum(spacing: 0.85em, indent: 3pt, tight: false)
  show raw: set text(font: mono)
  show raw.where(block: true): it => box(
    width: 100%, fill: rgb("#F1F1F7"), inset: (x: 12pt, y: 10pt), radius: 5pt,
    text(size: 13pt, it),
  )
  show raw.where(block: false): it => box(
    fill: rgb("#F1F1F7"), inset: (x: 4pt, y: 1pt), radius: 3pt,
    text(size: 0.92em, it),
  )
  show strong: set text(fill: fmid)
  body
}

// --- chips -------------------------------------------------------------
#let tip-tag(label, accent: fblue) = box(
  fill: accent, inset: (x: 11pt, y: 5pt), radius: 3pt,
  text(fill: white, weight: 600, size: 14pt, tracking: 0.5pt, upper(label)),
)

#let subtip-tag(label, accent: fblue) = box(
  fill: fitem, inset: (x: 10pt, y: 4pt), radius: 3pt,
  text(fill: accent, weight: 600, size: 12pt, tracking: 0.3pt, upper(label)),
)

// --- cards -------------------------------------------------------------
// Coloured-border card for the L1/L2/L3 slides.
#let level-card(accent, title, body, body-size: 14pt, height: auto, title-size: 16pt) = box(
  width: 100%, height: height, fill: card,
  stroke: 2pt + accent, radius: 5pt, inset: 10pt,
)[
  #text(fill: accent, weight: 700, size: title-size)[#title]
  #v(4pt)
  #text(size: body-size, fill: rgb("#333333"))[#body]
]

// TASK / BACKGROUND / DO NOT box.
#let pattern-box(accent, label, body, label-size: 17pt, body-size: 15pt, pad: 8pt) = box(
  width: 100%, fill: card,
  stroke: 2pt + accent, radius: 5pt, inset: (x: 13pt, y: pad),
)[
  #text(fill: accent, weight: 700, size: label-size)[#label]
  #v(2pt)
  #text(size: body-size, fill: rgb("#333333"))[#body]
]

// Frankfurt Item box — dark-blue header, light-lavender body.
#let item-box(title, body, body-size: 15pt) = box(
  width: 100%, radius: 5pt, clip: true, stroke: none,
)[
  #block(width: 100%, fill: fmid, inset: (x: 12pt, y: 6pt))[
    #text(fill: white, weight: 700, size: 16pt)[#title]
  ]
  #block(width: 100%, fill: fitem, inset: 12pt)[
    #text(size: body-size)[#body]
  ]
]

// Generic outlined panel.
#let panel(body, accent: fblue, fill: fitem) = box(
  width: 100%, fill: fill, stroke: 1.5pt + accent, radius: 5pt, inset: 13pt,
)[#body]

// --- layout helpers ----------------------------------------------------
#let twocol(left, right, gutter: 24pt) = grid(
  columns: (1fr, 1fr), gutter: gutter, left, right,
)

#let caption(body, size: 15pt) = align(
  center, text(size: size, fill: muted)[#body],
)

#let bquote(body) = box(
  width: 100%, fill: fitem, inset: (left: 20pt, rest: 10pt),
  stroke: (left: 5pt + fblue),
)[#text(fill: rgb("#333333"))[#body]]

// 3-column figure grid; each thumbnail in a fixed-height bordered cell.
#let fig-grid(items, img-height: 3.3cm, caption-size: 11pt) = grid(
  columns: (1fr, 1fr, 1fr), gutter: 12pt, row-gutter: 8pt,
  ..items.map(it => box(width: 100%)[
    #box(width: 100%, height: img-height, stroke: 0.5pt + hair, radius: 2pt,
         clip: true, inset: 3pt)[
      #align(center + horizon, image(it.path, height: 100%))
    ]
    #v(2pt)
    #align(center, text(size: caption-size, fill: muted)[#it.cap])
  ]),
)
