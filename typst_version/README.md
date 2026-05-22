# Typst version of the JRC C4 "Practical AI" decks

A [Typst](https://typst.app/) port of the two presentation decks in
`../slides/`. Self-contained — no external Typst packages, so it compiles
offline.

## Contents

| Path | What it is |
|---|---|
| `talk_v2.typ` | 24-slide standalone deck — "Seven tips for working with AI". Port of `../slides/talk_v2.md` (Marp). |
| `jrc_section_v2.typ` | 8-slide JRC C4 drop-in section — "Practical Use Cases". Port of `../slides/jrc_section_v2_beamer.tex` (Beamer). Footer numbers 40–46. |
| `jrc-theme.typ` | Shared theme — page setup, palette, fonts, and the styled components (cards, tip chips, pattern boxes, figure grid). Imported by both decks. |
| `assets/` | The 10 figures used by the decks, copied in so the folder is portable. |
| `talk_v2.pdf`, `jrc_section_v2.pdf` | Compiled decks. |
| `png/` | One PNG per slide, 300 ppi: `talk_v2-01.png … talk_v2-24.png` and `jrc_section_v2-1.png … jrc_section_v2-8.png`. |
| `svg/` | One SVG per slide — same names, vector, good for embedding in a web page. |

**The compiled `pdf` / `png` / `svg` files are the canonical artifacts.**
They were rendered on macOS with Helvetica Neue. The `.typ` sources fall
back to Helvetica → Arial → Liberation Sans → DejaVu Sans, so a recompile
on Linux still works but may shift spacing slightly if none of the
preferred fonts are installed.

## Recompiling

Install the Typst CLI (any of these):

```sh
brew install typst                       # macOS
cargo install --locked typst-cli         # any platform with Rust
# or download a release binary from https://github.com/typst/typst/releases
```

Then, from this directory:

```sh
typst compile talk_v2.typ                # -> talk_v2.pdf
typst compile jrc_section_v2.typ         # -> jrc_section_v2.pdf
```

Re-export per-slide images. `{0p}` is the page number zero-padded to the
width of *that deck's* total page count — so the 24-page deck gets
`-01 … -24` and the 8-page deck gets `-1 … -8`:

```sh
typst compile talk_v2.typ        "png/talk_v2-{0p}.png"        --ppi 300
typst compile talk_v2.typ        "svg/talk_v2-{0p}.svg"        --format svg
typst compile jrc_section_v2.typ "png/jrc_section_v2-{0p}.png" --ppi 300
typst compile jrc_section_v2.typ "svg/jrc_section_v2-{0p}.svg" --format svg
```

Live preview while editing: `typst watch talk_v2.typ`.

## Page counts

- `talk_v2` — 24 pages.
- `jrc_section_v2` — 8 pages: 1 plain section divider (no footer) + 7
  content frames numbered 40–46.
