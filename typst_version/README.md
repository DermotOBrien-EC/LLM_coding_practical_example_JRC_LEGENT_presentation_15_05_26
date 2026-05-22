# Typst version of the JRC C4 "Practical AI" decks

A [Typst](https://typst.app/) port of the two presentation decks in
`../slides/`, styled with a re-creation of the **Frankfurt** theme
(black section-navigation bar with per-slide progress dots, full-bleed
blue title bar, three-tone footer info line). Self-contained — no
external Typst packages, so it compiles offline.

## Contents

| Path | What it is |
|---|---|
| `talk_v2.typ` | 24-slide standalone deck — "Seven tips for working with AI". Port of `../slides/talk_v2.md` (Marp). |
| `jrc_section_v2.typ` | 8-slide JRC C4 drop-in section — "Practical Use Cases". Port of `../slides/jrc_section_v2_beamer.tex` (Beamer). |
| `jrc-theme.typ` | The shared Frankfurt theme — nav bar, title bar, footer, palette, and styled components. Imported by both decks. |
| `assets/` | The 9 figures and the EC emblem used by the decks, copied in so the folder is portable. |
| `talk_v2.pdf`, `jrc_section_v2.pdf` | Compiled decks — **committed; these are the canonical artifacts.** |
| `png/`, `svg/` | One image per slide. **Generated locally, not committed** (see below). |

### `png/` and `svg/` are local-only

`png/` and `svg/` are git-ignored — they are not in the repository. A
colleague who clones the branch gets the `.typ` source and the two PDFs.
To obtain per-slide images, either run the export commands below, or ask
the deck author for the rendered files directly.

The compiled `pdf` (and the locally-rendered `png`/`svg`) are the
canonical artifacts. They were rendered on macOS with Helvetica Neue; the
`.typ` sources fall back to Helvetica → Arial → Liberation Sans → DejaVu
Sans, so a recompile on Linux still works but spacing may shift slightly.

## Theme — Frankfurt

- **Section nav bar** (black, top): one column per section, each with the
  section name and a ●/○ dot per slide. `talk_v2` sections are
  *Setup · The experiment · More tips · Closing*; `jrc_section_v2` are
  *Tips · The experiment · Takeaways*.
- **Title bar**: full-bleed `#3333B3` blue, white title, optional
  wayfinding chip (e.g. "TIP 4").
- **Footer info line**: author · title · date + page number.
- Section ranges are declared in each deck's `frankfurt-deck(...)` call —
  if you add or remove a slide, update the page ranges there.

## Recompiling

Install the Typst CLI (any of these):

```sh
brew install typst                       # macOS
cargo install --locked typst-cli         # any platform with Rust
# or a release binary from https://github.com/typst/typst/releases
```

Then, from this directory:

```sh
typst compile talk_v2.typ                # -> talk_v2.pdf
typst compile jrc_section_v2.typ         # -> jrc_section_v2.pdf
```

Re-export per-slide images. `{0p}` is the page number zero-padded to the
width of *that deck's* total page count — the 24-page deck gets
`-01 … -24`, the 8-page deck gets `-1 … -8`:

```sh
typst compile talk_v2.typ        "png/talk_v2-{0p}.png"        --ppi 300
typst compile talk_v2.typ        "svg/talk_v2-{0p}.svg"        --format svg
typst compile jrc_section_v2.typ "png/jrc_section_v2-{0p}.png" --ppi 300
typst compile jrc_section_v2.typ "svg/jrc_section_v2-{0p}.svg" --format svg
```

Live preview while editing: `typst watch talk_v2.typ`.

## Page counts

- `talk_v2` — 24 pages.
- `jrc_section_v2` — 8 pages: 1 cover + 7 content frames. The footer keeps
  the parent-deck slot numbers **40 / 46 … 46 / 46** (the cover is
  unnumbered), so the section still drops into the JRC C4 parent deck.
