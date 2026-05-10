// Teaser figure — 3 rows (25%, 50%, 95% fill of input_1) by 4 columns
// (raw input, Lekanidis 2020 + GT, Almazán-Lázaro 2022 + GT, Ours + GT).
//
// Compile:
//   typst compile paper/typst/figures/teaser.typ paper/typst/figures/teaser.pdf
//
// Panel PNGs are pre-rendered at 960x540 by _dev tooling (see
// build_teaser_panels.py); CeTZ adds vector text labels and the
// bottom legend so the figure stays text-searchable in the final PDF.

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let lek-color    = rgb("#00FF00")     // bright green
#let alm-color    = rgb("#FF00FF")     // bright magenta
#let ours-color   = rgb("#00FFFF")     // bright cyan
#let gt-color     = rgb("#FFFFFF")     // white
#let bg           = rgb("#000000")     // black backdrop for boundary panels
#let b70          = rgb("#5C5C5C")
#let b90          = rgb("#363636")

// Pre-rotate the row labels here so the typst built-in rotate() is in
// scope. Inside cetz.canvas, `rotate` would shadow to cetz.draw.rotate.
#let row-label-content = name => rotate(-90deg, reflow: true,
  text(weight: 700, size: 11pt, fill: b70)[#name])

#let row-labels-rotated = (
  row-label-content("25 % fill"),
  row-label-content("50 % fill"),
  row-label-content("95 % fill"),
)

#cetz.canvas({
  import cetz.draw: *

  // Layout, in CeTZ units (1 unit ≈ 1 cm at default scale).
  let tile-w = 4.5
  let tile-h = 2.53     // 4.5 * 540/960
  let gap    = 0.1
  let col-label-h = 0.18
  let row-label-w = 0.8
  let legend-h = 0.6

  let columns = (
    "input frame",
    "Lekanidis & Vosniakos 2020",
    "Almazán-Lázaro 2022",
    "Ours (this work)",
  )
  let row-labels = ("25 % fill", "50 % fill", "95 % fill")

  // Column headers: along the top of the grid.
  for (c, name) in columns.enumerate() {
    let x = row-label-w + c * (tile-w + gap) + tile-w / 2
    content((x, col-label-h / 2),
            text(weight: 700, size: 10pt, fill: b90)[#name])
  }

  // Rows of panels.
  for (r, label) in row-labels.enumerate() {
    let y = -col-label-h - r * (tile-h + gap) - tile-h / 2

    // Row label, rotated 90 degrees CCW so it reads up the page.
    let _ = label
    content((row-label-w / 2, y), row-labels-rotated.at(r))

    // Four panels.
    for c in range(columns.len()) {
      let x = row-label-w + c * (tile-w + gap) + tile-w / 2
      content((x, y),
              image("teaser_panels/row" + str(r) + "_col" + str(c) + ".png",
                    width: tile-w * 1cm, height: tile-h * 1cm))
    }
  }

  // Bottom legend: swatch + short label per pipeline. Full citations
  // already live in the column headers above.
  let legend-y = -col-label-h - 3 * (tile-h + gap) - legend-h / 2 - 0.05
  let entries = (
    ("GT polygon",        gt-color),
    ("Lekanidis 2020",    lek-color),
    ("Almazán 2022",      alm-color),
    ("Ours",              ours-color),
  )
  let swatch-w = 0.5
  let swatch-h = 0.22
  let label-pad = 0.15
  let item-w = 3.0      // 0.5 swatch + 0.15 pad + ~2.0 label + buffer
  let total-w = entries.len() * item-w
  let layout-w = row-label-w + 4 * tile-w + 3 * gap
  let leg-x0 = (layout-w - total-w) / 2

  for (i, (name, color)) in entries.enumerate() {
    let xa = leg-x0 + i * item-w
    rect((xa, legend-y - swatch-h / 2),
         (xa + swatch-w, legend-y + swatch-h / 2),
         fill: color, stroke: 0.5pt + b70)
    content((xa + swatch-w + label-pad, legend-y),
            text(size: 9.5pt, fill: b90, weight: 600)[#name],
            anchor: "west")
  }
})
