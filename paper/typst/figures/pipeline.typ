// Pipeline diagram — 16-stage VRIFA detection pipeline.
//
// Compile:
//   typst compile paper/typst/figures/pipeline.typ paper/typst/figures/pipeline.pdf
//
// Brand palette: garnet, atlantic, horseshoe, warm grey, b70.
// No rounded edges. High contrast.

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 8pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet    = rgb("#73000A")
#let atlantic  = rgb("#466A9F")
#let horseshoe = rgb("#65780B")
#let warmgrey  = rgb("#676156")
#let b70       = rgb("#5C5C5C")
#let b50       = rgb("#A2A2A2")
#let b10       = rgb("#ECECEC")

#let stage(label, code, color) = (label: label, code: code, color: color)

// Four families: ingest (atlantic), detect (garnet), clean (horseshoe), export (warmgrey).
#let stages = (
  // Row 1: ingest (6).
  stage("Colorspace",   "colorspace.rs",   atlantic),
  stage("ROI",          "roi.rs",          atlantic),
  stage("Stabilize",    "registration.rs", atlantic),
  stage("Pre-blur",     "blur::frame",     atlantic),
  stage("Reference",    "reference.rs",    atlantic),
  stage("Peak track",   "peak.rs",         atlantic),
  // Row 2: detect + clean (6).
  stage("Delta",        "delta::compute",  garnet),
  stage("Post-blur",    "blur::plane",     garnet),
  stage("Threshold",    "threshold.rs",    garnet),
  stage("Morph close",  "morphology.rs",   horseshoe),
  stage("Morph open",   "morphology.rs",   horseshoe),
  stage("Lock",         "lock.rs",         horseshoe),
  // Row 3: export (4).
  stage("Overlay",      "overlay.rs",      warmgrey),
  stage("Heatmap",      "heatmap.rs",      warmgrey),
  stage("Contours",     "contours.rs",     warmgrey),
  stage("Sampling",     "sampling.rs",     warmgrey),
)

#cetz.canvas({
  import cetz.draw: *

  let cell-w = 2.2
  let cell-h = 0.95
  let gap = 0.4
  let row-pad = 0.6
  let row-layout = (6, 6, 4)

  // Black-filled stealth arrowhead, reused on every connector.
  let arrow = (end: "stealth", fill: black, scale: 0.8)

  // Row width (units) given the cells in that row.
  let row-widths = row-layout.map(n => n * cell-w + (n - 1) * gap)
  let max-row-width = calc.max(..row-widths)

  // x offset that centers a row of `r` cells inside the widest row.
  let row-x(r) = (max-row-width - row-widths.at(r)) / 2

  // Map stage index i to (row, col) within the (6, 6, 4) layout.
  let stage-pos(i) = {
    let acc = 0
    let result = (row: 0, col: 0)
    for r in range(row-layout.len()) {
      let n = row-layout.at(r)
      if i >= acc and i < acc + n {
        result = (row: r, col: i - acc)
      }
      acc = acc + n
    }
    result
  }

  // Cell rendering.
  for (i, st) in stages.enumerate() {
    let pos = stage-pos(i)
    let x0 = row-x(pos.row) + pos.col * (cell-w + gap)
    let y0 = -pos.row * (cell-h + gap + row-pad)
    let x1 = x0 + cell-w
    let y1 = y0 - cell-h

    rect((x0, y0), (x1, y1),
         fill: st.color.lighten(82%),
         stroke: 0.7pt + st.color)

    content((x0 + cell-w / 2, y0 - 0.3),
            text(weight: 700, size: 10pt, fill: black)[#st.label])
    content((x0 + cell-w / 2, y0 - 0.62),
            text(size: 7.5pt, fill: black, font: "Menlo")[#st.code])
  }

  // Connector arrows: within-row hops, plus inter-row wraps.
  for i in range(stages.len() - 1) {
    let pos = stage-pos(i)
    let next-pos = stage-pos(i + 1)

    if next-pos.row == pos.row {
      // Within-row arrow: short horizontal hop into the next cell.
      let x0 = row-x(pos.row) + pos.col * (cell-w + gap) + cell-w
      let y0 = -pos.row * (cell-h + gap + row-pad) - cell-h / 2
      line((x0, y0), (x0 + gap, y0),
           stroke: 0.8pt + black, mark: arrow)
    } else {
      // Wrap arrow: out of last cell of source row, down to the
      // mid-line between rows, across to just left of the target
      // row's first cell, down again, then right into the target.
      let elbow-out = 0.55
      let elbow-back = 0.55
      let exit-x = row-x(pos.row) + pos.col * (cell-w + gap) + cell-w
      let exit-y = -pos.row * (cell-h + gap + row-pad) - cell-h / 2
      let entry-x = row-x(next-pos.row) + next-pos.col * (cell-w + gap)
      let entry-y = -next-pos.row * (cell-h + gap + row-pad) - cell-h / 2
      let source-bottom = -pos.row * (cell-h + gap + row-pad) - cell-h
      let target-top = -next-pos.row * (cell-h + gap + row-pad)
      let mid-y = (source-bottom + target-top) / 2

      line(
        (exit-x, exit-y),
        (exit-x + elbow-out, exit-y),
        (exit-x + elbow-out, mid-y),
        (entry-x - elbow-back, mid-y),
        (entry-x - elbow-back, entry-y),
        (entry-x, entry-y),
        stroke: 0.8pt + black,
        mark: arrow,
      )
    }
  }

  // Family legend, centered horizontally below the last row.
  let families = (
    ("Ingest", atlantic),
    ("Detect", garnet),
    ("Clean",  horseshoe),
    ("Export", warmgrey),
  )
  let swatch-w = 0.4
  let label-pad = 0.08
  let item-w = 1.3
  let leg-spacing = 1.55
  let legend-w = (families.len() - 1) * leg-spacing + item-w
  let leg-x0 = (max-row-width - legend-w) / 2
  let leg-y = -row-layout.len() * (cell-h + gap + row-pad) + 0.85
  for (i, (name, c)) in families.enumerate() {
    let x = leg-x0 + i * leg-spacing
    rect((x, leg-y), (x + swatch-w, leg-y - 0.32),
         fill: c.lighten(82%), stroke: 0.7pt + c)
    content((x + swatch-w + label-pad, leg-y - 0.16),
            text(size: 9pt, fill: b70)[#name],
            anchor: "west")
  }
})
