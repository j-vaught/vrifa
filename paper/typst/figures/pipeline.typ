// Pipeline diagram — 12-stage VRIFA detection pipeline.
//
// Compile:
//   typst compile paper/typst/figures/pipeline.typ paper/typst/figures/pipeline.pdf
//
// Brand palette: garnet, atlantic, horseshoe, warm grey, b70.
// No rounded edges. High contrast.

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 8pt, fill: white)
#set text(font: ("Inter", "Helvetica"), size: 9pt)

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
  stage("Colorspace",   "colorspace.rs",  atlantic),
  stage("ROI",          "roi.rs",         atlantic),
  stage("Reference",    "reference.rs",   atlantic),
  stage("Peak track",   "peak.rs",        atlantic),
  stage("Delta",        "delta.rs",       garnet),
  stage("Threshold",    "threshold.rs",   garnet),
  stage("Morphology",   "morphology.rs",  horseshoe),
  stage("Lock",         "lock.rs",        horseshoe),
  stage("Overlay",      "overlay.rs",     warmgrey),
  stage("Heatmap",      "heatmap.rs",     warmgrey),
  stage("Contours",     "contours.rs",    warmgrey),
  stage("Sampling",     "sampling.rs",    warmgrey),
)

#cetz.canvas({
  import cetz.draw: *

  let cell-w = 2.6
  let cell-h = 0.95
  let gap = 0.12
  let cols = 6
  let rows = 2

  for (i, st) in stages.enumerate() {
    let col = calc.rem(i, cols)
    let row = int(i / cols)
    let x0 = col * (cell-w + gap)
    let y0 = -row * (cell-h + gap + 0.7)
    let x1 = x0 + cell-w
    let y1 = y0 - cell-h

    rect((x0, y0), (x1, y1),
         fill: st.color.lighten(82%),
         stroke: 0.7pt + st.color)

    content((x0 + cell-w / 2, y0 - 0.3),
            text(weight: 700, size: 10pt, fill: st.color)[#st.label])
    content((x0 + cell-w / 2, y0 - 0.62),
            text(size: 7.5pt, fill: b70, font: "Menlo")[#st.code])
  }

  // Connect stages left-to-right within rows, then row-to-row.
  for i in range(stages.len() - 1) {
    let col = calc.rem(i, cols)
    let row = int(i / cols)
    let next-col = calc.rem(i + 1, cols)
    let next-row = int((i + 1) / cols)

    let x0 = col * (cell-w + gap) + cell-w
    let y0 = -row * (cell-h + gap + 0.7) - cell-h / 2
    let x1 = next-col * (cell-w + gap)
    let y1 = -next-row * (cell-h + gap + 0.7) - cell-h / 2

    if next-row == row {
      line((x0, y0), (x0 + gap, y0), stroke: 0.7pt + b70, mark: (end: ">"))
    } else {
      // Wrap from end of row 0 down to start of row 1.
      let mid-x = x0 + 0.4
      let above-y = y0
      let below-y = y1
      line((x0, above-y), (mid-x, above-y), stroke: 0.7pt + b70)
      line((mid-x, above-y), (mid-x, below-y), stroke: 0.7pt + b70)
      line((mid-x, below-y), (x1 - cell-w * cols - gap * (cols - 1), below-y),
           stroke: 0.7pt + b70)
      // Simpler: draw the wrap as one line straight to the start of the next row.
      let start-x = 0
      line((mid-x, below-y), (start-x, below-y), stroke: 0.7pt + b70, mark: (end: ">"))
    }
  }

  // Family legend underneath.
  let leg-y = -2 * (cell-h + gap + 0.7) + 0.2
  let leg-x = 0
  let families = (
    ("Ingest",   atlantic),
    ("Detect",   garnet),
    ("Clean",    horseshoe),
    ("Export",   warmgrey),
  )
  for (i, (name, c)) in families.enumerate() {
    let x = i * 3.5
    rect((x, leg-y), (x + 0.5, leg-y - 0.35),
         fill: c.lighten(82%), stroke: 0.7pt + c)
    content((x + 1.4, leg-y - 0.18),
            text(size: 9pt, fill: b70)[#name])
  }
})
