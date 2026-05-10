// Pipeline diagram — 14-stage VRIFA detection pipeline.
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
  stage("Colorspace",   "colorspace.rs",  atlantic),
  stage("ROI",          "roi.rs",         atlantic),
  stage("Stabilize",    "registration.rs", atlantic),
  stage("Pre-blur",     "delta::blur_frame", atlantic),
  stage("Reference",    "reference.rs",   atlantic),
  stage("Peak track",   "peak.rs",        atlantic),
  stage("Delta",        "delta::compute_delta", garnet),
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

  let cell-w = 2.4
  let cell-h = 0.95
  let gap = 0.45
  let row-pad = 0.6
  let cols = 7
  let rows = 2

  // Black-filled stealth arrowhead, reused on every connector.
  let arrow = (end: "stealth", fill: black, scale: 0.8)

  for (i, st) in stages.enumerate() {
    let col = calc.rem(i, cols)
    let row = int(i / cols)
    let x0 = col * (cell-w + gap)
    let y0 = -row * (cell-h + gap + row-pad)
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

  // Connect stages left-to-right within rows, then row-to-row.
  for i in range(stages.len() - 1) {
    let col = calc.rem(i, cols)
    let row = int(i / cols)
    let next-row = int((i + 1) / cols)

    let x0 = col * (cell-w + gap) + cell-w
    let y0 = -row * (cell-h + gap + row-pad) - cell-h / 2

    if next-row == row {
      // Within-row arrow: short horizontal hop into the next cell.
      line((x0, y0), (x0 + gap, y0),
           stroke: 0.8pt + black, mark: arrow)
    } else {
      // Wrap arrow: routed around the cell layout so it does not cut
      // through any row-1 cell. Six points, five segments:
      //   right out of Threshold, down into the inter-row gap,
      //   long left past the layout, down to row-1 mid, right into Morphology.
      let elbow-out = 0.55
      let elbow-back = 0.55
      // Center the horizontal segment between row 0's bottom and row 1's top.
      let row0-bottom = -cell-h
      let row1-top = -(cell-h + gap + row-pad)
      let intermediate-y = (row0-bottom + row1-top) / 2
      let entry-y = -next-row * (cell-h + gap + row-pad) - cell-h / 2
      line(
        (x0, y0),
        (x0 + elbow-out, y0),
        (x0 + elbow-out, intermediate-y),
        (-elbow-back, intermediate-y),
        (-elbow-back, entry-y),
        (0, entry-y),
        stroke: 0.8pt + black,
        mark: arrow,
      )
    }
  }

  // Family legend, centered horizontally under the cell layout.
  let families = (
    ("Ingest",   atlantic),
    ("Detect",   garnet),
    ("Clean",    horseshoe),
    ("Export",   warmgrey),
  )
  let swatch-w = 0.4
  let label-pad = 0.08    // gap between swatch right edge and label left edge
  let item-w = 1.3        // approximate full item width (swatch + pad + label)
  let leg-spacing = 1.55  // distance between consecutive item starts
  let layout-w = cols * cell-w + (cols - 1) * gap
  let legend-w = (families.len() - 1) * leg-spacing + item-w
  let leg-x0 = (layout-w - legend-w) / 2
  let leg-y = -2 * (cell-h + gap + row-pad) + 0.85
  for (i, (name, c)) in families.enumerate() {
    let x = leg-x0 + i * leg-spacing
    rect((x, leg-y), (x + swatch-w, leg-y - 0.32),
         fill: c.lighten(82%), stroke: 0.7pt + c)
    content((x + swatch-w + label-pad, leg-y - 0.16),
            text(size: 9pt, fill: b70)[#name],
            anchor: "west")
  }
})
