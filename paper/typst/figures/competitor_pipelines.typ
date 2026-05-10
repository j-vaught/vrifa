// Competitor-pipeline comparison — three-row per-frame stage chain
// for Lekanidis-Vosniakos 2020, Almazán-Lázaro 2022, and the
// integrated pipeline of this work.
//
// Compile:
//   typst compile paper/typst/figures/competitor_pipelines.typ \
//                 paper/typst/figures/competitor_pipelines.pdf
//
// Color rules:
//   Garnet outline   = stage present only in the integrated pipeline.
//   Atlantic outline = stage present only in prior work, no analogue here.
//   Neutral outline  = stage shared across pipelines or with a conceptual
//                      analogue in this work.
//
// Brand palette consistent with pipeline.typ. No rounded edges.

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 8pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet   = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let b70      = rgb("#5C5C5C")
#let b50      = rgb("#A2A2A2")
#let b10      = rgb("#ECECEC")

// Per-stage record: (label, kind).
//   kind = "garnet"   → unique to integrated pipeline
//   kind = "atlantic" → unique to prior pipeline, no analogue here
//   kind = "neutral"  → shared or analogue
#let s(label, kind) = (label: label, kind: kind)

// Row 1 — Lekanidis & Vosniakos 2020 (IJMMS). 11 stages.
#let lekanidis = (
  s("Decode",         "neutral"),
  s("ROI crop",       "neutral"),
  s("Gauss blur",     "neutral"),
  s("Grayscale",      "neutral"),
  s("Contrast str.",  "neutral"),
  s("Otsu",           "neutral"),
  s("FG swap",        "neutral"),
  s("Close d13",      "neutral"),
  s("Sobel edge",     "neutral"),
  s("Open A>120",     "neutral"),
  s("Dilation",       "neutral"),
)

// Row 2 — Almazán-Lázaro 2022 (J Manuf Processes). 11 stages.
#let almazan = (
  s("Decode",         "neutral"),
  s("ROI crop",       "neutral"),
  s("Scaramuzza",     "atlantic"),
  s("Hist. eq.",      "neutral"),
  s("Abs diff F0",    "neutral"),
  s("Grayscale",      "neutral"),
  s("Mean 5×5",       "neutral"),
  s("Sobel grad.",    "neutral"),
  s("Erode",          "neutral"),
  s("Dilate",         "neutral"),
  s("Small-area",     "neutral"),
)

// Row 3 — Integrated (this work). 12 detection-path stages.
#let integrated = (
  s("Colorspace",     "neutral"),
  s("ROI",            "neutral"),
  s("Stabilize",      "garnet"),
  s("Pre-blur",       "neutral"),
  s("Reference",      "garnet"),
  s("Peak track",     "garnet"),
  s("Delta (clip)",   "garnet"),
  s("Post-blur",      "garnet"),
  s("Threshold",      "neutral"),
  s("Morph close",    "neutral"),
  s("Open+area",      "garnet"),
  s("Lock",           "garnet"),
)

#let rows = (
  (name: "Lekanidis 2020",        stages: lekanidis),
  (name: "Almazán-Lázaro 2022",   stages: almazan),
  (name: "Integrated (this work)", stages: integrated),
)

#let stroke-color-for(kind) = {
  if kind == "garnet" { garnet }
  else if kind == "atlantic" { atlantic }
  else { b70 }
}

#let fill-color-for(kind) = {
  if kind == "garnet" { garnet.lighten(85%) }
  else if kind == "atlantic" { atlantic.lighten(85%) }
  else { b10 }
}

#cetz.canvas({
  import cetz.draw: *

  let cell-w = 1.55
  let cell-h = 0.85
  let gap = 0.22
  let row-gap = 0.55

  // Label column on the left.
  let label-w = 2.6
  let label-pad = 0.25

  // Max cells across rows; used to right-pad shorter rows for legend math.
  let max-cells = calc.max(..rows.map(r => r.stages.len()))
  let row-width(n) = n * cell-w + (n - 1) * gap
  let max-row-w = row-width(max-cells)

  let arrow = (end: "stealth", fill: black, scale: 0.7)

  // Draw each row.
  for (ri, row) in rows.enumerate() {
    let n = row.stages.len()
    let y0 = -ri * (cell-h + row-gap)
    let y-mid = y0 - cell-h / 2

    // Row label, right-aligned in the label column.
    content(
      (label-w - label-pad, y-mid),
      text(size: 9pt, weight: 600, fill: black)[#row.name],
      anchor: "east",
    )

    // Cells.
    for (ci, st) in row.stages.enumerate() {
      let x0 = label-w + ci * (cell-w + gap)
      let x1 = x0 + cell-w
      let y1 = y0 - cell-h

      rect(
        (x0, y0), (x1, y1),
        fill: fill-color-for(st.kind),
        stroke: (
          if st.kind == "garnet" { 1.0pt + garnet }
          else if st.kind == "atlantic" { 1.0pt + atlantic }
          else { 0.6pt + b70 }
        ),
      )

      content(
        (x0 + cell-w / 2, y0 - cell-h / 2),
        text(
          size: 7.8pt,
          weight: if st.kind == "neutral" { 400 } else { 700 },
          fill: if st.kind == "neutral" { black } else { stroke-color-for(st.kind) },
        )[#st.label],
      )
    }

    // Connector arrows between cells in this row.
    for ci in range(n - 1) {
      let x0 = label-w + ci * (cell-w + gap) + cell-w
      line(
        (x0, y-mid),
        (x0 + gap, y-mid),
        stroke: 0.7pt + black,
        mark: arrow,
      )
    }
  }

  // Legend below the last row.
  let legend-y = -rows.len() * (cell-h + row-gap) + row-gap - 0.05
  let legend-items = (
    (text: "Unique to this work",       kind: "garnet"),
    (text: "Unique to prior, no analogue", kind: "atlantic"),
    (text: "Shared or analogue",        kind: "neutral"),
  )
  let swatch-w = 0.55
  let swatch-h = 0.32
  let item-gap = 0.18
  let item-text-pad = 0.14
  let item-widths = (3.0, 4.4, 2.8)
  let total-legend-w = item-widths.sum() + (legend-items.len() - 1) * 0.6
  let legend-x0 = label-w + (max-row-w - total-legend-w) / 2

  let cursor = legend-x0
  for (i, item) in legend-items.enumerate() {
    rect(
      (cursor, legend-y),
      (cursor + swatch-w, legend-y - swatch-h),
      fill: fill-color-for(item.kind),
      stroke: (
        if item.kind == "garnet" { 1.0pt + garnet }
        else if item.kind == "atlantic" { 1.0pt + atlantic }
        else { 0.6pt + b70 }
      ),
    )
    content(
      (cursor + swatch-w + item-text-pad, legend-y - swatch-h / 2),
      text(size: 8pt, fill: b70)[#item.text],
      anchor: "west",
    )
    cursor = cursor + item-widths.at(i) + 0.6
  }
})
