// Colorspace projection figure — four columns showing the same canonical
// input frame projected into the four working colorspaces supported by
// the pipeline, with a wet-vs-dry histogram strip below each column.
//
// Canonical frame: input_1.mp4 frame 352 (50% fill, hand-labeled).
// Per-column projection (with Viridis colormap on single-channel ones):
//   col0  raw BGR (true-color)
//   col1  CIELAB L* (integrated configuration's working channel)
//   col2  HSV V
//   col3  8-bit grayscale
//
// Panel PNGs are pre-rendered at 960x540 by build_colorspace_panels.py;
// histograms come from the matching CSVs. CeTZ adds vector text labels
// and the histogram strips so the figure stays text-searchable.
//
// Compile:
//   typst compile paper/typst/figures/colorspace_projection.typ \
//                 paper/typst/figures/colorspace_projection.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet   = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let b70      = rgb("#5C5C5C")
#let b50      = rgb("#A2A2A2")
#let b10      = rgb("#ECECEC")
#let txt      = rgb("#000000")

// Read the four histogram CSVs once at compile time.
#let hist-bgr   = csv("colorspace_panels/hist_bgr.csv")
#let hist-lab   = csv("colorspace_panels/hist_lab_l.csv")
#let hist-hsv   = csv("colorspace_panels/hist_hsv_v.csv")
#let hist-gray  = csv("colorspace_panels/hist_gray.csv")

#let columns = (
  (header: "raw BGR",        panel: "colorspace_panels/col0_bgr.png",   hist: hist-bgr),
  (header: "CIELAB " + $L^*$, panel: "colorspace_panels/col1_lab_l.png", hist: hist-lab),
  (header: "HSV V",          panel: "colorspace_panels/col2_hsv_v.png", hist: hist-hsv),
  (header: "grayscale",      panel: "colorspace_panels/col3_gray.png",  hist: hist-gray),
)

// Parse one CSV (list of rows, first row is header) into three lists:
// bin_centers, wet_density, dry_density. Each distribution is normalized
// to integrate to ~1 so the two shapes are directly comparable in spite
// of the imbalanced pixel counts at mid-fill.
#let parse-hist(rows) = {
  let bins = ()
  let wet = ()
  let dry = ()
  for r in rows.slice(1) {
    bins.push(float(r.at(0)))
    wet.push(float(r.at(1)))
    dry.push(float(r.at(2)))
  }
  (bins: bins, wet: wet, dry: dry)
}

#let column-data = columns.map(c => (
  header: c.header,
  panel: c.panel,
  hist: parse-hist(c.hist),
))

// Find a single y-max across all four histograms so they share a vertical
// scale and the relative magnitudes are honest.
#let column-max(c) = calc.max(..c.hist.wet, ..c.hist.dry)
#let y-max = calc.max(..column-data.map(column-max))

#cetz.canvas({
  import cetz.draw: *

  // Geometry, in CeTZ units (≈ cm).
  let tile-w = 4.5
  let tile-h = 2.53            // 4.5 * 540/960
  let gap = 0.15
  let hdr-h = 0.22             // column header band
  let hist-h = 1.4             // histogram strip height
  let hist-pad = 0.18          // padding between panel bottom and histogram top
  let bar-w-frac = 0.85        // bar width as fraction of bin width

  // Panel + header row.
  for (i, c) in column-data.enumerate() {
    let x0 = i * (tile-w + gap)

    // Column header band, centered above the panel.
    content(
      (x0 + tile-w / 2, hdr-h / 2),
      text(weight: 700, size: 10pt, fill: txt)[#c.header],
    )

    // Image panel.
    content(
      (x0 + tile-w / 2, -hdr-h - tile-h / 2),
      image(c.panel, width: tile-w * 1cm, height: tile-h * 1cm),
    )
  }

  // Histogram strip beneath each column, sharing a single y-scale.
  let hist-y0 = -hdr-h - tile-h - hist-pad
  let hist-y1 = hist-y0 - hist-h

  for (i, c) in column-data.enumerate() {
    let x0 = i * (tile-w + gap)
    let x1 = x0 + tile-w

    // Axis frame for the histogram (rectangle around the plot area).
    rect(
      (x0, hist-y0),
      (x1, hist-y1),
      stroke: 0.5pt + b70,
      fill: white,
    )

    // Stair-step outlines: each distribution is drawn as a piecewise
    // horizontal path that hops up/down at every bin boundary. Outlines
    // are used instead of filled bars because the wet and dry pixel
    // counts are wildly imbalanced at mid-fill (~6:1), so filled bars
    // hide the smaller distribution. Outlines keep both shapes visible.
    let n = c.hist.bins.len()
    let bin-w = tile-w / n

    let stair-points(series, color) = {
      let pts = ()
      pts.push((x0, hist-y1))
      for j in range(n) {
        let h = (series.at(j) / y-max) * hist-h
        let xl = x0 + j * bin-w
        let xr = xl + bin-w
        pts.push((xl, hist-y1 + h))
        pts.push((xr, hist-y1 + h))
      }
      pts.push((x0 + tile-w, hist-y1))
      line(..pts, stroke: 1.0pt + color, close: false, fill: none)
    }

    // Draw dry first (lighter weight) then wet on top so the wet line
    // is the foreground.
    stair-points(c.hist.dry, atlantic)
    stair-points(c.hist.wet, garnet)

    // X-axis tick marks at 0, 64, 128, 192, 255 (byte range).
    for (t, label) in ((0, "0"), (64, "64"), (128, "128"), (192, "192"), (255, "255")) {
      let xt = x0 + (t / 255) * tile-w
      line(
        (xt, hist-y1),
        (xt, hist-y1 - 0.05),
        stroke: 0.4pt + b70,
      )
      content(
        (xt, hist-y1 - 0.18),
        text(size: 6.5pt, fill: b70)[#label],
      )
    }
  }

  // Legend below the histogram strip.
  let legend-y = hist-y1 - 0.45
  let entries = (
    ("wet (GT inside ROI)", garnet),
    ("dry (GT inside ROI)", atlantic),
  )
  let swatch-w = 0.4
  let swatch-h = 0.2
  let pad = 0.12
  let item-w = 3.4
  let total-w = entries.len() * item-w
  let row-w = 4 * tile-w + 3 * gap
  let leg-x0 = (row-w - total-w) / 2

  for (k, (name, color)) in entries.enumerate() {
    let xa = leg-x0 + k * item-w
    rect(
      (xa, legend-y - swatch-h / 2),
      (xa + swatch-w, legend-y + swatch-h / 2),
      fill: color, stroke: 0.4pt + b70,
    )
    content(
      (xa + swatch-w + pad, legend-y),
      text(size: 9pt, fill: txt)[#name],
      anchor: "west",
    )
  }
})
