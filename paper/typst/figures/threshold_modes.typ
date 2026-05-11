// Threshold-mode comparison — six binary masks produced by applying
// each threshold mode to the same normalized response field $tilde(D)_t$
// from input_1 frame 352, plus a histogram of $tilde(D)_t$ inside the
// ROI with Otsu and Triangle threshold values annotated.
//
// Layout:
//   row 0: full-width histogram strip of $tilde(D)_t$ with vertical
//          lines at the Otsu and Triangle threshold values.
//   row 1: masks for otsu+offset, triangle+offset, manual ($tau_man = 64$).
//   row 2: masks for percentile ($p = 70$), adaptive-mean ($b = 21$,
//          $C = 10$), adaptive-gaussian ($b = 21$, $C = 10$).
//
// Compile:
//   typst compile paper/typst/figures/threshold_modes.typ \
//                 paper/typst/figures/threshold_modes.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet    = rgb("#73000A")
#let atlantic  = rgb("#466A9F")
#let horseshoe = rgb("#65780B")
#let b70       = rgb("#5C5C5C")
#let b10       = rgb("#ECECEC")
#let txt       = rgb("#000000")

#let hist-rows = csv("threshold_modes_panels/histogram.csv")
#let thr-rows  = csv("threshold_modes_panels/thresholds.csv")

#let parse-hist() = {
  let bins = ()
  let counts = ()
  for r in hist-rows.slice(1) {
    bins.push(float(r.at(0)))
    counts.push(float(r.at(1)))
  }
  (bins: bins, counts: counts)
}
#let hist = parse-hist()

#let parse-thr() = {
  let m = (:)
  for r in thr-rows.slice(1) {
    m.insert(r.at(0), float(r.at(1)))
  }
  m
}
#let thrs = parse-thr()

// 2-by-3 mask grid: panel labels + paths.
#let masks = (
  (
    ([Otsu + offset (integrated)],            "threshold_modes_panels/mask_otsu.png"),
    ([Triangle + offset (+16)],               "threshold_modes_panels/mask_triangle.png"),
    ([manual ($tau_("man") = 64$)],            "threshold_modes_panels/mask_manual.png"),
  ),
  (
    ([percentile ($p = 70$)],                  "threshold_modes_panels/mask_percentile.png"),
    ([adaptive-mean ($b = 1001$, $C = 15$)],     "threshold_modes_panels/mask_adaptive_mean.png"),
    ([adaptive-gaussian ($b = 1001$, $C = 20$)], "threshold_modes_panels/mask_adaptive_gaussian.png"),
  ),
)

#cetz.canvas({
  import cetz.draw: *

  // Mask grid geometry.
  let ch-w = 3.6
  let ch-h = ch-w * 9 / 16
  let ch-gap-x = 0.30
  let ch-gap-y = 0.42
  let label-h = 0.22
  let label-size = 6.5pt
  let label-pad = 0.08
  let row-block-h = ch-h + label-h + label-pad

  let grid-w = 3 * ch-w + 2 * ch-gap-x

  // Histogram strip sits above the mask grid, same width.
  let hist-h = 1.2
  let hist-gap = 0.40
  let hist-y0 = ch-gap-y + hist-h     // y of top of histogram
  let hist-y1 = ch-gap-y              // y of bottom of histogram
  let hist-x0 = 0
  let hist-x1 = grid-w

  // Mask-grid origin: first mask row tile-center y starts below the histogram.
  let grid-top-y = 0    // first row tile-center y is at grid-top-y - ch-h/2

  // --- Histogram strip ---
  let n = hist.bins.len()
  let y-max = calc.max(..hist.counts)

  // Frame.
  rect((hist-x0, hist-y1), (hist-x1, hist-y0),
       stroke: 0.6pt + b70, fill: none)

  // Bars.
  let bar-w-px = (hist-x1 - hist-x0) / n * 0.92
  let bar-pad = ((hist-x1 - hist-x0) / n - bar-w-px) / 2
  for j in range(n) {
    let xj = hist-x0 + j * (hist-x1 - hist-x0) / n + bar-pad
    let h = hist.counts.at(j) / y-max * hist-h
    rect((xj, hist-y1), (xj + bar-w-px, hist-y1 + h),
         fill: b70.lighten(40%), stroke: none)
  }

  // X-axis tick marks at 0, 64, 128, 192, 255.
  for t in (0, 64, 128, 192, 255) {
    let xt = hist-x0 + t / 255 * (hist-x1 - hist-x0)
    line((xt, hist-y1), (xt, hist-y1 - 0.05), stroke: 0.5pt + b70)
    content((xt, hist-y1 - 0.18),
            text(size: 7pt, fill: b70)[#t])
  }

  // Threshold annotations: vertical lines at Otsu and Triangle values.
  let otsu-x  = hist-x0 + thrs.at("otsu_raw")     / 255 * (hist-x1 - hist-x0)
  let tri-x   = hist-x0 + thrs.at("triangle_raw") / 255 * (hist-x1 - hist-x0)

  line((otsu-x, hist-y1), (otsu-x, hist-y0),
       stroke: (paint: txt, thickness: 1.4pt, dash: "dashed"))
  content((otsu-x, hist-y0 + 0.08),
          text(size: 7pt, fill: txt, weight: 600)[Otsu (#calc.round(thrs.at("otsu_raw"))) ],
          anchor: "south")

  line((tri-x, hist-y1), (tri-x, hist-y0),
       stroke: (paint: txt, thickness: 1.4pt, dash: "dashed"))
  content((tri-x, hist-y0 + 0.08),
          text(size: 7pt, fill: txt, weight: 600)[Triangle (#calc.round(thrs.at("triangle_raw")))],
          anchor: "south")

  // Y-axis label.
  content((-0.10, (hist-y0 + hist-y1) / 2),
          text(size: 7pt, fill: b70)[count],
          anchor: "east")
  // X-axis label.
  content(((hist-x0 + hist-x1) / 2, hist-y1 - 0.42),
          text(size: 7.5pt, fill: txt)[$tilde(D)_t$ value (8-bit)],
          anchor: "north")

  // --- Mask grid ---
  for (r, row) in masks.enumerate() {
    for (c, (label, path)) in row.enumerate() {
      let x-center = c * (ch-w + ch-gap-x) + ch-w / 2
      let tile-y = grid-top-y - r * (row-block-h + ch-gap-y) - ch-h / 2
      let label-y = tile-y - ch-h / 2 - label-pad - label-h / 2 - hist-h * 0  // align

      content(
        (x-center, tile-y),
        image(path, width: ch-w * 1cm, height: ch-h * 1cm),
      )
      rect(
        (x-center - ch-w / 2, tile-y + ch-h / 2),
        (x-center + ch-w / 2, tile-y - ch-h / 2),
        stroke: 0.6pt + txt,
        fill: none,
      )
      content(
        (x-center, label-y),
        text(size: label-size, fill: txt)[#label],
      )
    }
  }
})
