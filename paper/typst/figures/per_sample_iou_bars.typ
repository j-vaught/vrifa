// Per-sample IoU bar chart for fig:per_sample_iou_bars in results.typ.
// One row per sample, sorted ascending by mean mask IoU.
// Bars: atlantic for 1080p clips, warmgrey for 524p cropped operator-view.
// Whisker = bootstrap 95% CI. Vertical garnet rule = overall mean IoU.
//
// Compile:
//   typst compile paper/typst/figures/per_sample_iou_bars.typ
//                 paper/typst/figures/per_sample_iou_bars.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet    = rgb("#73000A")
#let atlantic  = rgb("#466A9F")
#let warmgrey  = rgb("#676156")
#let black     = rgb("#000000")
#let b30       = rgb("#C7C7C7")

// Per-sample integrated-config mean IoU values from
// tab:agreement_per_sample. CI half-widths computed via bootstrap
// over 5 frames per sample (10000 resamples, seed=0); std/CI inflated
// for small n. Values are conservative.
//
// Resolution bucket: 1080p (input_1..3) -> atlantic, 524p (input_4..11)
// -> warmgrey, per the dataset section's resolution tiering.
// Per-sample integrated-config IoU from agreement_metrics_integrated.json
// (lock_frames=3, morph_kernel=13 default). The "ci" column here is
// the half-width of the per-sample 95% bootstrap CI over 5 frames.
#let rows = (
  ("input_1",  0.748, 0.273, atlantic),
  ("input_2",  0.960, 0.023, atlantic),
  ("input_3",  0.970, 0.012, atlantic),
  ("input_4",  0.601, 0.184, warmgrey),
  ("input_5",  0.674, 0.180, warmgrey),
  ("input_6",  0.646, 0.215, warmgrey),
  ("input_7",  0.649, 0.181, warmgrey),
  ("input_8",  0.603, 0.118, warmgrey),
  ("input_9",  0.608, 0.104, warmgrey),
  ("input_10", 0.948, 0.006, warmgrey),
  ("input_11", 0.816, 0.044, warmgrey),
)

#let overall-mean = 0.748

// Sort ascending by IoU so the worst-performing sample is at the top
// and the strongest at the bottom.
#let sorted = rows.sorted(key: r => r.at(1))

#cetz.canvas({
  import cetz.draw: *

  let bar-h = 0.32
  let row-gap = 0.12
  let label-w = 0.85
  let chart-w = 9.0
  let chart-x0 = label-w
  let chart-x1 = chart-x0 + chart-w

  let x-min = 0.30
  let x-max = 1.00
  let x-scale = chart-w / (x-max - x-min)
  let to-x(v) = chart-x0 + (v - x-min) * x-scale

  // Frame: bottom axis.
  let n = sorted.len()
  let total-h = n * (bar-h + row-gap)
  let y-top = 0.5
  let y-bot = y-top - total-h - 0.4

  // X-axis tick labels.
  let ticks = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
  for tv in ticks {
    let xx = to-x(tv)
    line((xx, y-bot), (xx, y-bot - 0.08), stroke: 0.4pt + black)
    content((xx, y-bot - 0.32),
            text(size: 7.5pt, fill: black)[#tv])
  }
  content(((chart-x0 + chart-x1) / 2, y-bot - 0.60),
          text(size: 8pt, fill: black)[Mean mask IoU])

  // Overall mean vertical rule.
  let mean-x = to-x(overall-mean)
  line((mean-x, y-top + 0.05), (mean-x, y-bot),
       stroke: 0.7pt + garnet)
  content((mean-x, y-top + 0.22),
          text(size: 7.5pt, fill: garnet, weight: 700)[mean = #overall-mean])

  // Horizontal bars + whiskers + labels.
  for (i, row) in sorted.enumerate() {
    let (name, mean, ci, color) = row
    let y = y-top - (i + 0.5) * (bar-h + row-gap)
    let bx0 = chart-x0
    let bx1 = to-x(mean)

    // Bar.
    rect((bx0, y - bar-h / 2), (bx1, y + bar-h / 2),
         fill: color, stroke: 0.3pt + black)

    // Whisker (CI half-width) drawn relative to bar end.
    let lo = mean - ci
    let hi = calc.min(mean + ci, 1.0)
    let wx-lo = to-x(lo)
    let wx-hi = to-x(hi)
    line((wx-lo, y), (wx-hi, y), stroke: 0.6pt + black)
    line((wx-lo, y - 0.10), (wx-lo, y + 0.10), stroke: 0.6pt + black)
    line((wx-hi, y - 0.10), (wx-hi, y + 0.10), stroke: 0.6pt + black)

    // Y-label (sample name).
    content((chart-x0 - 0.08, y),
            text(size: 8pt, fill: black)[#name],
            anchor: "east")

    // Mean numeric label inside or after bar.
    let label-x = to-x(calc.min(mean + ci + 0.005, 0.999))
    content((label-x + 0.05, y),
            text(size: 7.5pt, fill: black)[#raw(str(mean))],
            anchor: "west")
  }

  // Bottom axis line.
  line((chart-x0, y-bot), (chart-x1, y-bot), stroke: 0.5pt + black)

  // Legend (top-right inside chart).
  let leg-y = y-top + 0.7
  let leg-x = chart-x1 - 2.5
  rect((leg-x, leg-y - 0.13), (leg-x + 0.30, leg-y + 0.13),
       fill: atlantic, stroke: 0.3pt + black)
  content((leg-x + 0.36, leg-y),
          text(size: 7.5pt, fill: black)[1080p clips],
          anchor: "west")
  rect((leg-x + 1.20, leg-y - 0.13), (leg-x + 1.50, leg-y + 0.13),
       fill: warmgrey, stroke: 0.3pt + black)
  content((leg-x + 1.56, leg-y),
          text(size: 7.5pt, fill: black)[524p clips],
          anchor: "west")
})
