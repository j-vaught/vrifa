// fig:component_bars — Per leave-one-out variant, mean ΔIoU bar plus
// strip plot of per-sample ΔIoU values. Strip dots crossing zero
// outlined in garnet to flag primitives that are neutral or
// counterproductive on at least one sample.
//
// NOTE: Values populated from data/agreement_metrics_<variant>.json
// produced by the leave-one-out runs on COMECH-2422.
//
// Compile:
//   typst compile paper/typst/figures/component_ablation.typ
//                 paper/typst/figures/component_ablation.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet    = rgb("#73000A")
#let atlantic  = rgb("#466A9F")
#let warmgrey  = rgb("#676156")
#let black     = rgb("#000000")
#let b50       = rgb("#A2A2A2")

// Rows: (component, mean_delta_iou, per_sample_deltas[11])
// Per-sample order: input_1..input_11.
// Values are placeholders; the build script overwrites when JSONs land.
// Per-sample ΔIoU per leave-one-out variant.
// ΔIoU = variant_IoU - integrated_IoU per sample.
// Negative = primitive matters (its removal hurts).
// Positive = primitive is neutral or counterproductive on this sample.
#let rows = (
  ("Peak reference",     +0.0189, (+0.1446, -0.0063, -0.0119, -0.0004, +0.0027, +0.0193, +0.0000, -0.0004, +0.0035, +0.0004, +0.0562)),
  ("Darken-only clip",   -0.0219, (-0.0827, -0.0116, -0.0157, +0.0017, -0.0002, +0.0339, +0.0005, -0.0006, -0.0021, -0.0001, -0.1646)),
  ("Dynamic-lag ref",    +0.0000, (+0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000)),
  ("ROI restriction",    -0.0266, (-0.2924, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000, +0.0000)),
  ("Morph cleanup",      +0.0060, (-0.0008, -0.0003, -0.0023, +0.0053, +0.0092, +0.0026, +0.0046, +0.0150, +0.0210, -0.0035, +0.0150)),
  ("Camera-shift reg.",  -0.1008, (+0.0069, -0.0860, -0.5809, -0.0003, -0.0015, -0.0001, -0.0007, -0.0012, -0.0030, -0.2728, -0.1697)),
)

#cetz.canvas({
  import cetz.draw: *

  let row-h = 0.42
  let label-w = 2.0
  let bar-w = 2.8
  let strip-w = 4.6
  let chart-x0 = label-w
  let bar-x0 = chart-x0
  let bar-x1 = bar-x0 + bar-w
  let strip-x0 = bar-x1 + 0.30
  let strip-x1 = strip-x0 + strip-w

  let delta-min = -0.60
  let delta-max = 0.16
  // Compute bar-x scale (mean ΔIoU bars). Zero anchored.
  let bar-scale = bar-w / (delta-max - delta-min)
  let bar-to-x(v) = bar-x0 + (v - delta-min) * bar-scale
  // Strip x scale (same domain).
  let strip-scale = strip-w / (delta-max - delta-min)
  let strip-to-x(v) = strip-x0 + (v - delta-min) * strip-scale

  let n = rows.len()
  let total-h = n * row-h
  let y-top = 0.5
  let y-bot = y-top - total-h - 0.4

  // X axis ticks (mean bars only)
  let ticks = (-0.50, -0.40, -0.30, -0.20, -0.10, 0.0, 0.10)
  for tv in ticks {
    let xx = bar-to-x(tv)
    line((xx, y-bot), (xx, y-bot - 0.08), stroke: 0.4pt + black)
    content((xx, y-bot - 0.28),
            text(size: 7pt, fill: black)[#raw(str(tv))])
  }
  content((bar-x0 + bar-w / 2, y-bot - 0.55),
          text(size: 7.5pt, fill: black)[Mean ΔIoU (11 samples)])

  // Strip axis ticks
  for tv in ticks {
    let xx = strip-to-x(tv)
    line((xx, y-bot), (xx, y-bot - 0.08), stroke: 0.4pt + black)
    content((xx, y-bot - 0.28),
            text(size: 7pt, fill: black)[#raw(str(tv))])
  }
  content((strip-x0 + strip-w / 2, y-bot - 0.55),
          text(size: 7.5pt, fill: black)[Per-sample ΔIoU])

  // Zero rules.
  let bar-zero-x = bar-to-x(0)
  let strip-zero-x = strip-to-x(0)
  line((bar-zero-x, y-top + 0.05), (bar-zero-x, y-bot),
       stroke: 0.5pt + b50)
  line((strip-zero-x, y-top + 0.05), (strip-zero-x, y-bot),
       stroke: 0.5pt + b50)

  // Rows.
  for (i, row) in rows.enumerate() {
    let (name, mean, deltas) = row
    let y = y-top - (i + 0.5) * row-h

    // Row label.
    content((chart-x0 - 0.08, y),
            text(size: 8pt, fill: black)[#name],
            anchor: "east")

    // Bar (mean ΔIoU).
    let bar-color = if mean < 0 { garnet } else { atlantic }
    rect((bar-zero-x, y - 0.13),
         (bar-to-x(mean), y + 0.13),
         fill: bar-color, stroke: 0.3pt + black)
    content((bar-to-x(mean) + (if mean < 0 { -0.08 } else { 0.08 }), y),
            text(size: 7pt, fill: black)[#raw(str(mean))],
            anchor: if mean < 0 { "east" } else { "west" })

    // Strip: per-sample dots.
    let any-positive = deltas.any(d => d >= 0)
    for d in deltas {
      let sx = strip-to-x(d)
      let stroke-color = if d >= 0 { garnet } else { black }
      let fill-color = if d >= 0 { garnet } else { warmgrey }
      circle((sx, y), radius: 0.06,
             fill: fill-color, stroke: 0.4pt + stroke-color)
    }
  }

  // Bottom axis lines.
  line((bar-x0, y-bot), (bar-x1, y-bot), stroke: 0.5pt + black)
  line((strip-x0, y-bot), (strip-x1, y-bot), stroke: 0.5pt + black)
})
