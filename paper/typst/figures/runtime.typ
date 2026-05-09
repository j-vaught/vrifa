// Runtime breakdown — three-tier perf gate measurements.
//
// Numbers are measured medians from three back-to-back perf-gate runs.
// Source: vrifa-rs/crates/vrifa-cli/benches/perf_gate.rs and the
// engineer's reported close-out runs.
//
// Compile:
//   typst compile paper/typst/figures/runtime.typ paper/typst/figures/runtime.pdf

#import "@preview/cetz:0.4.0"
#import "@preview/cetz-plot:0.1.2": chart

#set page(width: auto, height: auto, margin: 8pt, fill: white)
#set text(font: ("Inter", "Helvetica"), size: 9pt)

#let garnet    = rgb("#73000A")
#let atlantic  = rgb("#466A9F")
#let horseshoe = rgb("#65780B")
#let b70       = rgb("#5C5C5C")
#let b50       = rgb("#A2A2A2")

// Median seconds across three runs.
// Run 1: 23.563 / 27.169 / 70.330 / 3.798 / 4.543 / 8.641
// Run 2: 23.232 / 28.411 / 75.473 / 3.635 / 4.355 / 8.388
// Run 3: 25.001 / 28.204 / 71.473 / 3.220 / 3.986 / 7.912
//
// Median: input_1 detector 23.56, core 28.20, full 71.47.
//         input_2 detector 3.64,  core 4.36,  full 8.39.

#let data = (
  ("input_1 detector",  23.56,  garnet),
  ("input_1 core",      28.20,  garnet.lighten(35%)),
  ("input_1 full",      71.47,  garnet.lighten(60%)),
  ("input_2 detector",   3.64,  atlantic),
  ("input_2 core",       4.36,  atlantic.lighten(35%)),
  ("input_2 full",       8.39,  atlantic.lighten(60%)),
)

#let gates = (
  ("input_1 detector",  32.0),
  ("input_1 core",      35.0),
  ("input_1 full",      90.0),
  ("input_2 detector",   5.0),
  ("input_2 core",       6.0),
  ("input_2 full",      11.0),
)

#cetz.canvas({
  import cetz.draw: *

  let plot-w = 11.0
  let plot-h = 6.0
  let max-x = 95.0
  let bar-h = 0.55
  let bar-gap = 0.4

  // Y axis baseline.
  line((0, 0), (0, -plot-h), stroke: 0.8pt + b70)
  // X axis.
  line((0, -plot-h), (plot-w, -plot-h), stroke: 0.8pt + b70)

  // X gridlines and ticks.
  for v in (0, 20, 40, 60, 80) {
    let x = v / max-x * plot-w
    line((x, -plot-h), (x, -plot-h - 0.15), stroke: 0.6pt + b70)
    if v > 0 {
      line((x, 0), (x, -plot-h), stroke: 0.4pt + b50)
    }
    content((x, -plot-h - 0.45), text(size: 8pt, fill: b70)[#v])
  }
  content((plot-w / 2, -plot-h - 0.95),
          text(size: 9pt, fill: b70)[wall-clock seconds (median of three runs)])

  for (i, row) in data.enumerate() {
    let (label, val, color) = row
    let y = -i * (bar-h + bar-gap) - 0.5
    let bar-x = val / max-x * plot-w

    // Bar.
    rect((0, y), (bar-x, y - bar-h), fill: color, stroke: 0.5pt + color.darken(20%))
    // Label on left of axis.
    content((-0.15, y - bar-h / 2),
            text(size: 8.5pt, fill: b70)[#label],
            anchor: "east")
    // Value at end of bar.
    content((bar-x + 0.15, y - bar-h / 2),
            text(size: 8pt, fill: garnet, weight: 700)[#str(val) s],
            anchor: "west")

    // Gate marker as a vertical hash to the right of each bar.
    let gate = gates.at(i).at(1)
    let gate-x = gate / max-x * plot-w
    line((gate-x, y + 0.1), (gate-x, y - bar-h - 0.1),
         stroke: 1.2pt + horseshoe)
    content((gate-x, y + 0.28),
            text(size: 7pt, fill: horseshoe, weight: 700)[gate #str(gate)],
            anchor: "south")
  }

  // Title-like header.
  content((plot-w / 2, 0.5),
          text(size: 10pt, weight: 700, fill: garnet)[Three-tier perf-gate, measured medians vs configured gates])
})
