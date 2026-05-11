// fig:runtime_bars — Grouped horizontal bars per sample, CPU (atlantic)
// vs CUDA (garnet). X-axis frames per second. Sorted by frame count
// ascending. Values populated from runtime_benchmark.json.
//
// NOTE: This file regenerates from data/runtime_benchmark.json. To
// refresh values, run:
//   python3 _dev/figures/build_runtime_bars.py
//
// Compile:
//   typst compile paper/typst/figures/runtime_bars.typ
//                 paper/typst/figures/runtime_bars.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet    = rgb("#73000A")
#let atlantic  = rgb("#466A9F")
#let warmgrey  = rgb("#676156")
#let black     = rgb("#000000")

// Per-sample CPU and CUDA fps. Values are placeholders; the build
// script overwrites this list when the benchmark JSON is regenerated.
//   (sample, frames, cpu_fps, cuda_fps)
// Per-sample CPU and CUDA fps from runtime_benchmark.json.
// CPU times captured under 8-worker concurrent execution on
// COMECH-2422; single-process wall-clock is contention-inflated.
//   (sample, frames, cpu_fps, cuda_fps)
// Sequential single-process CPU and CUDA fps from runtime_benchmark.json
// (lock_frames=0 integrated config on COMECH-2422; no concurrent
// workload).  (sample, frames, cpu_fps, cuda_fps)
#let rows = (
  ("input_2",  100,   11.50,  65.70),
  ("input_3",  200,   12.00,  96.60),
  ("input_4",  542,   24.50, 249.60),
  ("input_5",  542,   26.20, 238.60),
  ("input_6",  542,   23.70, 244.00),
  ("input_7",  542,   24.40, 243.80),
  ("input_1",  706,   10.80, 109.70),
  ("input_10", 767,   23.70, 249.60),
  ("input_8",  842,   23.10, 253.30),
  ("input_11", 997,   24.30, 260.60),
  ("input_9",  1037,  25.30, 274.70),
)

#cetz.canvas({
  import cetz.draw: *

  let bar-h = 0.20
  let group-gap = 0.10
  let row-gap = 0.18
  let label-w = 1.10
  let chart-w = 8.0
  let chart-x0 = label-w

  // X-axis scale (fps).
  let x-max = 360.0
  let x-min = 0.0
  let x-scale = chart-w / (x-max - x-min)
  let to-x(v) = chart-x0 + (v - x-min) * x-scale

  let n = rows.len()
  let total-h = n * (2 * bar-h + group-gap + row-gap)
  let y-top = 0.5
  let y-bot = y-top - total-h - 0.4

  // X-axis ticks.
  let ticks = (0, 50, 100, 150, 200, 250, 300, 350)
  for tv in ticks {
    let xx = to-x(tv)
    line((xx, y-bot), (xx, y-bot - 0.08), stroke: 0.4pt + black)
    content((xx, y-bot - 0.32), text(size: 7.5pt, fill: black)[#tv])
  }
  content((chart-x0 + chart-w / 2, y-bot - 0.65),
          text(size: 8pt, fill: black)[Frames per second])

  // Rows.
  for (i, row) in rows.enumerate() {
    let (name, frames, cpu, cuda) = row
    let y = y-top - (i + 0.5) * (2 * bar-h + group-gap + row-gap)
    // CPU bar (lower)
    let y-cpu = y - bar-h / 2 - group-gap / 2
    rect((chart-x0, y-cpu - bar-h / 2),
         (to-x(cpu), y-cpu + bar-h / 2),
         fill: atlantic, stroke: 0.3pt + black)
    content((to-x(cpu) + 0.08, y-cpu),
            text(size: 7pt, fill: black)[#raw(str(cpu))],
            anchor: "west")
    // CUDA bar (upper)
    let y-cuda = y + bar-h / 2 + group-gap / 2
    rect((chart-x0, y-cuda - bar-h / 2),
         (to-x(cuda), y-cuda + bar-h / 2),
         fill: garnet, stroke: 0.3pt + black)
    content((to-x(cuda) + 0.08, y-cuda),
            text(size: 7pt, fill: black)[#raw(str(cuda))],
            anchor: "west")

    // Y-label (sample name + frames count).
    content((chart-x0 - 0.08, y),
            text(size: 7.5pt, fill: black)[#name],
            anchor: "east")
    content((chart-x0 - 0.08, y - 0.20),
            text(size: 6.5pt, fill: warmgrey)[#raw(str(frames)) frames],
            anchor: "east")
  }

  // Bottom axis line.
  line((chart-x0, y-bot), (chart-x0 + chart-w, y-bot), stroke: 0.5pt + black)

  // Legend.
  let leg-y = y-top + 0.7
  let leg-x = chart-x0 + chart-w - 2.5
  rect((leg-x, leg-y - 0.10), (leg-x + 0.30, leg-y + 0.10),
       fill: atlantic, stroke: 0.3pt + black)
  content((leg-x + 0.36, leg-y),
          text(size: 7.5pt, fill: black)[CPU],
          anchor: "west")
  rect((leg-x + 1.20, leg-y - 0.10), (leg-x + 1.50, leg-y + 0.10),
       fill: garnet, stroke: 0.3pt + black)
  content((leg-x + 1.56, leg-y),
          text(size: 7.5pt, fill: black)[CUDA],
          anchor: "west")
})
