// Peak-brightness figure — four pixels of input_1 tracked across all
// 706 frames, with raw $L^*$ in solid colored lines and the per-pixel
// running peak $P$ in matching dashed lines. The four pixels are
// chosen to span the run: arrival frames ~80, ~230, ~400, ~590.
//
// Story the figure tells:
//   1. Each pixel's $L^*$ drifts upward during lamp warm-up.
//   2. The running peak absorbs that drift, so a fixed reference at
//      frame 0 would have treated the warm-up as a (negative) wetting
//      event but the running peak does not.
//   3. When the front arrives at a pixel, $L^*$ collapses sharply
//      below $P$. The size of that drop is the signal the threshold
//      detects.
//   4. The four pixels show this happening at different times, which
//      is why a single global reference frame is wrong: every pixel
//      wets on its own schedule.
//
// Compile:
//   typst compile paper/typst/figures/peak_reference.typ \
//                 paper/typst/figures/peak_reference.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet    = rgb("#73000A")
#let atlantic  = rgb("#466A9F")
#let horseshoe = rgb("#65780B")
#let rose      = rgb("#CC2E40")
#let b70       = rgb("#5C5C5C")
#let b50       = rgb("#A2A2A2")
#let b10       = rgb("#ECECEC")
#let txt       = rgb("#000000")

// Read CSVs at compile time.
#let trace-rows = csv("peak_data/traces.csv")
#let pixel-rows = csv("peak_data/pixels.csv")

#let pixel-colors = (garnet, atlantic, horseshoe, rose)

// Parse: traces.csv columns are (frame, L1, P1, L2, P2, L3, P3, L4, P4).
#let parse-traces() = {
  let frames = ()
  let L = ((), (), (), ())
  let P = ((), (), (), ())
  for r in trace-rows.slice(1) {
    frames.push(int(r.at(0)))
    for i in range(4) {
      L.at(i).push(float(r.at(1 + 2 * i)))
      P.at(i).push(float(r.at(2 + 2 * i)))
    }
  }
  (frames: frames, L: L, P: P)
}

#let data = parse-traces()
#let pixels = pixel-rows.slice(1).map(r => (
  x: int(r.at(1)),
  y: int(r.at(2)),
  arrival: int(r.at(3)),
  peak: int(r.at(4)),
))

// Plot extents.
#let x-min = 0
#let x-max = data.frames.at(data.frames.len() - 1)
#let y-min = 70
#let y-max = 180

// Pre-rotate the y-axis label outside the cetz.canvas so the
// Typst built-in rotate() is in scope (cetz.draw.rotate shadows it).
#let y-axis-label = rotate(-90deg, reflow: true,
  text(size: 8pt, fill: rgb("#000000"))[delta magnitude])

#cetz.canvas({
  import cetz.draw: *

  // Plot box geometry.
  let plot-w = 13.0
  let plot-h = 5.0

  // Map data to canvas coordinates.
  let to-x(f) = (f - x-min) / (x-max - x-min) * plot-w
  let to-y(v) = (v - y-min) / (y-max - y-min) * plot-h

  // X-axis tick marks every 100 frames.
  let x-ticks = (0, 100, 200, 300, 400, 500, 600, 700)
  // Y-axis tick marks every 20 L* units.
  let y-ticks = (80, 100, 120, 140, 160, 180)

  // Plot frame.
  rect((0, 0), (plot-w, plot-h), stroke: 0.6pt + b70, fill: none)

  // Horizontal gridlines (light).
  for v in y-ticks {
    let y = to-y(v)
    line((0, y), (plot-w, y), stroke: 0.3pt + b10)
  }

  // Vertical gridlines (light).
  for f in x-ticks {
    let x = to-x(f)
    line((x, 0), (x, plot-h), stroke: 0.3pt + b10)
  }

  // Per-pixel arrival markers: a dotted vertical line in each pixel's
  // own color, at the frame where that pixel first drops more than 30
  // L* units below its running peak. Tells the reader exactly when the
  // pipeline considers each pixel to have wetted.
  for (i, p) in pixels.enumerate() {
    let color = pixel-colors.at(i)
    let x = to-x(p.arrival)
    line((x, 0), (x, plot-h),
         stroke: (paint: color, thickness: 0.5pt, dash: "dotted"))
    content((x, plot-h + 0.14),
            text(size: 6pt, fill: color)[#p.arrival],
            anchor: "south")
  }

  // Draw running peak (dashed) and raw L* (solid) for each pixel.
  // Peak goes first so the solid L* line is drawn on top.
  for i in range(4) {
    let color = pixel-colors.at(i)
    let pts-P = ()
    let pts-L = ()
    for j in range(data.frames.len()) {
      let f = data.frames.at(j)
      pts-P.push((to-x(f), to-y(data.P.at(i).at(j))))
      pts-L.push((to-x(f), to-y(data.L.at(i).at(j))))
    }
    line(..pts-P, stroke: (paint: color, thickness: 0.7pt, dash: "dashed"))
    line(..pts-L, stroke: 1.0pt + color)
  }

  // X-axis tick labels.
  for f in x-ticks {
    let x = to-x(f)
    line((x, 0), (x, -0.08), stroke: 0.5pt + b70)
    content((x, -0.28), text(size: 7pt, fill: b70)[#f])
  }
  content((plot-w / 2, -0.62),
          text(size: 8pt, fill: txt)[frame index],
          anchor: "north")

  // Y-axis tick labels.
  for v in y-ticks {
    let y = to-y(v)
    line((0, y), (-0.08, y), stroke: 0.5pt + b70)
    content((-0.18, y), text(size: 7pt, fill: b70)[#v], anchor: "east")
  }
  content((-0.92, plot-h / 2), y-axis-label, anchor: "center")

  // Legend (top-right corner of plot area, inside box).
  let leg-row-h = 0.26
  let leg-w = 2.0
  let leg-x = plot-w - leg-w - 0.05
  let leg-y = plot-h - 0.25
  rect((leg-x - 0.10, leg-y + 0.16),
       (plot-w - 0.05, leg-y - 4 * leg-row-h),
       fill: white, stroke: 0.4pt + b70)
  for (i, p) in pixels.enumerate() {
    let color = pixel-colors.at(i)
    let y = leg-y - i * leg-row-h
    // Color swatch
    line((leg-x, y), (leg-x + 0.4, y), stroke: 1.2pt + color)
    content((leg-x + 0.5, y),
            text(size: 6.5pt, fill: txt)[(#p.x, #p.y)],
            anchor: "west")
  }
})
