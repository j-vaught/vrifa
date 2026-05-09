// Peak-vs-first reference time-series plot.
//
// Tracks one pixel's L* (CIELAB lightness) across all 706 frames of
// input_1. Raw L* drops sharply when the front arrives; the running
// peak tracks the maximum and absorbs slow lighting drift before the
// front, providing a stable reference.
//
// Data: paper/assets/method/peak_reference.csv (frame, time_s, L_raw, L_peak)

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 8pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet    = rgb("#73000A")
#let atlantic  = rgb("#466A9F")
#let horseshoe = rgb("#65780B")
#let warmgrey  = rgb("#676156")
#let b70       = rgb("#5C5C5C")
#let b50       = rgb("#A2A2A2")

// Load the CSV produced by _dev/figures/extract_method_data.py.
#let raw = csv("/assets/method/peak_reference.csv")
#let rows = raw.slice(1) // drop header

// Convert string columns into numeric arrays.
#let frames = rows.map(r => float(r.at(0)))
#let l-raw = rows.map(r => float(r.at(2)))
#let l-peak = rows.map(r => float(r.at(3)))

#let max-frame = calc.max(..frames)
#let min-l = calc.min(..l-raw, ..l-peak)
#let max-l = calc.max(..l-raw, ..l-peak)

#cetz.canvas({
  import cetz.draw: *

  let plot-w = 11.5
  let plot-h = 5.0
  let xrange = max-frame - 0.0
  let yrange = (max-l - min-l) * 1.10
  let y0 = min-l - (max-l - min-l) * 0.05

  let to-x = (f) => f / xrange * plot-w
  let to-y = (l) => -((max-l + (max-l - min-l) * 0.05 - l) / yrange) * plot-h

  // Axis frame.
  rect((0, 0), (plot-w, -plot-h), stroke: 0.6pt + b70, fill: white)

  // X-axis ticks (frames).
  for f in (0, 100, 200, 300, 400, 500, 600, 700) {
    let x = to-x(f)
    line((x, -plot-h), (x, -plot-h - 0.12), stroke: 0.5pt + b70)
    content((x, -plot-h - 0.4), text(size: 8pt, fill: b70)[#f])
  }
  content((plot-w / 2, -plot-h - 0.85),
          text(size: 9pt, fill: b70)[frame index])

  // Y-axis ticks (L*).
  for l in (60, 80, 100, 120, 140, 160) {
    if l >= min-l - 5 and l <= max-l + 5 {
      let y = to-y(l)
      line((-0.12, y), (0, y), stroke: 0.5pt + b70)
      content((-0.25, y), text(size: 8pt, fill: b70)[#l], anchor: "east")
    }
  }
  content((-1.0, -plot-h / 2),
          text(size: 9pt, fill: b70)[L\*],
          anchor: "south", angle: 90deg)

  // Plot the raw L* line.
  let raw-points = ()
  for i in range(frames.len()) {
    raw-points.push((to-x(frames.at(i)), to-y(l-raw.at(i))))
  }
  line(..raw-points, stroke: 1.0pt + atlantic)

  // Plot the running peak line.
  let peak-points = ()
  for i in range(frames.len()) {
    peak-points.push((to-x(frames.at(i)), to-y(l-peak.at(i))))
  }
  line(..peak-points, stroke: 1.6pt + garnet)

  // Annotation: front-arrival region. Find first frame where raw drops
  // by more than 25 below the running peak.
  let arrival = -1
  for i in range(frames.len()) {
    if arrival == -1 and l-peak.at(i) - l-raw.at(i) > 25 {
      arrival = i
    }
  }
  if arrival > 0 {
    let ax = to-x(frames.at(arrival))
    line((ax, 0.3), (ax, -plot-h), stroke: (paint: horseshoe, dash: "dashed", thickness: 0.6pt))
    content((ax, 0.55),
            text(size: 8pt, fill: horseshoe, weight: 700)[front arrival],
            anchor: "south")
  }

  // Legend.
  let lx = plot-w - 3.5
  let ly = -0.6
  line((lx, ly), (lx + 0.7, ly), stroke: 1.6pt + garnet)
  content((lx + 0.85, ly), text(size: 8.5pt, fill: b70)[running peak _P_], anchor: "west")
  line((lx, ly - 0.4), (lx + 0.7, ly - 0.4), stroke: 1.0pt + atlantic)
  content((lx + 0.85, ly - 0.4), text(size: 8.5pt, fill: b70)[raw L\* at one pixel], anchor: "west")
})
