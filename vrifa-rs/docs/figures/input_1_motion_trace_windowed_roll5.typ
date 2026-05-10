// Corrected motion trace for input_1 after normalized, windowed phase correlation
// and a 5-frame rolling cumulative trigger.
//
// Compile:
//   typst compile --root vrifa-rs/docs \
//     vrifa-rs/docs/figures/input_1_motion_trace_windowed_roll5.typ \
//     vrifa-rs/docs/artifacts/input_1_motion_trace_windowed_roll5.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 8pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet = rgb("#73000A")
#let black = rgb("#000000")
#let white = rgb("#FFFFFF")
#let b90 = rgb("#363636")
#let b70 = rgb("#5C5C5C")
#let b50 = rgb("#A2A2A2")
#let b30 = rgb("#C7C7C7")
#let b10 = rgb("#ECECEC")
#let sandstorm = rgb("#FFF2E3")
#let rose = rgb("#CC2E40")
#let atlantic = rgb("#466A9F")
#let horseshoe = rgb("#65780B")

#let trace = json("../data/input_1_motion_trace_windowed_roll5.json")
#let rows = trace.at("rows")
#let frame-count = trace.at("frame_count")
#let threshold = trace.at("motion_per_frame_threshold")
#let rolling-threshold = trace.at("cumulative_motion_threshold")
#let highlight-windows = trace.at("highlight_windows")
#let time-markers = trace.at("time_markers")

#let max-y = 3.4

#let x-pos(frame, plot-w) = ((frame - 1) / (frame-count - 1)) * plot-w
#let y-pos(value, plot-h) = (value / max-y) * plot-h

#cetz.canvas({
  import cetz.draw: *

  let plot-w = 12.4
  let plot-h = 5.4
  let left-pad = 1.2
  let bottom-pad = 1.2
  let top-pad = 0.95

  let x0 = left-pad
  let y0 = bottom-pad
  let x1 = left-pad + plot-w
  let y1 = bottom-pad + plot-h

  rect((x0, y0), (x1, y1), fill: white, stroke: 0.6pt + b30)

  for window in highlight-windows {
    let shade-x0 = x0 + x-pos(window.at("start"), plot-w)
    let shade-x1 = x0 + x-pos(window.at("end"), plot-w)
    rect((shade-x0, y0), (shade-x1, y1), fill: sandstorm, stroke: none)
  }

  for tick in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0) {
    let y = y0 + y-pos(tick, plot-h)
    line((x0, y), (x1, y), stroke: 0.4pt + b10)
    content((x0 - 0.18, y), text(size: 8pt, fill: b70)[#tick], anchor: "east")
  }

  for tick in (1, 100, 200, 300, 400, 500, 600, 706) {
    let x = x0 + x-pos(tick, plot-w)
    line((x, y0), (x, y0 - 0.10), stroke: 0.6pt + b70)
    if tick != 1 and tick != 706 {
      line((x, y0), (x, y1), stroke: 0.35pt + b10)
    }
    content((x, y0 - 0.28), text(size: 8pt, fill: b70)[#tick], anchor: "north")
  }

  line((x0, y0), (x1, y0), stroke: 0.8pt + black)
  line((x0, y0), (x0, y1), stroke: 0.8pt + black)

  let thresh-y = y0 + y-pos(threshold, plot-h)
  line((x0, thresh-y), (x1, thresh-y), stroke: 0.8pt + garnet)
  content((x1, thresh-y + 0.08),
          text(size: 8pt, fill: garnet, weight: 700)[per-frame threshold 1.5 px],
          anchor: "south-east")

  let rolling-y = y0 + y-pos(rolling-threshold, plot-h)
  line((x0, rolling-y), (x1, rolling-y), stroke: 0.8pt + atlantic)
  content((x1, rolling-y + 0.08),
          text(size: 8pt, fill: atlantic, weight: 700)[rolling-5 threshold 3.0 px],
          anchor: "south-east")

  for marker in time-markers {
    let x = x0 + x-pos(marker.at("frame"), plot-w)
    line((x, y0), (x, y1), stroke: 0.7pt + horseshoe)
    if marker.at("frame") == 61 {
      content((x, y0 - 0.55),
              text(size: 7.5pt, fill: horseshoe, weight: 700)[#marker.at("label")],
              anchor: "north")
    }
  }

  // Per-frame magnitude.
  for i in range(rows.len() - 1) {
    let a = rows.at(i)
    let b = rows.at(i + 1)
    let ax = x0 + x-pos(a.at("frame"), plot-w)
    let ay = y0 + y-pos(a.at("magnitude"), plot-h)
    let bx = x0 + x-pos(b.at("frame"), plot-w)
    let by = y0 + y-pos(b.at("magnitude"), plot-h)
    line((ax, ay), (bx, by), stroke: 0.9pt + garnet)
  }

  // Rolling-5 cumulative magnitude.
  for i in range(rows.len() - 1) {
    let a = rows.at(i)
    let b = rows.at(i + 1)
    let ax = x0 + x-pos(a.at("frame"), plot-w)
    let ay = y0 + y-pos(a.at("recent_window_magnitude"), plot-h)
    let bx = x0 + x-pos(b.at("frame"), plot-w)
    let by = y0 + y-pos(b.at("recent_window_magnitude"), plot-h)
    line((ax, ay), (bx, by), stroke: 0.9pt + atlantic)
  }

  for row in rows {
    if row.at("fit_applied") {
      let x = x0 + x-pos(row.at("frame"), plot-w)
      rect((x - 0.02, y0 - 0.10), (x + 0.02, y0 + 0.18),
           fill: rose, stroke: 0.4pt + rose)
    }
  }

  let early-start = 67
  let early-peak = 71
  let late-frame = 310
  let early-row = rows.at(early-peak - 1)
  let early-x = x0 + x-pos(early-peak, plot-w)
  let early-y = y0 + y-pos(early-row.at("magnitude"), plot-h)
  rect((early-x - 0.05, early-y - 0.05), (early-x + 0.05, early-y + 0.05),
       fill: black, stroke: 0.4pt + black)
  line((early-x, early-y + 0.07), (early-x, early-y + 0.42), stroke: 0.7pt + black)
  content((early-x, early-y + 0.50),
          text(size: 8pt, fill: b90, weight: 700)[frame 71, 2.34 s],
          anchor: "south")

  let late-row = rows.at(late-frame - 1)
  let late-x = x0 + x-pos(late-frame, plot-w)
  let late-y = y0 + y-pos(late-row.at("magnitude"), plot-h)
  rect((late-x - 0.05, late-y - 0.05), (late-x + 0.05, late-y + 0.05),
       fill: b50, stroke: 0.4pt + b50)
  line((late-x, late-y + 0.07), (late-x, late-y + 0.40), stroke: 0.7pt + b50)
  content((late-x, late-y + 0.48),
          text(size: 8pt, fill: b70)[frame 310, low motion],
          anchor: "south")

  content(((x0 + x1) / 2, y1 + top-pad),
          text(size: 10pt, weight: 700, fill: garnet)[input_1 motion trace after windowed phase correlation],
          anchor: "south")
  content(((x0 + x1) / 2, y1 + 0.63),
          text(size: 8pt, fill: b70)[garnet is per-frame motion, blue is rolling 5-frame cumulative motion, red ticks are ECC refits],
          anchor: "south")
  content(((x0 + x1) / 2, y0 - 0.78),
          text(size: 9pt, fill: b70)[frame index],
          anchor: "north")
  content((x0, y1 + 0.18),
          text(size: 8pt, fill: b70)[shift magnitude (px)],
          anchor: "south-west")
})
