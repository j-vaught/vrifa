// IoU vs parameter curves from the input_1 ablation.
//
// Source: data/ablation_results.json (phase 1 + phase 2).
// Data is embedded inline so the figure compiles standalone.

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 12pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet    = rgb("#73000A")
#let atlantic  = rgb("#466A9F")
#let horseshoe = rgb("#65780B")
#let warmgrey  = rgb("#676156")
#let rose      = rgb("#CC2E40")
#let b70       = rgb("#5C5C5C")
#let b50       = rgb("#A2A2A2")
#let b10       = rgb("#ECECEC")

// Each curve: list of (x, iou) pairs.
#let threshold_offset = (
  (-50, 0.3263), (-45, 0.3266), (-40, 0.3306),
  (-35, 0.3453), (-30, 0.4571), (-25, 0.5329),
  (-20, 0.6062), (-15, 0.6768), (-10, 0.7171),
)
#let min_area = (
  (0, 0.4542), (50, 0.4542), (100, 0.4543), (200, 0.4552),
  (400, 0.4571), (600, 0.4584), (800, 0.4590),
  (1200, 0.4607), (1600, 0.4615),
)
#let morph_kernel = (
  (3, 0.4728), (5, 0.4734), (7, 0.4730), (9, 0.4674),
  (11, 0.4612), (13, 0.4571), (15, 0.4492),
  (17, 0.4405), (19, 0.4332), (21, 0.4262),
)
#let blur_kernel = (
  (3, 0.3412), (5, 0.4037), (7, 0.4366),
  (9, 0.4571), (11, 0.4610), (13, 0.4730), (15, 0.4846),
)
#let lock_frames = (
  (0, 0.6401), (1, 0.4323), (2, 0.4487),
  (3, 0.4571), (5, 0.4702), (10, 0.4952), (20, 0.5201),
)

#let categorical = (
  ("CIELAB",       0.4571, atlantic),
  ("RGB",          0.4431, atlantic),
  ("Grayscale",    0.3506, atlantic),
  ("HSV",          0.1276, rose),
  ("darken=true",  0.4571, atlantic),
  ("darken=false", 0.3261, rose),
  ("peak=true",    0.4571, atlantic),
  ("peak=false",   0.6750, horseshoe),
  ("blur=on",      0.4571, atlantic),
  ("blur=off",     0.3259, rose),
  ("ref=first",    0.4571, atlantic),
  ("ref=running",  0.4571, atlantic),
)

#let baseline_iou = 0.4571
#let optimum_iou = 0.8333

// Helper: draw one axis-profile panel.
#let panel(
  data,
  title,
  baseline_x,
  x_min, x_max,
  y_min: 0.1, y_max: 0.9,
  width: 5.5, height: 3.6,
) = {
  cetz.canvas({
    import cetz.draw: *
    let to-x = (x) => (x - x_min) / (x_max - x_min) * width
    let to-y = (y) => -((y_max - y) / (y_max - y_min)) * height

    // Background grid + frame.
    rect((0, 0), (width, -height), stroke: 0.5pt + b70, fill: white)

    // Y gridlines.
    let y_step = 0.2
    let y = y_min
    while y <= y_max + 0.001 {
      let yp = to-y(y)
      line((0, yp), (width, yp), stroke: 0.3pt + b10)
      content((-0.1, yp),
              text(size: 7pt, fill: b70)[#calc.round(y, digits: 1)],
              anchor: "east")
      y = y + y_step
    }

    // Baseline horizontal line at IoU 0.4571.
    line((0, to-y(baseline_iou)), (width, to-y(baseline_iou)),
         stroke: (paint: b50, dash: "dashed", thickness: 0.5pt))

    // Optimum horizontal at 0.8333.
    if y_max > optimum_iou {
      line((0, to-y(optimum_iou)), (width, to-y(optimum_iou)),
           stroke: (paint: horseshoe, dash: "dotted", thickness: 0.6pt))
    }

    // Curve.
    let pts = ()
    for p in data {
      pts.push((to-x(p.at(0)), to-y(p.at(1))))
    }
    line(..pts, stroke: 1.6pt + garnet)

    // Markers.
    for p in data {
      let xv = p.at(0)
      let yv = p.at(1)
      let is_baseline = (calc.abs(xv - baseline_x) < 0.001)
      let r = if is_baseline { 0.13 } else { 0.08 }
      let fill = if is_baseline { atlantic } else { garnet }
      circle((to-x(xv), to-y(yv)), radius: r, fill: fill, stroke: 0.4pt + fill.darken(20%))
    }

    // Title above the panel.
    content((width / 2, 0.45),
            text(size: 9.5pt, weight: 700, fill: garnet)[#title])

    // X axis ticks and labels.
    let xticks = ()
    for p in data { xticks.push(p.at(0)) }
    for x in xticks {
      let xp = to-x(x)
      line((xp, -height), (xp, -height - 0.08), stroke: 0.4pt + b70)
    }
    // Sparse x labels: first, middle, last.
    let labels = (xticks.first(), xticks.at(int(xticks.len() / 2)), xticks.last())
    for x in labels {
      content((to-x(x), -height - 0.32),
              text(size: 7pt, fill: b70)[#x])
    }
  })
}

// Categorical panel.
#let categorical_panel(
  data,
  width: 11.6, height: 3.6,
) = {
  cetz.canvas({
    import cetz.draw: *
    let n = data.len()
    let bar_w = width / (n + 0.5) * 0.78
    let pad = (width - bar_w * n) / (n + 1)
    let y_min = 0.0
    let y_max = 0.8
    let to-y = (v) => -((y_max - v) / (y_max - y_min)) * height

    rect((0, 0), (width, -height), stroke: 0.5pt + b70, fill: white)

    // Y ticks.
    let y = y_min
    while y <= y_max + 0.001 {
      let yp = to-y(y)
      line((0, yp), (width, yp), stroke: 0.3pt + b10)
      content((-0.1, yp),
              text(size: 7pt, fill: b70)[#calc.round(y, digits: 1)],
              anchor: "east")
      y = y + 0.2
    }

    // Baseline & optimum reference lines.
    line((0, to-y(baseline_iou)), (width, to-y(baseline_iou)),
         stroke: (paint: b50, dash: "dashed", thickness: 0.5pt))
    line((0, to-y(optimum_iou)), (width, to-y(optimum_iou)),
         stroke: (paint: horseshoe, dash: "dotted", thickness: 0.6pt))

    // Bars + labels.
    for (i, item) in data.enumerate() {
      let (label, value, color) = item
      let x0 = pad + i * (bar_w + pad)
      rect((x0, to-y(0)), (x0 + bar_w, to-y(value)),
           fill: color.lighten(35%), stroke: 0.5pt + color.darken(15%))
      // Value above bar.
      content((x0 + bar_w / 2, to-y(value) + 0.18),
              text(size: 6.5pt, fill: color.darken(20%), weight: 600)[#calc.round(value, digits: 3)])
      // Label below bar (horizontal; small font to fit).
      content((x0 + bar_w / 2, -height - 0.32),
              text(size: 6pt, fill: b70)[#label])
    }

    content((width / 2, 0.45),
            text(size: 9.5pt, weight: 700, fill: garnet)[Categorical])
  })
}

// Top of figure: title.
#align(center)[
  #text(size: 11pt, weight: 700, fill: garnet)[
    Single-axis ablation curves on input_1
  ]
  #v(-0.6em)
  #text(size: 8pt, fill: b70)[
    dashed grey = baseline IoU 0.457   ·   dotted green = joint optimum IoU 0.833   ·   blue marker = default value
  ]
  #v(0.4em)
]

// Two columns of small panels.
#grid(
  columns: (auto, auto),
  column-gutter: 1em,
  row-gutter: 1em,
  panel(threshold_offset, "threshold-offset", -30, -50, -10),
  panel(min_area,         "min-area",          400,    0, 1600),
  panel(morph_kernel,     "morph-kernel",       13,    3,   21),
  panel(blur_kernel,      "blur-kernel",         9,    3,   15),
  panel(lock_frames,      "lock-frames",         3,    0,   20),
  rect(width: 5.5cm, height: 3.6cm, fill: white, stroke: none),
)

#v(0.6em)

#align(center)[
  #categorical_panel(categorical)
]
