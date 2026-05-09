// Frame montage — raw vs overlay pairs at six anchor frames plus
// labeled-vs-predicted comparisons from three additional samples.
//
// Placeholder until paper/assets/ contains the regenerated frames.

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("Inter", "Helvetica"), size: 8.5pt)

#let garnet = rgb("#73000A")
#let b70    = rgb("#5C5C5C")
#let b50    = rgb("#A2A2A2")
#let b10    = rgb("#ECECEC")

#cetz.canvas({
  import cetz.draw: *

  let cell-w = 2.7
  let cell-h = 1.9
  let gap = 0.18

  let labels = (
    ("input_1 frame 50",   "raw"),    ("input_1 frame 50",   "overlay"),
    ("input_1 frame 200",  "raw"),    ("input_1 frame 200",  "overlay"),
    ("input_1 frame 500",  "raw"),    ("input_1 frame 500",  "overlay"),
    ("input_2 frame 30",   "raw"),    ("input_2 frame 30",   "overlay"),
    ("input_2 frame 60",   "raw"),    ("input_2 frame 60",   "overlay"),
    ("input_2 frame 90",   "raw"),    ("input_2 frame 90",   "overlay"),
  )

  for (i, (name, kind)) in labels.enumerate() {
    let col = calc.rem(i, 4)
    let row = int(i / 4)
    let x0 = col * (cell-w + gap)
    let y0 = -row * (cell-h + gap + 0.6)
    rect((x0, y0), (x0 + cell-w, y0 - cell-h),
         fill: b10, stroke: 0.6pt + b70)
    content((x0 + cell-w / 2, y0 - cell-h - 0.22),
            text(size: 8pt, fill: b70)[#name #h(0.4em) _#kind _])
  }
})
