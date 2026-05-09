// Reference-mode comparison figure.
//
// Side-by-side mask outputs for `first` / `running` / `dynamic`
// reference modes on three frames spanning the fill (input_1
// 50 / 200 / 500). Demonstrates the configurability claim.
//
// Placeholder grid; the actual mask PNGs land here once the
// regeneration runs complete (paper/assets/refmode_<mode>_<idx>.png).

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("Inter", "Helvetica"), size: 9pt)

#let garnet    = rgb("#73000A")
#let atlantic  = rgb("#466A9F")
#let horseshoe = rgb("#65780B")
#let b70       = rgb("#5C5C5C")
#let b50       = rgb("#A2A2A2")
#let b10       = rgb("#ECECEC")

#cetz.canvas({
  import cetz.draw: *

  let cell-w = 3.4
  let cell-h = 2.4
  let gap = 0.25

  let modes = ("first", "running", "dynamic")
  let frames = ("frame 50", "frame 200", "frame 500")

  // Column headers (frames).
  for (j, fr) in frames.enumerate() {
    let x = (j + 1) * (cell-w + gap)
    content((x + cell-w / 2, 0.3),
            text(weight: 700, fill: b70, size: 10pt)[#fr])
  }

  // Rows: one per mode.
  for (i, mode) in modes.enumerate() {
    let y0 = -(i + 1) * (cell-h + gap)
    // Row label.
    content((cell-w / 2, y0 - cell-h / 2),
            text(weight: 700, fill: garnet, size: 10pt)[#raw(mode)])

    for j in range(frames.len()) {
      let x0 = (j + 1) * (cell-w + gap)
      rect((x0, y0), (x0 + cell-w, y0 - cell-h),
           fill: b10, stroke: 0.7pt + b70)
      content((x0 + cell-w / 2, y0 - cell-h / 2),
              text(size: 9pt, fill: b50, style: "italic")[mask placeholder])
    }
  }
})
