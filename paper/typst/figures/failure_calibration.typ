// fig:failure_calibration — Contrast a clean monotonic-darkening sample
// (input_5) against a long-fill sample where the dynamic-reference
// calibration window has less data (input_9). Each row shows dry vs
// wet frame; the longer run is more sensitive to a poor calibration.
//
// Compile:
//   typst compile paper/typst/figures/failure_calibration.typ
//                 paper/typst/figures/failure_calibration.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let warmgrey = rgb("#676156")
#let black = rgb("#000000")

#cetz.canvas({
  import cetz.draw: *

  let panel-w = 4.0
  let panel-h = 2.0
  let gap = 0.20

  let rows = (
    (
      label: "input_5  (270 s, clean darkening)",
      color: atlantic,
      frames: (
        ("frames/calib_clean_dry.png", "frame 405 (dry)"),
        ("frames/calib_clean_wet.png", "frame 7694 (wet)"),
      ),
    ),
    (
      label: "input_9  (516 s, longest run)",
      color: garnet,
      frames: (
        ("frames/calib_long_dry.png",  "frame 773 (dry)"),
        ("frames/calib_long_wet.png",  "frame 14695 (wet)"),
      ),
    ),
  )

  for (r, row) in rows.enumerate() {
    let y-base = -r * (panel-h + 1.0)
    // Row label
    content((-0.05, y-base),
            text(size: 9pt, fill: row.color, weight: 700)[#row.label],
            anchor: "east")

    for (i, (path, sub)) in row.frames.enumerate() {
      let x0 = i * (panel-w + gap) + 0.1
      content((x0 + panel-w / 2, y-base),
              image(path, width: panel-w * 1cm, height: panel-h * 1cm))
      content((x0 + panel-w / 2, y-base - panel-h / 2 - 0.20),
              text(size: 7.5pt, fill: black)[#sub])
    }
  }
})
