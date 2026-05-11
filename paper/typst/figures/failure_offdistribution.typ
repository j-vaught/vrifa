// fig:failure_offdistribution — Three off-distribution conditions
// shown as a single row of panels, matching the format used by the
// other Method/Discussion figures.
//
// Compile:
//   typst compile paper/typst/figures/failure_offdistribution.typ
//                 paper/typst/figures/failure_offdistribution.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let txt = rgb("#000000")

#let panels = (
  ([Bright bag with AE rebound],          "frames/offdist_bright_bag.png"),
  ([Dark fabric, low contrast],           "frames/offdist_dark_fabric.png"),
  ([Smoothed (heavily compressed) bag],   "frames/offdist_textured_bag.png"),
)

#cetz.canvas({
  import cetz.draw: *

  let ch-w = 3.6
  let ch-h = ch-w * 9 / 16
  let ch-gap-x = 0.30
  let label-h = 0.22
  let label-size = 6.5pt
  let label-pad = 0.08

  for (i, (label, path)) in panels.enumerate() {
    let x-center = i * (ch-w + ch-gap-x) + ch-w / 2
    let tile-y = -ch-h / 2
    let label-y = -ch-h - label-pad - label-h / 2

    content(
      (x-center, tile-y),
      image(path, width: ch-w * 1cm, height: ch-h * 1cm),
    )
    rect(
      (x-center - ch-w / 2, tile-y + ch-h / 2),
      (x-center + ch-w / 2, tile-y - ch-h / 2),
      stroke: 0.6pt + txt,
      fill: none,
    )
    content(
      (x-center, label-y),
      text(size: label-size, fill: txt)[#label],
    )
  }
})
