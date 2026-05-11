// Sample-montage figure — eleven thumbnails (one per input video) at
// the labeled 50%-fill frame, in a 3-by-4 grid with one cell empty.
//
// Compile:
//   typst compile paper/typst/figures/sample_montage.typ \
//                 paper/typst/figures/sample_montage.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let txt = rgb("#000000")

// 3 rows of 4 thumbnails. Twelfth cell is intentionally empty.
#let panels = (
  ("input_1",  "sample_montage_panels/input_1.png"),
  ("input_2",  "sample_montage_panels/input_2.png"),
  ("input_3",  "sample_montage_panels/input_3.png"),
  ("input_4",  "sample_montage_panels/input_4.png"),
  ("input_5",  "sample_montage_panels/input_5.png"),
  ("input_6",  "sample_montage_panels/input_6.png"),
  ("input_7",  "sample_montage_panels/input_7.png"),
  ("input_8",  "sample_montage_panels/input_8.png"),
  ("input_9",  "sample_montage_panels/input_9.png"),
  ("input_10", "sample_montage_panels/input_10.png"),
  ("input_11", "sample_montage_panels/input_11.png"),
)

#let cols = 4
#let rows = 3

#cetz.canvas({
  import cetz.draw: *

  let ch-w = 3.3
  let ch-h = ch-w * 9 / 16
  let ch-gap-x = 0.22
  let ch-gap-y = 0.36
  let label-h = 0.20
  let label-size = 6.5pt
  let label-pad = 0.06
  let row-block-h = ch-h + label-h + label-pad

  for (i, (label, path)) in panels.enumerate() {
    let r = calc.div-euclid(i, cols)
    let c = calc.rem-euclid(i, cols)
    let x-center = c * (ch-w + ch-gap-x) + ch-w / 2
    let tile-y = -r * (row-block-h + ch-gap-y) - ch-h / 2
    let label-y = tile-y - ch-h / 2 - label-pad - label-h / 2

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
