// Darken-only delta comparison — three panels on input_2.mp4.
//
// Reference frame: input_2 frame 0 (dry preform).
// Current frame:   input_2 frame 15.
//
// Panel 0: raw current frame.
// Panel 1: naive Euclidean delta in CIELAB. Sign-insensitive, so the
//          bag-side brightening on the left is flagged as signal even
//          though no resin is present there.
// Panel 2: darken-only delta max(0, L_ref - L_cur) on L*. The bag-side
//          brightening is clipped to zero; only the true wetting on
//          the right survives.
//
// Compile:
//   typst compile paper/typst/figures/darken_only_compare.typ \
//                 paper/typst/figures/darken_only_compare.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let txt = rgb("#000000")

#let panels = (
  ([Raw input (input_2 frame 15)],   "darken_only_panels/panel0_input.png"),
  ([Naive Euclidean delta],          "darken_only_panels/panel1_euclidean.png"),
  ([Darken-only delta (integrated)], "darken_only_panels/panel2_darken.png"),
)

#cetz.canvas({
  import cetz.draw: *

  let ch-w = 3.4
  let ch-h = ch-w * 9 / 16
  let ch-gap-x = 0.30
  let label-h = 0.22
  let label-size = 6pt
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
