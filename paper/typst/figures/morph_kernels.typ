// Morphological-kernel comparison — three panels showing the cleaned
// mask under each structuring-element shape, all at $k_m = 13$.
//
// All three panels run the same upstream pipeline (input_1 frame 200,
// integrated reference, Otsu + integrated offset, paired close + open
// + area filter) and differ only in the structuring-element shape.
//
// Compile:
//   typst compile paper/typst/figures/morph_kernels.typ \
//                 paper/typst/figures/morph_kernels.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let txt = rgb("#000000")

#let panels = (
  ([ellipse (integrated)], "morph_kernels_panels/morph_ellipse.png"),
  ([rectangle],            "morph_kernels_panels/morph_rect.png"),
  ([cross],                "morph_kernels_panels/morph_cross.png"),
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
