// ROI figure — three panels showing how the region-of-interest mask
// $R$ is constructed.
//
// Canonical frame: input_1.mp4 frame 352 (50% fill, hand-labeled).
//
// Panel 0: raw input frame, no overlay.
// Panel 1: rectangular form with 15% fractional margins per edge.
//          Inside the rectangle the original image is shown clearly;
//          outside the rectangle is cross-hatched in garnet.
// Panel 2: imported PNG form (data/roi_masks/input_1.png). Same
//          inside-clear / outside-hatched rendering as panel 1.
//
// The cross-hatch is painted by the panel-build Python step
// (build_roi_panels.py); CeTZ adds vector text labels and outlines.
//
// Compile:
//   typst compile paper/typst/figures/roi_crop.typ \
//                 paper/typst/figures/roi_crop.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let txt = rgb("#000000")

#let panels = (
  ([Raw input],                       "roi_panels/panel0_input.png"),
  ([Rectangular form, 15\% margin],    "roi_panels/panel1_rect.png"),
  ([Imported PNG mask],                "roi_panels/panel2_mask.png"),
)

#cetz.canvas({
  import cetz.draw: *

  // Geometry — same conventions as the colorspace figure.
  let ch-w = 3.4
  let ch-h = ch-w * 9 / 16          // 16:9 aspect
  let ch-gap-x = 0.30
  let label-h = 0.22
  let label-size = 6pt

  for (i, (label, panel)) in panels.enumerate() {
    let x-center = i * (ch-w + ch-gap-x) + ch-w / 2
    let tile-y = -ch-h / 2
    let label-y = -ch-h - label-h / 2

    // Tile.
    content(
      (x-center, tile-y),
      image(panel, width: ch-w * 1cm, height: ch-h * 1cm),
    )
    // 0.6pt black outline around the tile.
    rect(
      (x-center - ch-w / 2, tile-y + ch-h / 2),
      (x-center + ch-w / 2, tile-y - ch-h / 2),
      stroke: 0.6pt + txt,
      fill: none,
    )
    // Label below the tile.
    content(
      (x-center, label-y),
      text(size: label-size, fill: txt)[#label],
    )
  }
})
