// Mask-cleanup montage — five panels showing the binary mask flowing
// through stages 9-11 on input_1 frame 200.
//
// Panel 0: normalized response field $tilde(D)_t$ (Turbo heatmap).
// Panel 1: thresholded binary mask (Otsu + integrated offset).
// Panel 2: after morphological closing (elliptical SE, $k_m = 13$).
// Panel 3: after morphological opening (same SE).
// Panel 4: after connected-components area filter ($a_("min") = 400$).
//
// Compile:
//   typst compile paper/typst/figures/mask_cleanup.typ \
//                 paper/typst/figures/mask_cleanup.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let txt = rgb("#000000")

// Two rows: top row 3 panels, bottom row 2 panels (centered).
#let rows = (
  (
    ([(1) normalized response], "mask_cleanup_panels/panel0_response.png"),
    ([(2) thresholded mask],    "mask_cleanup_panels/panel1_threshold.png"),
    ([(3) after closing],       "mask_cleanup_panels/panel2_closed.png"),
  ),
  (
    ([(4) after opening],       "mask_cleanup_panels/panel3_opened.png"),
    ([(5) after area filter],   "mask_cleanup_panels/panel4_area_filter.png"),
  ),
)

#cetz.canvas({
  import cetz.draw: *

  let ch-w = 3.6
  let ch-h = ch-w * 9 / 16
  let ch-gap-x = 0.30
  let ch-gap-y = 0.42
  let label-h = 0.22
  let label-size = 6.5pt
  let label-pad = 0.08
  let row-block-h = ch-h + label-h + label-pad

  // Row width based on the widest (top) row, used to center bottom row.
  let max-cols = calc.max(..rows.map(r => r.len()))
  let row-width(n) = n * ch-w + (n - 1) * ch-gap-x
  let max-row-w = row-width(max-cols)

  for (r, row) in rows.enumerate() {
    let n = row.len()
    let row-w = row-width(n)
    let row-x0 = (max-row-w - row-w) / 2
    let tile-y = -r * (row-block-h + ch-gap-y) - ch-h / 2
    let label-y = tile-y - ch-h / 2 - label-pad - label-h / 2

    for (c, (label, path)) in row.enumerate() {
      let x-center = row-x0 + c * (ch-w + ch-gap-x) + ch-w / 2

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
  }
})
