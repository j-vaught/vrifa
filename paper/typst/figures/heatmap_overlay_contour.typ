// Render-output figure — four panels showing what the pipeline emits
// for one frame (input_1 frame 352, 50 percent fill).
//
// Panel 0: raw BGR input.
// Panel 1: Turbo heatmap of normalized $tilde(D)_t$.
// Panel 2: locked-mask boundary in red over the raw frame.
// Panel 3: Douglas-Peucker contour polygon over the raw frame with
//          vertices marked.
//
// Compile:
//   typst compile paper/typst/figures/heatmap_overlay_contour.typ \
//                 paper/typst/figures/heatmap_overlay_contour.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let txt = rgb("#000000")

// 2-by-2 grid.
#let rows = (
  (
    ([raw input],              "heatmap_overlay_contour_panels/panel0_input.png"),
    ([heatmap of $tilde(D)_t$], "heatmap_overlay_contour_panels/panel1_heatmap.png"),
  ),
  (
    ([locked-mask overlay],    "heatmap_overlay_contour_panels/panel2_overlay.png"),
    ([COCO contour export],    "heatmap_overlay_contour_panels/panel3_contour.png"),
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

  for (r, row) in rows.enumerate() {
    for (c, (label, path)) in row.enumerate() {
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
  }
})
