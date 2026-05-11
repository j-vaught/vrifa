// Pre/post-blur figure — two-row by three-column comparison of how
// pre-delta blur affects the working channel (row 1) and the eventual
// delta field after post-delta blur (row 2).
//
// Data: input_1.mp4 frame 10 (reference) vs frame 75 (current), with
// camera-shift stabilization deliberately turned off. The ~3.7 px
// residual bump produces high-frequency edge artifacts on every
// laminate edge, the regime where pre- and post-blur matter most.
//
// Columns: k_p = 0 (integrated, no pre-blur), 5, 9.
// All bottom-row deltas pass through the same k_b = 9 post-delta blur
// after their per-column pre-blur.
//
// Compile:
//   typst compile paper/typst/figures/pre_post_blur.typ \
//                 paper/typst/figures/pre_post_blur.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let txt = rgb("#000000")

// Two rows of three panels each.
//   row 0: working channel $L^*$ after pre-delta blur (grayscale).
//   row 1: resulting $D_t$ after pre-blur + $k_b = 9$ post-blur.
#let panels = (
  (
    ([$L^*$, $k_p = 0$ (integrated)], "pre_post_blur_panels/row0_kp0.png"),
    ([$L^*$, $k_p = 5$],              "pre_post_blur_panels/row0_kp5.png"),
    ([$L^*$, $k_p = 9$],              "pre_post_blur_panels/row0_kp9.png"),
  ),
  (
    ([pre-delta blur = 0 (integrated)], "pre_post_blur_panels/row1_kp0.png"),
    ([pre-delta blur = 5],              "pre_post_blur_panels/row1_kp5.png"),
    ([pre-delta blur = 9],              "pre_post_blur_panels/row1_kp9.png"),
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

  for (r, row) in panels.enumerate() {
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
