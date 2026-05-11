// Reference-mode figure — same canonical input frame (input_1 frame
// 352, 50% fill) compared against six different choices of reference
// $G_t$ via the pipeline's darken-only delta $D_t = max(0, G_t - F_t)$,
// rendered as a Turbo heatmap on a shared intensity scale.
//
// Top row: integrated (first-frame + peak), running EMA, previous fixed-offset.
// Bottom row: absolute pinned frame, dynamic sqrt-area, dynamic linear-lag.
//
// The integrated panel is the configuration the paper uses; its tile
// outline switches from neutral black to garnet to mark it.
//
// Compile:
//   typst compile paper/typst/figures/reference_modes.typ \
//                 paper/typst/figures/reference_modes.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet = rgb("#73000A")
#let txt    = rgb("#000000")

// Two rows of three panels each. Order matches the placeholder.
#let panels = (
  (([first-frame + peak (integrated)], "reference_modes_panels/panel0_integrated.png",     true),
   ([running EMA],                     "reference_modes_panels/panel1_running.png",         false),
   ([previous, fixed offset],          "reference_modes_panels/panel2_previous.png",        false)),
  (([absolute pinned frame],           "reference_modes_panels/panel3_absolute.png",        false),
   ([dynamic sqrt-area],               "reference_modes_panels/panel4_dynamic_sqrt.png",    false),
   ([dynamic linear lag],              "reference_modes_panels/panel5_dynamic_linear.png",  false)),
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
    for (c, cell) in row.enumerate() {
      let (label, path, is-integrated) = cell
      let x-center = c * (ch-w + ch-gap-x) + ch-w / 2
      let tile-y = -r * (row-block-h + ch-gap-y) - ch-h / 2
      let label-y = tile-y - ch-h / 2 - label-pad - label-h / 2

      // Tile.
      content(
        (x-center, tile-y),
        image(path, width: ch-w * 1cm, height: ch-h * 1cm),
      )
      // Outline: garnet 1.2pt for the integrated config, neutral 0.6pt
      // black for the others.
      rect(
        (x-center - ch-w / 2, tile-y + ch-h / 2),
        (x-center + ch-w / 2, tile-y - ch-h / 2),
        stroke: if is-integrated { 1.2pt + garnet } else { 0.6pt + txt },
        fill: none,
      )
      // Label.
      content(
        (x-center, label-y),
        text(
          size: label-size,
          fill: if is-integrated { garnet } else { txt },
          weight: if is-integrated { 600 } else { 400 },
        )[#label],
      )
    }
  }
})
