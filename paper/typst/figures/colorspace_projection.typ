// Colorspace projection figure — same canonical frame shown across all
// nine single-channel projections plus two reference panels.
//
// Canonical frame: input_1.mp4 frame 352 (50% fill, hand-labeled).
//
// Layout:
//
//                                       R       G       B
//     [grayscale reference]
//                                       L*      a*      b*
//     [raw color reference]
//                                       H       S       V
//
// The two reference panels on the left are stacked vertically and
// centered on the gaps between the three channel rows, so they read
// as a separate "anchor" column offset by half a row from the channel
// grid. All single-channel panels render as 8-bit grayscale (no
// colormap) so the reader can directly compare per-channel response
// to wetting.
//
// Compile:
//   typst compile paper/typst/figures/colorspace_projection.typ \
//                 paper/typst/figures/colorspace_projection.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let txt    = rgb("#000000")
#let b70    = rgb("#5C5C5C")

// Row data: three (descriptive channel name, panel path) per row.
#let rows = (
  (
    ([RGB R-channel], "colorspace_panels/rgb_r.png"),
    ([RGB G-channel], "colorspace_panels/rgb_g.png"),
    ([RGB B-channel], "colorspace_panels/rgb_b.png"),
  ),
  (
    ([CIELAB $L^*$-channel], "colorspace_panels/lab_l.png"),
    ([CIELAB $a^*$-channel], "colorspace_panels/lab_a.png"),
    ([CIELAB $b^*$-channel], "colorspace_panels/lab_b.png"),
  ),
  (
    ([HSV H-channel], "colorspace_panels/hsv_h.png"),
    ([HSV S-channel], "colorspace_panels/hsv_s.png"),
    ([HSV V-channel], "colorspace_panels/hsv_v.png"),
  ),
)

#let refs = (
  ([Raw input], "colorspace_panels/ref_color.png"),
  ([Grayscale], "colorspace_panels/ref_gray.png"),
)

#cetz.canvas({
  import cetz.draw: *

  // Geometry. Units are CeTZ canvas units (≈ cm at default scale).
  // All tiles (channel and reference) share the same size.
  let ch-w = 2.6
  let ch-h = ch-w * 9 / 16            // 16:9 aspect
  let ch-gap-x = 0.12
  let ch-gap-y = 0.28                 // larger y-gap so references fit between rows
  let label-h = 0.22                  // per-cell label band below each tile

  // Reference tiles use the same dimensions as channel tiles.
  let ref-w = ch-w
  let ref-h = ch-h

  // Text sizes scale to the figure size. Labels below each tile.
  let ch-label-size  = 6pt
  let ref-label-size = 6pt

  // Layout: each row "block" is tile-on-top + label-below.
  // y goes downward (more negative = further down). Start with row 1
  // tile-top at y = 0. Tile occupies ch-h, then label band below.
  let row-block-h = ch-h + label-h
  // y at top of row r (top of tile):
  let row-top(r) = -r * (row-block-h + ch-gap-y)
  // y at bottom of row r (bottom of label):
  let row-bot(r) = row-top(r) - row-block-h
  // y center of the gap between row r and row r+1:
  let gap-center(r) = (row-bot(r) + row-top(r + 1)) / 2

  // Reference panel tile centers — between row 0/1 and between row 1/2.
  // Bias the y-center upward by label-h/2 so the ref-tile sits visually
  // halfway between the two adjacent channel-tile bands, accounting for
  // the label band beneath the reference itself.
  let ref-tile-y(i) = gap-center(i) + label-h / 2

  // X layout: refs on the left, channel grid on the right. All horizontal
  // gaps between adjacent tiles share the same value so spacing is uniform.
  let ref-x0 = 0
  let ref-x-center = ref-x0 + ref-w / 2
  let grid-x0 = ref-x0 + ref-w + ch-gap-x
  let ch-x-center(c) = grid-x0 + c * (ch-w + ch-gap-x) + ch-w / 2

  // Draw reference panels: tile centered on the inter-row line, label
  // beneath the tile.
  for (i, (label, panel)) in refs.enumerate() {
    let tile-y = ref-tile-y(i)
    let label-y = tile-y - ref-h / 2 - label-h / 2
    // Tile.
    content(
      (ref-x-center, tile-y),
      image(panel, width: ref-w * 1cm, height: ref-h * 1cm),
    )
    // Label below tile.
    content(
      (ref-x-center, label-y),
      text(size: ref-label-size, fill: txt)[#label],
    )
  }

  // Draw the 3 × 3 channel grid: each tile gets a label beneath it.
  for (r, row) in rows.enumerate() {
    let row-y-top = row-top(r)
    let tile-y = row-y-top - ch-h / 2
    let label-y = row-y-top - ch-h - label-h / 2

    for (c, (label, panel)) in row.enumerate() {
      // Tile.
      content(
        (ch-x-center(c), tile-y),
        image(panel, width: ch-w * 1cm, height: ch-h * 1cm),
      )
      // Per-cell label band below tile.
      content(
        (ch-x-center(c), label-y),
        text(size: ch-label-size, fill: txt)[#label],
      )
    }
  }
})
