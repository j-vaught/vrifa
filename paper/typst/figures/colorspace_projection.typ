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

// Row data: row label + three (channel name, panel path).
#let rows = (
  (group: "RGB", cells: (
    ("R",  "colorspace_panels/rgb_r.png"),
    ("G",  "colorspace_panels/rgb_g.png"),
    ("B",  "colorspace_panels/rgb_b.png"),
  )),
  (group: "CIELAB", cells: (
    ([$L^*$], "colorspace_panels/lab_l.png"),
    ([$a^*$], "colorspace_panels/lab_a.png"),
    ([$b^*$], "colorspace_panels/lab_b.png"),
  )),
  (group: "HSV", cells: (
    ("H", "colorspace_panels/hsv_h.png"),
    ("S", "colorspace_panels/hsv_s.png"),
    ("V", "colorspace_panels/hsv_v.png"),
  )),
)

#let refs = (
  ("grayscale",  "colorspace_panels/ref_gray.png"),
  ("raw input",  "colorspace_panels/ref_color.png"),
)

#cetz.canvas({
  import cetz.draw: *

  // Geometry. Units are CeTZ canvas units (≈ cm at default scale).
  // Channel tiles: small, 3 per row, 3 rows.
  let ch-w = 2.6
  let ch-h = ch-w * 9 / 16            // 16:9 aspect
  let ch-gap-x = 0.12
  let ch-gap-y = 0.18
  let label-h = 0.18                  // per-cell label band above each tile

  // Reference tiles: slightly larger than channel tiles, 16:9.
  let ref-w = 3.6
  let ref-h = ref-w * 9 / 16

  // Text sizes scale to the figure size rather than the paper's body
  // font: at this tile size, 11pt absolute labels overwhelm the image.
  let ch-label-size  = 8pt
  let ref-label-size = 8.5pt
  let group-size     = 8pt

  // Left-column reference panels are stacked vertically, centered on the
  // gaps between channel rows. Compute the row centers and gap centers.
  // y goes downward (more negative = further down). Start with row 1 top
  // at y = 0; each row block = label-h + ch-h; gap below = ch-gap-y.
  let row-block-h = label-h + ch-h
  // y at top of row r (label band top):
  let row-top(r) = -r * (row-block-h + ch-gap-y)
  // y at bottom of row r (tile bottom):
  let row-bot(r) = row-top(r) - row-block-h
  // y center of the gap between row r and row r+1:
  let gap-center(r) = (row-bot(r) + row-top(r + 1)) / 2

  // Reference panel y-centers — between row 0/1 and between row 1/2.
  let ref-y-centers = (gap-center(0), gap-center(1))

  // X layout: refs on the left, channel grid on the right.
  let ref-x0 = 0
  let ref-x-center = ref-x0 + ref-w / 2
  let grid-x0 = ref-x0 + ref-w + 0.5    // gap between refs and grid
  let ch-x-center(c) = grid-x0 + c * (ch-w + ch-gap-x) + ch-w / 2

  // Draw reference panels with a label band above each tile.
  for (i, (label, panel)) in refs.enumerate() {
    let yc = ref-y-centers.at(i)
    let label-y = yc + ref-h / 2 + label-h / 2 + 0.02
    let tile-y = yc
    // Label.
    content(
      (ref-x-center, label-y),
      text(weight: 700, size: ref-label-size, fill: txt)[#label],
    )
    // Tile.
    content(
      (ref-x-center, tile-y),
      image(panel, width: ref-w * 1cm, height: ref-h * 1cm),
    )
  }

  // Optional thin group label on the right margin of each row.
  let group-label-x = grid-x0 + 3 * ch-w + 2 * ch-gap-x + 0.25

  // Draw the 3 × 3 channel grid.
  for (r, row) in rows.enumerate() {
    let row-y-top = row-top(r)
    let label-y = row-y-top - label-h / 2
    let tile-y = row-y-top - label-h - ch-h / 2

    // Cells.
    for (c, (label, panel)) in row.cells.enumerate() {
      // Per-cell label band.
      content(
        (ch-x-center(c), label-y),
        text(weight: 700, size: ch-label-size, fill: txt)[#label],
      )
      // Tile.
      content(
        (ch-x-center(c), tile-y),
        image(panel, width: ch-w * 1cm, height: ch-h * 1cm),
      )
    }

    // Group label on the right margin of the row.
    content(
      (group-label-x, tile-y),
      text(weight: 700, size: group-size, fill: b70)[#row.group],
      anchor: "west",
    )
  }
})
