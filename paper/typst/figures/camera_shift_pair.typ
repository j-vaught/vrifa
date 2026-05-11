// Camera-shift figure — three panels demonstrating that the
// registration stage detects the bumped-tripod event in input_1.mp4
// (frames ~65 to 70, ~3.7 px horizontal drift) and warps the live
// frame back into reference coordinates.
//
// Panel 0: pre-event reference frame (frame 63) with a red rectangle
//          marking the zoom region examined in panels 1 and 2.
// Panel 1: Sobel-edge overlay of pre vs post-event (frame 75). Pre
//          edges fill the red channel, post edges fill the green
//          channel, with a dark gray background. Aligned edges show
//          as yellow; the camera shift breaks them into parallel
//          red and green curves whose separation is the 3.7 px shift.
// Panel 2: same Sobel-edge overlay after applying the registration
//          warp to the post-event frame. The parallel red/green
//          curves collapse back into a single yellow edge, showing
//          alignment is restored.
//
// Compile:
//   typst compile paper/typst/figures/camera_shift_pair.typ \
//                 paper/typst/figures/camera_shift_pair.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let txt = rgb("#000000")

#let panels = (
  ([Reference frame (frame 63)],
   "camera_shift_panels/panel0_context.png"),
  ([Uncorrected delta field],
   "camera_shift_panels/panel1_uncorrected.png"),
  ([Corrected delta field after warp],
   "camera_shift_panels/panel2_corrected.png"),
)

#cetz.canvas({
  import cetz.draw: *

  let ch-w = 3.4
  let ch-h = ch-w * 9 / 16          // 16:9 aspect, matches the panels
  let ch-gap-x = 0.30
  let label-h = 0.22
  let label-size = 6pt
  let label-pad = 0.08

  for (i, (label, panel)) in panels.enumerate() {
    let x-center = i * (ch-w + ch-gap-x) + ch-w / 2
    let tile-y = -ch-h / 2
    let label-y = -ch-h - label-pad - label-h / 2

    // Tile.
    content(
      (x-center, tile-y),
      image(panel, width: ch-w * 1cm, height: ch-h * 1cm),
    )
    // 0.6pt black outline.
    rect(
      (x-center - ch-w / 2, tile-y + ch-h / 2),
      (x-center + ch-w / 2, tile-y - ch-h / 2),
      stroke: 0.6pt + txt,
      fill: none,
    )
    // Label below tile.
    content(
      (x-center, label-y),
      text(size: label-size, fill: txt)[#label],
    )
  }
})
