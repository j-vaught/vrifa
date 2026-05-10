// fig:failure_lock_phantom — Three-frame sequence showing the lock-window
// failure on input_6. The integrated config's lock_frames=3 holds
// detections through transient dropouts, which on this sample turns into
// phantom regions when the true wet front pauses.
//
// Frames extracted at indices 405 (pre, ~19% fill), 4050 (mid-fill peak),
// 7694 (post-fill plateau).
//
// Compile:
//   typst compile paper/typst/figures/failure_lock_phantom.typ
//                 paper/typst/figures/failure_lock_phantom.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let warmgrey = rgb("#676156")
#let black = rgb("#000000")

#let panels = (
  ("frames/lock_phantom_pre.png",  "frame 405", "before front advance (~19% fill)"),
  ("frames/lock_phantom_mid.png",  "frame 4050", "mid-fill peak (~90% fill)"),
  ("frames/lock_phantom_post.png", "frame 7694", "post-pause plateau (100% fill)"),
)

#cetz.canvas({
  import cetz.draw: *

  let panel-w = 5.0
  let panel-h = 2.6  // 524 / 1048 ratio ≈ 0.5, plus headroom for labels
  let gap = 0.18

  for (i, (path, label, desc)) in panels.enumerate() {
    let x0 = i * (panel-w + gap)
    // Frame image
    content((x0 + panel-w / 2, 0),
            image(path, width: panel-w * 1cm, height: (panel-w * 0.5) * 1cm))
    // Frame label below image
    let label-y = -(panel-w * 0.5) / 2 - 0.18
    content((x0 + panel-w / 2, label-y),
            text(size: 8pt, fill: black, weight: 700)[#label])
    content((x0 + panel-w / 2, label-y - 0.32),
            text(size: 7.5pt, fill: warmgrey)[#desc])
  }
})
