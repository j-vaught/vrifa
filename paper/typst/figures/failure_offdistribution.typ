// fig:failure_offdistribution — Three off-distribution conditions for
// the integrated configuration. From left to right: bright-bag with AE
// rebound (input_6 mid-fill), dark fabric with low contrast (input_9
// mid-fill), heavily-textured smoothed bag (input_2 mid-fill).
//
// Compile:
//   typst compile paper/typst/figures/failure_offdistribution.typ
//                 paper/typst/figures/failure_offdistribution.pdf

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 9pt)

#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let warmgrey = rgb("#676156")
#let black = rgb("#000000")

#let panels = (
  (
    path: "frames/offdist_bright_bag.png",
    title: "Bright bag with AE rebound",
    sub: "input_6, frame 4050",
  ),
  (
    path: "frames/offdist_dark_fabric.png",
    title: "Dark fabric, low contrast",
    sub: "input_9, frame 7734",
  ),
  (
    path: "frames/offdist_textured_bag.png",
    title: "Smoothed (heavily compressed) bag",
    sub: "input_2, frame 50",
  ),
)

#cetz.canvas({
  import cetz.draw: *

  let panel-w = 3.6
  let panel-h = 2.0
  let gap = 0.20

  for (i, p) in panels.enumerate() {
    let x0 = i * (panel-w + gap)
    content((x0 + panel-w / 2, 0),
            image(p.path, width: panel-w * 1cm, height: panel-h * 1cm))
    content((x0 + panel-w / 2, -panel-h / 2 - 0.20),
            text(size: 8pt, fill: black, weight: 700)[#p.title])
    content((x0 + panel-w / 2, -panel-h / 2 - 0.55),
            text(size: 7.5pt, fill: warmgrey)[#p.sub])
  }
})
