// Teaser figure (page 1).
//
// Placeholder until a representative overlay frame is regenerated and
// dropped into paper/assets/teaser_overlay.png. The chip with the
// headline numbers fills in once the agreement metrics resolve.

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 10pt)

#let garnet = rgb("#73000A")
#let b70    = rgb("#5C5C5C")
#let b10    = rgb("#ECECEC")

// TODO replace placeholder with image("../../assets/teaser_overlay.png")
// once the regenerated overlay frame is on disk.

#cetz.canvas({
  import cetz.draw: *

  rect((0, 0), (12, -7),
       fill: b10, stroke: 0.8pt + b70)
  content((6, -3.5),
          text(size: 12pt, fill: b70, style: "italic")[teaser overlay placeholder — input_1 frame ~200])

  // Headline-numbers chip at bottom-right.
  rect((8, -6.4), (11.8, -6.95), fill: garnet, stroke: none)
  content((9.9, -6.67),
          text(size: 9pt, fill: white, weight: 700)[706 frames · 30 fps · IoU TBD])
})
