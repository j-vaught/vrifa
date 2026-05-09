// Labeling protocol illustration.
//
// One example frame from the 55-frame labeling pass with the operator's
// polygon overlaid on the source. Replaced once labels arrive.

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("Inter", "Helvetica"), size: 9pt)

#let garnet = rgb("#73000A")
#let rose   = rgb("#CC2E40")
#let b70    = rgb("#5C5C5C")
#let b10    = rgb("#ECECEC")

#cetz.canvas({
  import cetz.draw: *

  // Placeholder frame.
  rect((0, 0), (10, -6),
       fill: b10, stroke: 0.7pt + b70)
  content((5, -3),
          text(size: 11pt, fill: b70, style: "italic")[labeling-protocol example placeholder])

  // Indicative polygon (drawn freehand-like so the figure looks like
  // a real labeled frame even without the underlying image).
  line((1.5, -1.6), (3.0, -1.0), (5.5, -0.9), (7.5, -1.4),
       (8.6, -2.5), (8.0, -4.2), (6.5, -5.0), (4.0, -5.2),
       (2.0, -4.6), (1.2, -3.5), close: true,
       stroke: 1.6pt + rose)

  // Caption-like overlay chip.
  rect((0.2, -5.55), (3.2, -5.95), fill: garnet, stroke: none)
  content((1.7, -5.75),
          text(size: 8.5pt, fill: white, weight: 700)[wet polygon])
})
