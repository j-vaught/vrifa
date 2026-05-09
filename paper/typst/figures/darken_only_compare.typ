// Darken-only vs Euclidean delta on the same frame.
//
// Three panels: raw input, naive Euclidean CIELAB delta, darken-only
// (L*-channel, negative-clipped) delta. The naive delta lights up on
// specular highlights; the darken-only delta rejects them.
//
// Source frame: input_1 frame 80, with reference frame 1.

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("Inter", "Helvetica"), size: 9pt)

#let garnet = rgb("#73000A")
#let b70    = rgb("#5C5C5C")
#let b10    = rgb("#ECECEC")

#let panel(label, src, caption) = block(
  width: 5.0cm,
)[
  #image(src, width: 100%)
  #v(0.2em)
  #align(center)[
    #text(size: 8pt, weight: 700, fill: garnet)[#label]
    #h(0.3em)
    #text(size: 8pt, fill: b70)[#caption]
  ]
]

#grid(
  columns: (1fr, 1fr, 1fr),
  column-gutter: 0.4cm,
  panel("raw",     "/assets/method/darken_raw.png",
    [input_1 frame 80]),
  panel("naive Δ", "/assets/method/darken_naive.png",
    [Euclidean over CIELAB]),
  panel("darken-only Δ", "/assets/method/darken_only.png",
    [_L\*_-only, clipped to (G − F)⁺]),
)
