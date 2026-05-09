// Mask-cleanup sequence.
//
// Five thumbnails of the same frame's mask through the cleanup
// pipeline, laid out as 3 + 2 (top row + bottom row).
// Source frame: input_2 frame 60.

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("TeX Gyre Termes", "Times New Roman", "Times"), size: 8.5pt)

#let garnet = rgb("#73000A")
#let b70    = rgb("#5C5C5C")
#let b10    = rgb("#ECECEC")

#let cell(label, src, caption) = block(
  width: 8cm,
)[
  #image(src, width: 100%)
  #v(-0.9em)
  #align(center)[
    #text(size: 8pt, weight: 700, fill: garnet)[#label]
  ]
  #v(-0.9em)
  #align(center)[
    #text(size: 7.5pt, fill: b70)[#caption]
  ]
]

#grid(
  columns: (auto, auto, auto),
  column-gutter: 0.25cm,
  cell([1. delta_norm], "/assets/method/cleanup_1_delta_norm.png",
    [normalized response]),
  cell([2. threshold],  "/assets/method/cleanup_2_binary.png",
    [Otsu plus offset]),
  cell([3. close],      "/assets/method/cleanup_3_close.png",
    [13-px ellipse]),
)

#v(0.3cm)

#align(center)[
  #grid(
    columns: (auto, auto),
    column-gutter: 0.25cm,
    cell([4. open],     "/assets/method/cleanup_4_open.png",
      [removes islands]),
    cell([5. min-area], "/assets/method/cleanup_5_final.png",
      [drops < 400 px components]),
  )
]
