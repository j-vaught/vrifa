// Mask-cleanup sequence.
//
// Five thumbnails of the same frame's mask through the cleanup
// pipeline. Source frame: input_1 frame 200.

#import "@preview/cetz:0.4.0"

#set page(width: auto, height: auto, margin: 6pt, fill: white)
#set text(font: ("Inter", "Helvetica"), size: 8.5pt)

#let garnet = rgb("#73000A")
#let b70    = rgb("#5C5C5C")
#let b10    = rgb("#ECECEC")

#let cell(label, src, caption) = block(
  width: 3.4cm,
)[
  #image(src, width: 100%)
  #v(0.15em)
  #align(center)[
    #text(size: 8pt, weight: 700, fill: garnet)[#label]
  ]
  #align(center)[
    #text(size: 7.5pt, fill: b70)[#caption]
  ]
]

#grid(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  column-gutter: 0.25cm,
  cell([1. delta_norm], "/assets/method/cleanup_1_delta_norm.png",
    [normalized response]),
  cell([2. threshold],  "/assets/method/cleanup_2_binary.png",
    [Otsu plus offset]),
  cell([3. close],      "/assets/method/cleanup_3_close.png",
    [13-px ellipse]),
  cell([4. open],       "/assets/method/cleanup_4_open.png",
    [removes islands]),
  cell([5. min-area],   "/assets/method/cleanup_5_final.png",
    [drops < 400 px components]),
)
