#let garnet = rgb("#73000A")
#let black = rgb("#000000")
#let white = rgb("#FFFFFF")
#let gray-90 = rgb("#363636")
#let gray-70 = rgb("#5C5C5C")
#let gray-50 = rgb("#A2A2A2")
#let gray-30 = rgb("#C7C7C7")
#let gray-10 = rgb("#ECECEC")
#let warm-gray = rgb("#676156")
#let sandstorm = rgb("#FFF2E3")
#let rose = rgb("#CC2E40")
#let atlantic = rgb("#466A9F")
#let congaree = rgb("#1F414D")
#let horseshoe = rgb("#65780B")
#let grass = rgb("#CED318")
#let honeycomb = rgb("#A49137")

#set page(
  paper: "us-letter",
  margin: (top: 0.55in, bottom: 0.7in, left: 0.62in, right: 0.62in),
)

#set text(
  font: "Times New Roman",
  size: 10pt,
  fill: black,
)

#set par(
  leading: 0.9em,
  justify: true,
)

#set figure.caption(position: bottom)

#let draft-title(title) = align(center)[
  #text(16pt, weight: "bold")[#title]
]

#let draft-authors() = align(center)[
  #text(10pt)[
    J.C. Vaught, Marshall Pigford, Alex Chayer, Declan Johnson, Ramtin Zand, and Darun Barazanchi\
    University of South Carolina, Columbia, South Carolina
  ]
  #v(4pt)
  #text(8.7pt, fill: gray-70)[Author order is provisional for overnight drafting.]
]

#let draft-abstract(body) = block(
  inset: (x: 14pt, y: 7pt),
  stroke: (paint: gray-30, thickness: 0.6pt),
  fill: white,
)[
  #text(10pt, weight: "bold")[Abstract.] #body
]

#let section-heading(label) = align(center)[
  #text(11pt, weight: "bold")[#label]
]

#let subsection-heading(label) = text(10pt, weight: "bold")[#label]

#let small-note(body) = text(8.5pt, fill: gray-70)[#body]

#let metric-chip(label, value) = box(
  inset: (x: 6pt, y: 4pt),
  stroke: (paint: gray-30, thickness: 0.6pt),
  fill: white,
)[
  #text(8.5pt, fill: gray-70)[#label]\ #text(10pt, weight: "bold", fill: garnet)[#value]
]
