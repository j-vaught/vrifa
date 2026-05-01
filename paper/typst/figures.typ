#import "@preview/cetz:0.5.0": canvas, draw
#import "theme.typ": *

#let data = json("../data/paper_data.json")

#let perf = data.performance
#let runtime = data.runtime
#let showcase = data.showcase_assets
#let progression = data.sampled_progression
#let repo-totals = data.repo_totals

#let img-path(rel) = "../" + rel

#let pipeline-figure() = figure(
  canvas(length: 5.5mm, {
    import draw: *

    rect((1.2, 17.0), (11.4, 19.2), stroke: black, fill: sandstorm)
    content((6.3, 18.1), [Input infusion video])

    rect((1.2, 13.9), (11.4, 16.1), stroke: black, fill: gray-10)
    content((6.3, 15.0), [
      ROI crop, color-space conversion, and
      darken-only or absolute differencing
    ])

    rect((1.2, 10.8), (11.4, 13.0), stroke: black, fill: gray-10)
    content((6.3, 11.9), [
      Peak-brightness or lagged reference,
      Gaussian smoothing, and threshold offset
    ])

    rect((1.2, 7.7), (11.4, 9.9), stroke: black, fill: gray-10)
    content((6.3, 8.8), [
      Morphology, component filtering,
      temporal locking, and contour extraction
    ])

    rect((0.2, 2.0), (4.0, 5.2), stroke: black, fill: white)
    rect((4.4, 2.0), (8.2, 5.2), stroke: black, fill: white)
    rect((8.6, 2.0), (12.4, 5.2), stroke: black, fill: white)
    content((2.1, 3.6), [Region masks and overlays])
    content((6.3, 3.6), [COCO, YOLO, and Darknet exports])
    content((10.5, 3.6), [Detector-ready training sets])

    line((6.3, 17.0), (6.3, 16.1), mark: (end: ">"), stroke: (paint: black, thickness: 0.8pt))
    line((6.3, 13.9), (6.3, 13.0), mark: (end: ">"), stroke: (paint: black, thickness: 0.8pt))
    line((6.3, 10.8), (6.3, 9.9), mark: (end: ">"), stroke: (paint: black, thickness: 0.8pt))
    line((6.3, 7.7), (6.3, 6.2), mark: (end: ">"), stroke: (paint: black, thickness: 0.8pt))
    line((6.3, 6.2), (2.1, 5.2), mark: (end: ">"), stroke: (paint: black, thickness: 0.8pt))
    line((6.3, 6.2), (6.3, 5.2), mark: (end: ">"), stroke: (paint: black, thickness: 0.8pt))
    line((6.3, 6.2), (10.5, 5.2), mark: (end: ">"), stroke: (paint: black, thickness: 0.8pt))
  }),
  caption: [
    VRIFA converts infusion video into region masks, contour geometry, and detector-ready annotations using an interpretable sequence of classical vision steps.
  ],
)

#let agreement-figure() = figure(
  canvas(length: 6mm, {
    import draw: *
    let metrics = (
      ("Objective", perf.baseline.objective_score, perf.optimized.objective_score),
      ("Mask IoU", perf.baseline.mask_iou, perf.optimized.mask_iou),
      ("Dice/F1", perf.baseline.dice_f1, perf.optimized.dice_f1),
      ("Boundary F1", perf.baseline.boundary_f1, perf.optimized.boundary_f1),
      ("Box IoU", perf.baseline.box_iou, perf.optimized.box_iou),
    )
    let chart-left = 1.4
    let chart-bottom = 1.8
    let chart-top = 10.2

    line((chart-left, chart-bottom), (chart-left, chart-top), stroke: (paint: black, thickness: 0.8pt))
    line((chart-left, chart-bottom), (16.2, chart-bottom), stroke: (paint: black, thickness: 0.8pt))

    for tick in range(0, 6) {
      let y = chart-bottom + tick * 1.6
      line((chart-left - 0.15, y), (chart-left, y), stroke: (paint: black, thickness: 0.6pt))
      content((chart-left - 0.28, y), anchor: "east", text(8pt)[#(tick * 0.2)])
    }

    for (index, metric) in metrics.enumerate() {
      let (label, baseline, optimized) = metric
      let group-left = 2.2 + index * 2.7
      let bar-width = 0.85
      let baseline-height = baseline * 8
      let optimized-height = optimized * 8

      rect((group-left, chart-bottom), (group-left + bar-width, chart-bottom + baseline-height), fill: gray-30, stroke: black)
      rect((group-left + 1.0, chart-bottom), (group-left + 1.0 + bar-width, chart-bottom + optimized-height), fill: garnet, stroke: black)
      content((group-left + 0.43, chart-bottom + baseline-height + 0.35), text(7.5pt)[#baseline])
      content((group-left + 1.43, chart-bottom + optimized-height + 0.35), text(7.5pt, fill: garnet)[#optimized])
      content((group-left + 0.95, 0.95), box(width: 1.7cm, align(center)[#text(7.5pt, fill: gray-70)[#label]]))
    }

    rect((10.8, 9.8), (11.2, 10.2), fill: gray-30, stroke: none)
    content((11.45, 10.0), anchor: "west", text(8pt)[Baseline])
    rect((13.0, 9.8), (13.4, 10.2), fill: garnet, stroke: none)
    content((13.65, 10.0), anchor: "west", text(8pt)[Best optimized])
  }),
  caption: [
    Agreement metrics from the 20-frame human-labeled evaluation set. The tuned configuration improves every plotted agreement measure over the baseline.
  ],
)

#let runtime-figure() = figure(
  canvas(length: 5.8mm, {
    import draw: *
    let max-minutes = calc.max(..runtime.map(row => row.seconds / 60))
    let chart-left = 1.2
    let chart-bottom = 1.6
    let chart-right = 15.5

    line((chart-left, chart-bottom), (chart-left, 10.8), stroke: (paint: black, thickness: 0.8pt))
    line((chart-left, chart-bottom), (chart-right, chart-bottom), stroke: (paint: black, thickness: 0.8pt))

    for tick in range(0, 5) {
      let y = chart-bottom + tick * 2.2
      line((chart-left - 0.15, y), (chart-left, y), stroke: (paint: black, thickness: 0.6pt))
      content((chart-left - 0.3, y), anchor: "east", text(8pt)[#(calc.round(max-minutes * tick / 4, digits: 1))])
    }

    for (index, row) in runtime.enumerate() {
      let minutes = row.seconds / 60
      let x0 = 2.1 + index * 3.2
      let x1 = x0 + 1.6
      let height = 8.6 * minutes / max-minutes
      let fill = if index == 3 { garnet } else { atlantic }
      rect((x0, chart-bottom), (x1, chart-bottom + height), fill: fill, stroke: black)
      content((x0 + 0.8, chart-bottom + height + 0.35), text(7.5pt)[#row.trials + " trials"])
      content((x0 + 0.8, 0.85), box(width: 2.0cm, align(center)[#text(7.4pt, fill: gray-70)[#row.stage]]))
    }
  }),
  caption: [
    Runtime scope for the inherited ablation study. The overnight drafts treat the 91-trial optimization study as the strongest quantitative evidence currently available in the repo-plus-draft package.
  ],
)

#let progression-figure() = figure(
  canvas(length: 6mm, {
    import draw: *
    let chart-left = 1.5
    let chart-bottom = 1.6
    let chart-width = 12.8
    let chart-height = 7.6
    let run-colors = (
      input1: garnet,
      input2: atlantic,
      input3: horseshoe,
    )

    line((chart-left, chart-bottom), (chart-left, chart-bottom + chart-height), stroke: (paint: black, thickness: 0.8pt))
    line((chart-left, chart-bottom), (chart-left + chart-width, chart-bottom), stroke: (paint: black, thickness: 0.8pt))

    for tick in range(0, 6) {
      let x = chart-left + tick * chart-width / 5
      line((x, chart-bottom), (x, chart-bottom - 0.15), stroke: (paint: black, thickness: 0.6pt))
      content((x, chart-bottom - 0.42), anchor: "north", text(8pt)[#(tick * 20) + "%"])
    }

    for tick in range(0, 6) {
      let y = chart-bottom + tick * chart-height / 5
      line((chart-left - 0.15, y), (chart-left, y), stroke: (paint: black, thickness: 0.6pt))
      content((chart-left - 0.3, y), anchor: "east", text(8pt)[#(tick * 20) + "%"])
    }

    for slug in ("input1", "input2", "input3") {
      let pts = progression
        .filter(row => row.slug == slug)
        .map(row => (
          chart-left + chart-width * row.time_norm,
          chart-bottom + chart-height * row.wet_norm,
        ))
      line(..pts, stroke: (paint: run-colors.at(slug), thickness: 1.1pt))
    }

    line((9.7, 9.9), (10.7, 9.9), stroke: (paint: garnet, thickness: 1.1pt))
    content((10.95, 9.9), anchor: "west", text(8pt)[Run A])
    line((9.7, 9.3), (10.7, 9.3), stroke: (paint: atlantic, thickness: 1.1pt))
    content((10.95, 9.3), anchor: "west", text(8pt)[Run B])
    line((9.7, 8.7), (10.7, 8.7), stroke: (paint: horseshoe, thickness: 1.1pt))
    content((10.95, 8.7), anchor: "west", text(8pt)[Run C])
    content((7.8, 0.35), text(8pt)[Normalized infusion time])
    content((0.4, 5.2), angle: 90deg, text(8pt)[Normalized detected-region fraction])
  }),
  caption: [
    Normalized region-growth traces derived directly from the exported annotations. Even without a detector benchmark, the repo already supports a process-progression story across three infusion runs.
  ],
)

#let montage-figure() = figure(
  canvas(length: 6mm, {
    import draw: *
    let img-width = 4.0
    let img-height = 2.3
    let gap = 0.35

    for (index, item) in showcase.enumerate() {
      let x = 0.2 + index * (img-width + gap)
      content((x, 5.35), image(img-path(item.raw), width: img-width * 1cm), anchor: "south-west")
      rect((x, 5.35), (x + img-width, 5.35 + img-height), stroke: black)
      content((x + img-width / 2, 7.95), text(7.5pt, fill: gray-70)[#item.label])

      content((x, 0.7), image(img-path(item.overlay), width: img-width * 1cm), anchor: "south-west")
      rect((x, 0.7), (x + img-width, 0.7 + img-height), stroke: black)
      content((x + img-width / 2, 3.3), text(7.5pt, fill: gray-70)[VRIFA overlay])
    }

    content((13.0, 6.45), angle: 90deg, text(8pt, weight: "bold")[Raw frames])
    content((13.0, 1.85), angle: 90deg, text(8pt, weight: "bold")[Region estimates])
  }),
  caption: [
    Representative progression snapshots from Run A. The same exported annotations that feed the quantitative plots also produce the visual panels used across the three draft variants.
  ],
)

#let detector-bridge-figure() = figure(
  canvas(length: 6mm, {
    import draw: *
    content((0.3, 2.6), image(img-path(showcase.at(1).raw), width: 4.2cm), anchor: "south-west")
    rect((0.3, 2.6), (4.5, 5.0), stroke: black)
    content((2.4, 5.3), text(8pt, fill: gray-70)[Infusion frame])

    rect((5.3, 2.95), (9.4, 4.65), stroke: black, fill: sandstorm)
    content((7.35, 3.8), [
      VRIFA polygon and box export\
      COCO, YOLO segmentation, and Darknet
    ])

    content((10.2, 2.6), image(img-path(data.yolo_demo), width: 4.2cm), anchor: "south-west")
    rect((10.2, 2.6), (14.4, 5.0), stroke: black)
    content((12.3, 5.3), text(8pt, fill: gray-70)[Detector overlay demo])

    line((4.55, 3.8), (5.15, 3.8), mark: (end: ">"), stroke: (paint: black, thickness: 0.8pt))
    line((9.45, 3.8), (10.05, 3.8), mark: (end: ">"), stroke: (paint: black, thickness: 0.8pt))
  }),
  caption: [
    The dataset-to-detector path already exists in the repo. This makes a credible draft angle even if the full detector benchmark still needs to be written up cleanly.
  ],
)
