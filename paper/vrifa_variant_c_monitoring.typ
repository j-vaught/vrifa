#import "typst/theme.typ": *
#import "typst/figures.typ": *

#let data = json("data/paper_data.json")

#draft-title[Vision-Based Process Monitoring for Vacuum-Assisted Resin Infusion Using VRIFA]
#v(6pt)
#draft-authors()
#v(10pt)
#draft-abstract[
  Resin infusion quality depends on whether the advancing wetted region can be observed, localized, and interpreted quickly enough to support manufacturing decisions. This draft frames VRIFA as a process-monitoring system for that requirement. The repo already contains three infusion runs, 1,006 labeled frames, 4,689 exported region annotations, and per-run summaries showing average compute times near 41 ms per processed frame on the sample exports. An inherited 91-trial optimization study further shows that VRIFA can be tuned from a baseline objective score of 0.583 to 0.807 on a 20-frame human-labeled subset, with boundary F1 improving from 0.206 to 0.559. This version emphasizes temporal observability, interpretable front tracking, and practical deployment value in the lab, while treating detector training as a secondary downstream benefit rather than the headline claim.
]

#v(10pt)
#align(center)[
  #metric-chip([Per-frame compute], [about 41 ms on sample runs])
  #h(10pt)
  #metric-chip([Boundary F1], [0.206 to 0.559 after tuning])
  #h(10pt)
  #metric-chip([Use case], [Process observability and traceability])
]

#v(16pt)
#section-heading[1. Monitoring Instead of Mere Visualization]

Many infusion videos are recorded for documentation but not exploited as quantitative process signals. That leaves useful information on the table. The darkening of the preform, the branching of the fill pattern, and the onset of irregular front behavior are all visible phenomena with process significance. A monitoring paper should therefore argue that the camera is not only a passive witness. It can be converted into a consistent source of region geometry and temporal progression information.

VRIFA is a good fit for that argument because its outputs are not limited to a single pretty overlay. The pipeline emits masks, contours, boxes, and annotation files that can be traced frame by frame. In process-monitoring terms, that means the method produces a time-indexed estimate of the wetted region rather than a one-off image enhancement.

#pipeline-figure()

#section-heading[2. Interpretable Design Choices]

The case for VRIFA in a manufacturing-monitoring context depends on interpretability. A technician or reviewer should be able to understand why the method changed its answer when lighting, ROI choice, or threshold sensitivity changed. The pipeline makes that possible. Darken-only differencing corresponds to the expected appearance change of resin arrival. Peak-brightness reference selection helps when the scene brightens before it wets. Morphology and minimum-area filtering control how aggressively the method rejects isolated artifacts. Temporal locking suppresses single-frame flicker.

These are not abstract software knobs. They are proxies for the practical issues that make shop-floor video difficult: glare, wrinkles in the bagging film, race-tracking pathways, camera repositioning, and heterogeneous contrast across the preform. The ability to explain the knobs is part of the monitoring value because it tells the user how to retune the method when the setup changes.

#montage-figure()

#section-heading[3. Temporal Progression as a Process Signal]

The progression traces derived directly from the exported annotations are one of the most useful figures in the current package. They show that VRIFA can turn each run into a compact region-growth history. This is important because manufacturing decisions are usually made from trends, not from isolated frames. A stall, an unexpectedly rapid edge advance, or a late-stage asymmetry is easier to recognize in a progression trace than in a folder of images.

#progression-figure()

The three-run normalized progression plot should be read as a process signature figure rather than a detector metric. Each trace captures how the detected region evolves over normalized infusion time. That structure opens the door to later work on anomaly detection, closed-loop intervention, or comparison between layups and flow-media configurations. Even in this extended-abstract form, it demonstrates that VRIFA is producing more than static masks. It is producing a time-resolved monitoring signal.

#runtime-figure()

#section-heading[4. Quantitative Accuracy and Runtime]

Monitoring methods still need accuracy, especially around the front boundary where decisions about dry spots or uneven advance are made. The inherited optimization study shows that boundary quality was the main weakness of the default pipeline and the main benefit of tuning. Boundary F1 rises from 0.206 to 0.559, while mean boundary distance drops from 138.8 px to 61.5 px. Those are meaningful changes for a monitoring-oriented paper because the front location is more process-relevant than coarse mask overlap alone.

#agreement-figure()

The runtime story is also better than it first appears. The repo run summaries show average compute times of roughly 41 ms per processed frame on the exported sample runs, while the scored optimized configuration in the inherited draft reports 120.7 ms per labeled frame under the evaluation pipeline. That is not yet a hard real-time claim, but it is comfortably in the range of practical near-real-time post-processing and lower-rate online monitoring for infusion experiments. For a SciTech extended abstract, that is enough to support a deployment-oriented argument without overselling maturity.

#section-heading[5. What This Version Claims]

This version should claim that VRIFA provides a practical, interpretable, vision-based monitoring pipeline for resin infusion experiments. It should claim that the method produces temporally coherent region-growth information, not just static masks. It should claim that tuning materially improves front-localization quality. It should not center the paper on detector benchmarking, because that is not the cleanest evidence currently in hand.

That positioning also fits the likely audience. A process-monitoring paper can speak to manufacturing researchers, sensing researchers, and applied computer-vision reviewers without depending on a complete machine-learning comparison study. It turns the strengths of the present repo into the thesis of the paper instead of treating them as warm-up material.

#section-heading[6. Conclusion]

The monitoring-first narrative is the most application-facing of the three draft variants. It presents VRIFA as a reproducible way to convert infusion video into process-state estimates, progression traces, and reviewable visual evidence. If the eventual full manuscript expands into control, anomaly detection, or detector training, this version still serves as a strong conference entry point because it establishes the process-observability problem and shows that the current system already addresses it in a measurable way.

#section-heading[References]

[1] E. W. Washburn, "The Dynamics of Capillary Flow," *Physical Review*, Vol. 17, No. 3, 1921, pp. 273-283.

[2] N. Otsu, "A Threshold Selection Method from Gray-Level Histograms," *IEEE Transactions on Systems, Man, and Cybernetics*, Vol. 9, No. 1, 1979, pp. 62-66.

[3] X. Li, et al., "AI-Based Monitoring of Resin Flow Front Using YOLO," *Materials Research Forum*, 2023.

[4] Ultralytics, "YOLOv8 Documentation," 2023.
