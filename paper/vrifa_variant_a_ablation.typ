#import "typst/theme.typ": *
#import "typst/figures.typ": *

#let data = json("data/paper_data.json")

#draft-title[VRIFA: Interpretable Vision-Based Flow-Front Assessment for Vacuum-Assisted Resin Infusion]
#v(6pt)
#draft-authors()
#v(10pt)
#draft-abstract[
  Vacuum-assisted resin infusion is attractive for large composite structures, but process quality depends on whether the advancing wetted region is visible, stable, and measurable early enough to support intervention. This draft frames VRIFA as an interpretable computer-vision pipeline for that problem. The inherited evidence package includes three exported infusion runs totaling 1,006 labeled frames and 4,689 region annotations, plus an ablation study over 91 trials on a 20-frame human-labeled evaluation set. In that study, the baseline configuration reached an objective score of 0.583, while the best tuned configuration reached 0.807 with mask IoU 0.935, Dice/F1 0.966, boundary F1 0.559, and box IoU 0.902. The paper direction in this version is deliberately conservative. It argues that the strongest immediate SciTech contribution is not detector hype, but a tunable, domain-specific, classical-vision workflow that already produces wetted-region masks, contours, and detector-ready exports from ordinary infusion video.
]

#v(10pt)
#align(center)[
  #metric-chip([Scope], [91 ablation and optimization trials])
  #h(10pt)
  #metric-chip([Repo package], [1,006 frames and 4,689 annotations])
  #h(10pt)
  #metric-chip([Best score], [0.807 weighted objective])
]

#v(16pt)
#section-heading[1. Problem Framing]

Vacuum-assisted resin transfer molding and related infusion workflows are attractive because they can produce large structures with comparatively simple tooling, but they are also unforgiving when the resin front deviates from expectation. A technician can often tell by eye when the preform is darkening, when race-tracking appears, or when the wetted region stalls, yet that intuition is hard to archive and even harder to convert into a reusable annotation set. That gap is what makes VRIFA interesting. Instead of treating the camera as a decorative record, it treats the camera as a process-state sensor that can be tuned, exported, and scored.

The current repo supports a strong first conference paper because it already connects three things that are usually separated in practice. It observes the infusion directly, it converts the observation into machine-readable region geometry, and it exposes the design knobs that control whether the extracted front is brittle or usable. The ablation study inherited from Marshall's draft is therefore more than appendix material. It is the main quantitative evidence that the method can be tuned into something materially better than its baseline.

#pipeline-figure()

#section-heading[2. VRIFA Design Space]

The VRIFA pipeline is intentionally classical and modular. Frames are cropped to a region of interest, converted into a chosen color representation, compared against a reference image, smoothed, thresholded, cleaned by morphology, filtered by component area, and finally converted into polygons and boxes for export. That decomposition matters because each stage corresponds to a plausible physical or imaging failure mode. ROI selection suppresses irrelevant background. Darken-only differencing encodes the prior that resin arrival usually darkens the visible fabric. Peak-brightness reference selection helps when the lighting history of a pixel matters more than the first frame. Threshold offsets tune early sensitivity. Morphological cleanup controls fragmentation and pinholes.

This is the part of the story that differentiates VRIFA from a generic thresholding script. The repo does not hide the design space behind a single opaque model checkpoint. It exposes the variables that a manufacturing researcher would actually want to reason about during debugging. That makes the pipeline scientifically legible. It also makes it possible to argue, with evidence, which elements matter most for this application and which ones are interchangeable.

The inherited optimization results already point to a coherent interpretation. The best overall configuration retained the peak-brightness reference and darken-only differencing, switched to RGB, used a threshold offset of approximately -33.8, and tightened the blur, morphology, and minimum-area settings. The strongest signal here is that the wetting prior survives optimization. In other words, the best result was not produced by abandoning the manufacturing intuition embedded in the baseline. It was produced by refining it.

#runtime-figure()

#section-heading[3. Experimental Scope]

The most defensible quantitative section for an extended abstract is the ablation and optimization study rather than an unfinished detector benchmark. The existing study used a 20-frame human-labeled evaluation set and compared baseline, broad ablation, focused ablation, and mixed-variable optimization configurations. The runtime scope is already respectable for a short paper. The full study covered 91 trials and completed in 1 h 21 min 3 s, with the heaviest cost coming from the mixed-variable optimization stage.

Two points are worth emphasizing in the prose. First, the scoring function was multi-objective rather than mask-only. That is the correct choice for flow-front monitoring because a large filled region can still have an unfaithful boundary. Second, the optimization was not prohibitively slow relative to the design-space size explored. That matters because a reviewer can reasonably believe that a lab could retune VRIFA on new videos without a large infrastructure burden.

#agreement-figure()

The baseline-to-optimized jump is the cleanest quantitative figure in the current material. The baseline objective score of 0.583 already indicates that the default pipeline is directionally useful. The optimized score of 0.807 shows that the design space is worth searching. More importantly, every agreement metric improves at once. Mask IoU rises from 0.737 to 0.935, Dice/F1 from 0.847 to 0.966, boundary F1 from 0.206 to 0.559, and box IoU from 0.837 to 0.902. Mean boundary distance also falls sharply, from 138.8 px to 61.5 px. The result is not merely a smoother mask. It is a materially better front estimate.

#montage-figure()

#section-heading[4. What This Version Claims]

This version of the paper should claim four things and no more. First, VRIFA is an interpretable and tunable vision pipeline for wetted-region assessment in infusion video. Second, the repo already demonstrates that the pipeline can export structured annotations at useful scale. Third, a nontrivial ablation study shows that domain-aware tuning materially improves agreement with human labels. Fourth, the same exported outputs can seed downstream detector work even though the detector paper is not yet the cleanest primary contribution.

What this version should not claim is a market-wide or literature-wide state-of-the-art result. The repo and the inherited draft do not yet contain a defensible external benchmark against other front-tracking systems or a complete YOLO training table. That is not a weakness if the abstract is honest about its center of gravity. SciTech audiences are generally comfortable with strong methods papers that emphasize process observability, reproducibility, and interpretable optimization.

#progression-figure()

The progression traces strengthen that manufacturing-monitoring argument. They show that the exported annotations are not isolated frame artifacts. They form coherent region-growth sequences across three runs. That matters because the useful engineering object is not a single segmented frame. It is the temporal evolution of the wetted region.

#section-heading[5. Conclusion]

For an immediate SciTech submission, the ablation-first narrative is the safest and strongest path. It lets the paper stand on quantitative evidence that already exists, uses the downstream detector story as motivation instead of overreach, and presents VRIFA as a practical bridge between ordinary infusion video and machine-readable flow-front geometry. If the full manuscript later grows into a detector paper, this extended abstract still reads as the right first installment because it establishes the annotation engine and the tuning logic that make later detector results credible.

#section-heading[References]

[1] E. W. Washburn, "The Dynamics of Capillary Flow," *Physical Review*, Vol. 17, No. 3, 1921, pp. 273-283.

[2] N. Otsu, "A Threshold Selection Method from Gray-Level Histograms," *IEEE Transactions on Systems, Man, and Cybernetics*, Vol. 9, No. 1, 1979, pp. 62-66.

[3] X. Li, et al., "AI-Based Monitoring of Resin Flow Front Using YOLO," *Materials Research Forum*, 2023.

[4] Ultralytics, "YOLOv8 Documentation," 2023.
