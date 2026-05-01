#import "typst/theme.typ": *
#import "typst/figures.typ": *

#let data = json("data/paper_data.json")

#draft-title[From Infusion Video to Detector Labels: VRIFA as an Annotation Engine for Composite Manufacturing]
#v(6pt)
#draft-authors()
#v(10pt)
#draft-abstract[
  Computer-vision models for composite manufacturing are often limited less by architecture choice than by the absence of labeled process imagery. This draft therefore frames VRIFA primarily as a label-generation engine rather than as a finished detector. The current evidence package contains three infusion runs with 1,006 exported frames and 4,689 region annotations, together with a 91-trial ablation study on a 20-frame human-labeled set. Those optimization results matter because a synthetic or algorithmic labeler is only useful if its outputs can be tuned toward human agreement. The best inherited configuration reaches an objective score of 0.807 with mask IoU 0.935 and box IoU 0.902. This version argues that the most compelling near-term paper is a bootstrapping paper. VRIFA turns ordinary infusion video into COCO-, YOLO-, and Darknet-style training assets, and a qualitative downstream detector overlay already exists in the repo.
]

#v(10pt)
#align(center)[
  #metric-chip([Label volume], [4,689 exported regions])
  #h(10pt)
  #metric-chip([Frame volume], [1,006 labeled frames])
  #h(10pt)
  #metric-chip([Downstream use], [Detector-ready exports in repo])
]

#v(16pt)
#section-heading[1. Why This Paper Angle Matters]

For many manufacturing-vision problems, the scarce resource is not a segmentation model. It is a labeled dataset tied to the actual process of interest. Resin infusion is a textbook example. The lab can record video easily, but manually tracing the advancing wetted region across hundreds of frames is expensive and inconsistent. That makes a good annotation engine intrinsically valuable. If the algorithm can convert video into region polygons and boxes at useful quality, it reduces the cost of building the next generation of learned detectors.

This framing aligns well with what the repo already proves. The current public material contains multiple exported runs, a qualitative YOLO-style overlay, and a sizeable amount of annotation volume produced without a human labeling campaign. The unfinished part is the polished detector benchmark table. That does not weaken the draft if the paper explicitly treats detector training as the downstream beneficiary rather than the sole claimed contribution.

#pipeline-figure()

#section-heading[2. VRIFA as a Structured Label Generator]

VRIFA takes a position that is both practical and scientifically interesting. It does not attempt to replace process knowledge with a black box. Instead, it uses process knowledge to manufacture labels. Darken-only differencing encodes the fact that resin arrival generally darkens the visible fabric. Peak-brightness reference selection absorbs illumination history. Threshold offsets and morphology control whether the generated label hugs the front or breaks into implausible fragments. Because every stage is explicit, the annotation engine is tunable.

That tunability is not a cosmetic benefit. If algorithmic labels are going to seed a YOLO training run, they must be improvable. Otherwise the detector inherits fixed label noise. The inherited ablation study is therefore central to this version of the paper. It shows that the export quality is not locked to a naive baseline. It can be pushed closer to human-labeled masks and boxes through systematic search over reference strategy, color space, threshold offset, blur, morphology, and component filtering.

#agreement-figure()

The agreement plot is what makes the detector-bootstrapping story credible. The tuned labeler is not merely prolific. It is measurably better than its default version on the available labeled subset. The baseline objective score of 0.583 rises to 0.807 after tuning, while box IoU rises from 0.837 to 0.902. For a dataset-generation narrative, those box-level improvements are especially important because detectors are directly sensitive to localization noise in their supervision.

#detector-bridge-figure()

#section-heading[3. Evidence Already Present in the Repo]

The repo-plus-draft package already supports a useful data-production claim. Three runs contribute 1,006 exported frames and 4,689 region annotations. Run A alone contributes 706 frames, which is enough to move beyond toy examples. The run summaries also show that the pipeline can operate across both higher-frame-rate and lower-frame-rate recordings while preserving the same basic export logic. That breadth helps position VRIFA as a data-engineering tool rather than a single-video curiosity.

Just as importantly, the repo contains a qualitative downstream detector overlay. A short paper does not need to oversell that artifact. It simply needs to show that the label-generation path is not hypothetical. VRIFA outputs can already be consumed by detector tooling, which is precisely why the export formats matter.

#montage-figure()

The progression montage shows why automated labeling is useful in this domain. The front shape is not a neat geometric object. It branches, fills, merges, and responds to local flow paths. Those irregular shapes are exactly the sort of supervision that becomes tedious to annotate by hand but valuable to expose to a detector. Classical vision is doing the expensive tracing work upfront.

#progression-figure()

#section-heading[4. What This Version Claims]

This version should claim that VRIFA is a practical label-generation bridge between infusion video and trainable detection datasets. It should claim that the generated labels can be tuned toward human agreement and exported in common detection formats. It should claim that the repo already demonstrates qualitative downstream detector use. It should not claim a finished detector benchmark against the broader market, because the current public package does not yet include the clean comparative table needed to defend that statement.

That restraint is not a concession. It is a strategic choice. Many conference papers fail because they claim a detector story without enough detector evidence. This draft avoids that trap by turning the annotation engine itself into the contribution. The full manuscript due later can still grow into an end-to-end learned-detector paper with training curves, held-out metrics, and comparison baselines.

#section-heading[5. Conclusion]

If the goal is to submit something credible now while preserving room to expand later, the dataset-to-detector framing is strong. It uses the repo's actual strengths, acknowledges that the learned-model story is still maturing, and still gives reviewers a concrete technical contribution: a domain-specific vision system that converts infusion video into large volumes of structured supervision for future detector training.

#section-heading[References]

[1] E. W. Washburn, "The Dynamics of Capillary Flow," *Physical Review*, Vol. 17, No. 3, 1921, pp. 273-283.

[2] N. Otsu, "A Threshold Selection Method from Gray-Level Histograms," *IEEE Transactions on Systems, Man, and Cybernetics*, Vol. 9, No. 1, 1979, pp. 62-66.

[3] X. Li, et al., "AI-Based Monitoring of Resin Flow Front Using YOLO," *Materials Research Forum*, 2023.

[4] Ultralytics, "YOLOv8 Documentation," 2023.
