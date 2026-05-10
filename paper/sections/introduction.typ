#import "@preview/droplet:0.3.1": dropcap

= Introduction

#figure(
  // image("/typst/figures/teaser.pdf", width: 95%),
  rect(width: 100%, height: 2.0in, stroke: 0.5pt, inset: 8pt)[
    _Teaser placeholder._ Three columns showing one canonical input
    frame at fill position 50%. Left: raw BGR input. Center:
    integrated-configuration overlay with the locked-mask boundary
    drawn in garnet. Right: ground-truth human polygon overlaid in
    rose for comparison. A small chip in the upper-right corner
    shows mean IoU and 95% bootstrap confidence interval over the
    eleven-sample subset. Replaced once the agreement run produces
    the final overlay frames.
  ],
  caption: [
    The integrated pipeline produces per-pixel wet-versus-dry masks
    against a multi-mold labeled benchmark.
  ],
) <fig:teaser>

#figure(
  // image("/typst/figures/problem_motivation.pdf", width: 95%),
  rect(width: 100%, height: 1.6in, stroke: 0.5pt, inset: 8pt)[
    _Problem-motivation placeholder._ Two-column comparison.
    Left: schematic of an embedded SMARTweave grid bonded into the
    lay-up, showing the per-cell wetness samples a dielectric grid
    can recover at its own pitch and the perturbation it introduces
    into the laminate. Right: single webcam looking down through
    the transparent vacuum bag, with the per-pixel front geometry
    the camera recovers and zero perturbation of the part. Numbers
    along the bottom note approximate hardware costs (\$10--50 for
    the camera, \$100s--\$50k for the embedded sensor stack).
  ],
  caption: [
    Why a camera. Embedded sensor grids alter the local
    permeability they measure and quantize the front to their own
    pitch; a camera through the transparent bag is non-perturbative
    and pixel-resolved at one to three orders of magnitude lower
    hardware cost.
  ],
) <fig:problem_motivation>

#dropcap(
  height: 2,
  hanging-indent: 0em,
  justify: true,
)[V #smallcaps([acuum])-Assisted Resin Transfer Molding (VARTM) is the dominant process for one-off composite laminates from research panels through small-series aerospace structural parts, and the principal observable signal available to the operator during a run is the visible advance of the resin flow front under the transparent vacuum bag @Konstantopoulos2014InlineSensing.] The position, shape, and rate of that front carry process-quality information that other, more instrumented modalities recover only indirectly. Distributed dielectric arrays @Tifkitsis2014Dielectric and area-sensor grids @Matsuzaki2011AreaSensor produce excellent local wetness measurements but require sensors embedded in the lay-up, and fiber-optic frequency-domain reflectometry @Matsuzaki2022FiberOptic demands instrumentation no operator on a research bench will install for a one-off panel. Direct video of the bag is the one modality that costs nothing extra, sees the entire laminate at once, and is already recorded in most labs as a procedural artifact.

Prior camera-based VARTM and Liquid Composite Molding (LCM) systems agree on the basic recipe. Difference the live frame against a reference image of the dry preform, threshold the response into a binary candidate mask, and clean the mask with morphology @LekanidisVosniakos2020IJMMS @AlmazanLazaro2018Materials @AlmazanLazaro2022JMP. They disagree on every detail that controls how well the recipe works. Some systems clip the difference so that only darkening counts as wetting and others do not @AlmazanLazaro2022JMP. Some track a per-pixel running maximum as the reference and others pin a single early frame @LekanidisVosniakos2020IJMMS. Some impose a temporal-locking rule that holds a positive detection through transient dropouts and others operate frame-by-frame @Esposito2025AIFusion. None of these systems reports the joint contribution of the components it uses, the marginal cost of each component it discards, or the agreement of the resulting mask against a labeled video benchmark, because no shared labeled benchmark exists. The practitioner is left to take the recipe at face value.

The work is hard for three concrete reasons that have to be addressed by anything claiming to do it. The vacuum bag produces strong specular reflections from lab lighting that look bright in every channel and shift on every operator move. The bag wrinkles and creases that pre-date the front are static dark features that do not represent wet fabric and must not be mistaken for it. The lighting itself drifts across a multi-minute infusion as lamps warm up and the operator moves equipment. A pipeline that fails to address any one of these will mark unwetted fabric as wet on most frames of most infusions. Prior camera-based systems individually defeat one or two of the three. None reports which component defeats which failure mode, and none reports the price paid for leaving any of them out.

This paper presents an integrated classical-computer-vision pipeline that combines, in one configuration, the visual primitives the prior systems use in isolation. The pipeline restricts the comparison to a region of interest, projects the working frame into the lightness channel of CIELAB, accumulates a per-pixel peak-brightness reference that absorbs lighting drift, computes a darken-only difference that rejects specular highlights, dynamically lags the reference frame against fill rate, thresholds the response with Otsu, cleans the mask with morphological closing and opening followed by connected-components filtering, and locks pixels whose wet label has persisted for several consecutive frames. On a 55-frame hand-labeled subset spanning eleven distinct VARTM infusion runs, the integrated configuration reaches mask Intersection-over-Union (IoU) of $X.X X X$ with a 95 % bootstrap confidence interval of $[a, b]$ and boundary $F_1$ of $Y.Y Y Y$ with confidence interval $[c, d]$. A per-sample component-removal ablation on the same subset reports the marginal IoU effect of disabling each named primitive in turn. The reported effects are not uniform in sign or magnitude across the eleven samples, which is the empirical content of the ablation rather than a defect, since a primitive that helps a long operator-view recording can be neutral or counterproductive on a high-frame-rate clip whose dynamics it was not tuned for. The pipeline runs at $30$ frames per second on a single central-processing-unit (CPU) thread, and a CUDA implementation reaches $K$ frames per second on the same eleven samples.

The contributions of this paper are the following.

+ An integrated VARTM flow-front segmentation pipeline that combines peak-brightness reference, darken-only difference, ROI restriction, dynamic-lag reference selection, Otsu-plus-offset thresholding, morphological cleanup, persistence-based temporal locking, and run-time camera-shift registration in one configuration, with mean IoU $X.X X X$ and boundary $F_1$ $Y.Y Y Y$ reported with 95 % bootstrap confidence intervals on a 55-frame hand-labeled subset spanning eleven distinct infusion runs.
+ A regime-indexed configuration lookup, derived from the per-sample ablation below, that recommends preprocessing settings for each pipeline component as a function of run circumstances (illumination drift, fabric type, fill rate, frame rate, camera stability). The lookup is the practitioner-facing translation of the per-sample evidence and is what an operator on a different bench would consult before tuning their own infusion.
+ A per-sample component-removal ablation on the same 55-frame subset that quantifies the marginal IoU effect of each named primitive, both as an eleven-sample mean and as a per-sample breakdown. The breakdown is reported in full because the effect of any one primitive is regime-dependent, so the joint composition is justified by the per-sample evidence rather than by a single mean. The ablation is the empirical content that the pipeline and the lookup both rest on.

Section~2 places this work among the three families of prior work that touch the problem and contrasts the integrated pipeline against the two prior camera-CV systems whose method sections describe enough operations to reproduce. Section~3 walks each component of the pipeline and points forward at the ablation evidence. Section~4 documents the labeling protocol and the eleven-sample dataset. Section~5 reports the integrated configuration's agreement against the labeled subset, the per-sample component-removal ablation, and the configuration lookup the ablation supplies. Section~6 reports CPU and CUDA throughput. Section~7 discusses failure modes and design tradeoffs. Section~8 concludes.
