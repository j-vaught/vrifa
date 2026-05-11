#colbreak(weak: true)

= Introduction

#figure(
  image("/typst/figures/teaser.pdf", width: 100%),
  caption: [
    Per-frame boundary comparison on the canonical reference video at
    three fill positions ($25 %$, $50 %$, $95 %$). Region IoU and
    boundary $F_1$ for the same comparison are reported in~@tab:headline_vs_baselines.
  ],
) <fig:teaser>

V#smallcaps([artm]), or Vacuum Assisted Resin Transfer Molding, is a widely
used process for one-off composite laminates in research, marine, wind, and
large-format defense applications @Mathur2001VARTMLargeStructures @Bhatt2020VARTMTaguchi, and is the principal route for parts whose
size or geometry rules out autoclave processing @Verma2014VERITyAutoclaveAlternative @Ali2019VARTMOutOfAutoclave @Konstantopoulos2014InlineSensing.
However, the process is susceptible to defects that can render parts unusable.
Dry spots from incomplete impregnation @Cui2023DrySpotMonitoring, race-tracking along edges and the distribution
medium @Vollmer2021EdgeRaceTracking, voids and porosity from trapped air @Hamidi2005VoidMorphology, and resin starvation in interior regions
of the laminate @Szarski2023ResinStarvation can all reduce stiffness and strength of the cured part @Mehdikhani2019VoidsReview, which is
often undesirable in production. Identifying, predicting, and preventing these defects
spans post-cure non-destructive evaluation, in-situ embedded sensing during infusion,
forward Darcy-flow simulation of the run, and digital-twin systems that combine these
signals. Each modality contributes a different slice of evidence, but most are
ultimately reasoning about which fabric wetted, where, and when. The resin flow front
therefore consolidates much of this information into a single quantity that can be
measured directly during the run. Real-time, precise information about the front
position gives the operator the most direct lever on part quality during an active
infusion @NielsenPitchumani2002ClosedLoop @SozerBickertonAdvani2000OnLine. Given that signal, the operator can re-sequence inlets to redirect flow
toward a starved region @LawrenceAdvani2001Sensors @MinaieChen2005Adaptive, open additional vents to break a race-track @HsiaoAdvani2004FlowSensingI @Devillard2005FlowSensingII @LawrenceFriedAdvani2005AuxiliaryGates, or stop the
infusion before resin gels @HickeyBickerton2013CureKinetics.

Deriving or measuring the position, shape, and rate of the flow front is non-trivial, since the front is a two-dimensional moving boundary inside an opaque lay-up, and any sensing modality must trade spatial resolution against intrusiveness in the part. Distributed dielectric arrays @Tifkitsis2014Dielectric and area-sensor grids @Matsuzaki2011AreaSensor produce excellent local wetness measurements at every electrode or cell, but both options require sensors embedded in the lay-up and so alter the local permeability of the part they measure. Fiber-optic frequency-domain reflectometry @Matsuzaki2022FiberOptic time-stamps resin arrival at every grating along an embedded fiber, resulting in excellent temporal precision, but the interrogator instrumentation is expensive enough to be impractical for one-off research panels, and front shapes between gratings must be interpolated. Vision-based methods, like a camera looking down through a transparent vacuum bag, provide dense pixel-level information about the flow front; however, the camera only sees the top ply visible through the bag, so a dry spot one ply deep can go undetected. Additionally, the visible-light contrast between wetted and dry fabric depends on the specific resin-fabric pair, the bag transparency, and the lighting at the bench, making contrast non-stationary across runs @Caglar2018Permeability. Thermal-imaging variants @Konstantopoulos2014InlineSensing partially recover through-thickness saturation by reading the exotherm of curing resin, but the entry cost of a research-grade infrared camera is one to two orders of magnitude above the visible-light option, and the calibration is sensitive to ambient temperature drift.

Vision-based systems are the focus of this paper because a camera does not affect the flow of the part it observes, applies to any geometry without re-instrumentation, and in most labs is already being used as a procedural artifact at near-zero marginal cost. In vision-based systems, like thermal or RGB, the primary goal remains to retrieve either the current edge of the flow front or the entire flow front mask. The dominant approach to derive the front from the live video feed involves computing the change in pixel intensity as the resin wets the fabric, thresholding the response into a binary candidate mask, and cleaning the mask with morphology @LekanidisVosniakos2020IJMMS @AlmazanLazaro2018Materials @AlmazanLazaro2022JMP. However, despite that deceptively simple description, the compute pipeline for this change in pixel intensity is very sensitive to disturbances and prior camera-CV pipelines disagree on the details that control how robust the model is to disturbances. Some systems clip the difference so that only darkening counts as wetting and others do not @AlmazanLazaro2022JMP. Some track a per-pixel running maximum as the reference and others pin a single early frame @LekanidisVosniakos2020IJMMS. Some impose a temporal-locking rule that holds a positive detection through transient dropouts and others operate frame-by-frame @Esperto2025AIBased.

Recent vision-based work has begun to replace these classical operators with learned detectors @Esperto2025AIBased @Stieber2020FlowFrontNet @Stieber2023SimToReal @Park2025FlowFrontGAN @Feng2026FlowCastNet and a parallel thread applies deep models to adjacent VARTM observables such as cure temperature distribution @Zhang2024VARIMTemperature. However, ML in this domain stalls on data scarcity since each VARTM run is its own regime of fabric, bag, lighting, and geometry @Malashin2025MLCompositeReview. Complicating matters further, no shared labeled video benchmark spans multiple molds, so a detector trained on one bench cannot be measured on the next @Stieber2023SimToReal. The classical-CV pipelines deploy without per-bench training data, but they too have not been evaluated against any shared video benchmark, leaving open how they compare to each other or to any future method.


This paper closes these gaps with an integrated classical-computer-vision pipeline evaluated on a 55-frame hand-labeled benchmark spanning eleven distinct VARTM infusion runs. Its four contributions are the following.

+ An integrated VARTM flow-front segmentation pipeline that combines peak-brightness reference, darken-only difference, ROI restriction, dynamic-lag reference selection, Otsu-plus-offset thresholding, morphological cleanup, persistence-based temporal locking, and run-time camera-shift registration in one configuration, with mean Intersection-over-Union (IoU) $0.921$ ($95 %$ bootstrap CI $[0.888, 0.942]$) and mean boundary $F_1$ $0.432$ (CI $[0.401, 0.464]$) on a 55-frame hand-labeled subset spanning eleven distinct infusion runs. The pipeline runs at $10.8$ frames per second on a single CPU and at $109.7$ frames per second on a CUDA implementation for a $1920 times 1080$ input.
+ A regime-indexed configuration lookup, derived from the per-sample ablation below, that recommends preprocessing settings for each pipeline component as a function of run circumstances. The lookup is the practitioner-facing translation of the per-sample evidence and is what an operator on a different bench would consult before tuning their own infusion.
+ A per-sample component-removal ablation on the same 55-frame subset that quantifies the marginal IoU effect of each named primitive, both as an eleven-sample mean and as a per-sample breakdown. The breakdown is reported in full so the joint composition is justified by the per-sample evidence rather than by a single mean. The ablation is the empirical content that the pipeline and the lookup both rest on.
+ A 55-frame hand-labeled benchmark spanning eleven distinct VARTM infusion runs, with per-sample region-of-interest masks and a documented labeling protocol. The two prior published classical-CV pipelines for VARTM flow-front segmentation, Lekanidis and Vosniakos 2020 @LekanidisVosniakos2020IJMMS and Almazán-Lázaro 2022 @AlmazanLazaro2022JMP, are reimplemented from their source papers and evaluated on this benchmark as reference points.

