# VRIFA — Literature Synthesis and Positioning

This document synthesizes the twelve focused literature reviews in `01_*.md` through `12_*.md` into a single positioning argument. It is structured to be droppable into the *Related Work*, *Background*, and *Discussion* sections of the SciTech extended abstract (any of variants A, B, C). Each subsection ends with one or two sentences that the paper can reuse verbatim.

## 1. The neighborhood of literatures VRIFA touches

VRIFA sits at the intersection of six bodies of work, each surveyed in its own file.

| Section | Topic | Anchoring claim |
|---|---|---|
| 01 | Sensor-based VARTM/LCM monitoring | Distributed sensors win on through-thickness data and timing certainty; they lose on full-field 2D geometry, instrumentation cost, and retrofittability. |
| 02 | Vision-based flow-front detection | Direct prior art is sparse, all closed-source, and reports no mask IoU or boundary-F1 on real infusion video. |
| 03 | Resin flow physics | Every quantitative LCM model (Darcy, Lucas-Washburn, Pillai-Advani dual-scale, Hsiao-Advani VARTM) takes time-resolved front geometry as its primary observable. |
| 04 | Classical CV building blocks | Each block VRIFA uses is conventional; the novelty is the *directional* peak-brightness rule tied to the monotonic optical signature of resin wetting. |
| 05 | Annotation engines and weak supervision | VRIFA is a domain-specific instance of the Snorkel + Cellpose + Scime-Beuth bootstrapping lineage, distinguished by tunable physical priors and triple-schema export. |
| 06 | Deep detectors / YOLO in manufacturing | The detector literature assumes labels exist; VRIFA targets the unsolved upstream bottleneck. |
| 07 | Defects and process anomalies | Front geometry is a near-deterministic precursor to dry spots, voids, and race-tracking; the spatial observable matters. |
| 08 | Process monitoring and digital twins | Bayesian/Kalman/PF permeability inversion already takes hand-segmented fronts as input; VRIFA is the missing automation layer. |
| 09 | Anomaly detection in manufacturing | VRIFA is upstream feeder for both image-anomaly (PatchCore family) and time-series-anomaly (TranAD/CUSUM) pipelines. |
| 10 | Reproducibility and open tooling | VRIFA's `run_summary.yaml` and CLI design instantiate the Sandve / Wilson recommendations explicitly; composites have very little open data and few open tools. |
| 11 | Comparable open-source tools | No public VARTM-image-analysis tool exists; closest architectural sibling is CoastSat (sub-pixel thresholding of a moving boundary inside a user-defined ROI). |
| 12 | Tuning, ablation, and metrics | VRIFA's multi-metric posture (IoU + Dice + boundary F1 + box IoU + mean boundary distance) matches the Maier-Hein "Metrics Reloaded" recommendations. |

## 2. The positioning argument in five paragraphs

### 2.1 Why front geometry is the right observable

Every quantitative model of LCM filling --- Darcy, Lucas-Washburn, Pillai-Advani dual-scale, Hsiao-Advani VARTM --- takes as its primary observable the time-resolved geometry of the wetted region. The front *position* fixes the $\sqrt{t}$ scaling and therefore the lumped quantity $K\Delta p/(\mu \phi)$; the front *shape* fixes the ratio of principal permeabilities and the presence of race-tracking; *departures* from a smooth Darcy prediction localize dry-spot risk (Tucker & Dessenberger 1994; Adams & Rebenfeld 1991; Pillai & Advani 1998; Hsiao et al. 2000). The defect literature reinforces this from the other direction: stalled-front geometry is a near-deterministic precursor to dry-spot entrapment (Gokce et al. 2005; Hsiao & Advani 2004), and front velocity sets the capillary-number regime governing void content (Patel & Lee 1996; Park & Lee 2011; Lebrun et al. 2008). VRIFA's reported metric set --- mask IoU validates the *area* observable, boundary F1 validates the *shape* observable, and mean boundary distance validates the *position* observable --- maps one-to-one to these physically meaningful quantities.

### 2.2 Where prior art falls short of producing this observable

Sensor-based monitoring (dielectric, electrical resistance, fiber-optic FBG/OFDR, ultrasonic, pressure, thermocouple, SMART/PZT, capacitive area-sensor films, ECT/EIT) is mature and well surveyed in Konstantopoulos & Hueber et al. 2019 and Sevenois & Koissin. These modalities deliver excellent timing and through-thickness coverage but require embedded instrumentation and do not natively produce 2D front geometry, with the partial exception of Matsuzaki et al.'s 2011 area-sensor array. Vision-based prior art on real infusion video is sparse: Pineda et al. 2010 (AVP), Almazán-Lázaro et al. 2018/2022, Mejía-Ugalde et al. 2020, and the 2025 dielectric+YOLO LCM monitor each apply image differencing or a learned detector to wetting video, but none publish (i) a mask IoU number on real infusion video, (ii) a boundary-F1 number, (iii) the underlying video, or (iv) the source code. Stieber et al.'s FlowFrontNet (2020/2021) is the strongest learned baseline in the LCM space, but it maps a sparse pressure-sensor grid to a dense flow-front *image* via simulation; the I/O contract is complementary to VRIFA's, not competitive. VRIFA appears to be the first open, MIT-licensed, end-to-end pipeline that takes ordinary infusion video and emits pixel-accurate region geometry with reported IoU, Dice, boundary-F1, and mean boundary distance on a human-labeled subset.

### 2.3 What VRIFA contributes algorithmically

The pipeline composes only conventional blocks --- ROI cropping, colorspace transform, Gaussian blur, Otsu thresholding (Otsu 1979), morphology (Serra 1982; Suzuki & Abe 1985), connected-component area filtering, and lock-frame temporal persistence (analogous to Canny 1986 hysteresis). The genuinely novel element is the *directional* per-pixel peak-brightness rule, $\text{score}_p = \text{peak}_p - \text{current}_p$, applied with a darken-only cutoff. The closest precedents in the change-detection literature are the codebook model of Kim et al. 2005 and the W4 system of Haritaoglu et al. 2000, both of which track *symmetric* min/max envelopes around each pixel. VRIFA discards the lower envelope and enforces a sign-asymmetric distance, justified physically by the optical signature of resin wetting being a monotonic darkening transition. A symmetric envelope would actively contaminate the score with specular brightening and auto-exposure flicker, and the 91-trial ablation confirms that retaining peak-reference + darken-only is what the optimizer converges to. The contribution is therefore not the envelope idea but its *directional half* tied to a domain-specific monotonic transition.

### 2.4 What VRIFA contributes as data infrastructure

The annotation-engine literature traces a clean lineage from Snorkel-style data programming (Ratner et al. 2017) through pseudo-labeling (Lee 2013; FixMatch 2020; Noisy Student 2020) to scientific-imaging bootstrapping (CellProfiler $\rightarrow$ Cellpose, Stringer et al. 2021) and process-imaging bootstrapping (Scime & Beuth 2018/2019; Gobert et al. 2018). VRIFA fits squarely in this lineage as a single, tunable labeling function with a physical prior, three-format export (COCO, YOLOv5-seg, Darknet), and a quantified ablation showing that label quality is improvable from objective 0.583 to 0.807. The downstream detector literature --- the YOLO family, U-Net, Mask R-CNN, DeepLab v3+, SAM/SAM2 --- assumes labels exist; VRIFA targets the unsolved upstream bottleneck. This positions the dataset-first paper variant (B) as the strongest near-term submission: it claims a label-generation contribution that is already supported by 4,689 exported regions across three runs, and it leaves the detector benchmark for follow-up work, in line with the label-noise robustness literature (Frenay & Verleysen; Rolnick et al.; Northcutt et al.) which predicts that a saturating mAP-vs-label-quality curve is the natural follow-up experiment.

### 2.5 What VRIFA contributes as monitoring and process infrastructure

Inverse identification of permeability from flow-front data (Causse et al. 2021; Matveev et al. 2021; Caglar et al. 2021) and digital-twin work for VARTM (Stieber et al. 2021; Werner et al.) all assume the existence of a flow-front observation; obtaining it is non-trivial and typically requires sensor arrays. VRIFA is the missing observation layer for these methods. Closed-loop control work (Devillard et al. 2003; Modi et al.; Lawrence et al.) explicitly takes "actual flow-front location" as the controller input, the slot VRIFA fills with a webcam in place of a dielectric or pressure array. The reproducibility-first design --- single-file CLI, every flag documented, `run_summary.yaml` emitted per run, MIT license, sample data shipped --- instantiates Sandve et al.'s "Ten Simple Rules for Reproducible Computational Research" and Wilson et al.'s "Best Practices for Scientific Computing" recommendations explicitly, and lowers the barrier for downstream Bayesian-inversion and digital-twin work that currently relies on hand-segmented fronts.

## 3. Honest limitations to retain

These are surfaced repeatedly across sections and should appear in the paper's *Limitations* discussion rather than be omitted.

- VRIFA needs a transparent vacuum bag and reasonably stable illumination; closed metallic molds keep the structural advantage of embedded sensors (sections 01, 02).
- VRIFA produces no through-thickness information; UT, FBG, and PZT remain superior for that observable (section 01).
- The reported boundary F1 of 0.559 lies below sub-millimeter E-TDR localization on the wire path (section 01) and well below the 0.992 IoU that learned segmenters achieve on simpler microfluidic wetting fronts (AI-CMCA 2025, section 02). A learned successor trained on VRIFA's own annotation outputs is the natural next step.
- VRIFA does not yet quantify how detector mAP responds to label tuning; the 0.583 $\rightarrow$ 0.807 ablation operates on label quality, not on downstream detector performance. The label-noise literature (section 05) flags this as the priority follow-up.
- No public VARTM in-process video dataset exists; the 1,006-frame VRIFA package is internally consistent but has not been compared head-to-head against an external benchmark, because none exists (sections 02, 10, 11).

## 4. Suggested *Related Work* outline (drop-in)

For the SciTech extended abstract, a single-page Related Work block can be assembled by stitching one paragraph from each of these sections, in this order, with one or two of the strongest cites each.

1. **LCM monitoring landscape** --- one paragraph anchored on Konstantopoulos & Hueber et al. 2019 (review) plus one sensor exemplar (Skordos & Partridge 2000 dielectric, Dominauskas et al. 2003 E-TDR, or Matsuzaki et al. 2011 area sensor).
2. **Vision-based flow-front prior art** --- one paragraph anchored on Pineda et al. 2010 (AVP), Almazán-Lázaro et al. 2018/2022, FlowFrontNet (Stieber et al. 2021), and the 2025 dielectric+YOLO LCM monitor.
3. **Annotation engines** --- one paragraph anchored on Ratner et al. 2017 (Snorkel), Stringer et al. 2021 (Cellpose), and Scime & Beuth 2018/2019 as the closest manufacturing analog.
4. **Inverse identification and digital twins as the natural consumer of VRIFA outputs** --- one paragraph anchored on Causse et al. 2021 and Matveev et al. 2021.
5. **Metrics and tuning posture** --- one or two sentences anchored on Maier-Hein et al. 2024 ("Metrics Reloaded") and Akiba et al. 2019 (Optuna), justifying VRIFA's multi-metric reporting and the 91-trial ablation methodology.

## 5. The single-sentence elevator positioning

> VRIFA is the first open, MIT-licensed CLI tool that turns ordinary VARTM infusion video into time-resolved 2D flow-front geometry and detector-ready supervision in COCO, YOLOv5, and Darknet formats simultaneously, using a directional peak-brightness rule that matches the monotonic optical signature of resin wetting, with a 91-trial ablation that improves a multi-metric agreement objective from 0.583 to 0.807 against a human-labeled subset.

## 6. The single-paragraph elevator positioning

> Liquid composite molding has a mature literature on sensor-based front detection (dielectric, fiber-optic, ultrasonic, pressure, capacitive) and a thinner literature on direct camera-based monitoring. The latter is dominated by closed-source classical-CV pipelines (Pineda et al. 2010; Almazán-Lázaro et al. 2018) and by sensor-input CNNs that synthesize an image from a sparse grid (Stieber et al. 2021), none of which publish mask IoU or boundary-F1 on real infusion video and none of which release code or video data. Front geometry is the right observable because every Darcy / dual-scale / VARTM-specific model takes it as the primary state, and the defect literature ties stalled-front geometry to dry-spot formation (Gokce et al. 2005; Devillard et al. 2003). VRIFA contributes a directional peak-brightness rule justified by the monotonic optical signature of resin wetting, a 91-trial ablation that quantifies the design space, three standardized annotation export formats that feed the existing detector and digital-twin pipelines (Snorkel-style and FlowFrontNet-style consumers respectively), and a fully reproducible CLI shipped under MIT license with sample data, occupying a niche --- "open, video-input, full-field, detector-ready, reproducible" --- that no prior tool fills.

## 7. Cross-section reference highlights

The single most-cited works across multiple sections:

- **Otsu (1979)** --- thresholding; load-bearing in 04 and 12.
- **Suzuki & Abe (1985)** --- contour extraction (the algorithm OpenCV's `findContours` uses, which VRIFA uses); 04.
- **Maier-Hein et al. (2024) "Metrics Reloaded"** --- multi-metric segmentation reporting; 10 and 12.
- **Sandve et al. (2013), Wilson et al. (2014)** --- reproducibility recommendations; 10.
- **Konstantopoulos & Hueber et al. (2019)** --- LCM monitoring review, the introductory anchor; 01 and 08.
- **Stieber et al. (2021) FlowFrontNet** --- the strongest learned LCM baseline; 02, 06, 08.
- **Pineda et al. (2010), Almazán-Lázaro et al. (2018)** --- direct optical-input prior art; 02.
- **Gokce, Hsiao, Advani (2005)** --- stalled-front $\rightarrow$ dry-spot link; 03 and 07.
- **Devillard et al. (2003)** --- closed-loop control with flow-front position as input; 07 and 08.
- **Ratner et al. (2017) Snorkel; Stringer et al. (2021) Cellpose; Scime & Beuth (2018/2019)** --- annotation-engine lineage; 05 and 06.
- **Akiba et al. (2019) Optuna** --- canonical tuning framework; 12.
- **Kim et al. (2005) codebook; Haritaoglu et al. (2000) W4** --- closest envelope-tracking precedents to VRIFA's peak-brightness rule; 04.

## 8. Items the user must verify before submission

Two citations were flagged by the agents as not independently confirmed in indexed proceedings.

- The currently-cited *X. Li et al., "AI-Based Monitoring of Resin Flow Front Using YOLO," Materials Research Forum, 2023* could not be located by either the section-02 or section-06 agent. The closest verifiable analog is the 2025 *AI-based approach for flow front monitoring and prediction in liquid composite molding processes based on dielectric and visual data elaboration* on ResearchGate. The user should either locate the original Li reference, replace it, or drop it.
- The same 2025 dielectric+YOLO LCM paper was reachable only through ResearchGate (403 on direct fetch), so the author list and venue should be re-verified before final submission.
