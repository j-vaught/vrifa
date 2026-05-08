# VRIFA Literature Review

Twelve focused reviews positioning VRIFA against the bodies of literature it touches, plus a synthesis that maps directly into the SciTech draft variants in `../`.

## Read order

Start with the synthesis. It cites the section files for detail.

- [`00_synthesis.md`](00_synthesis.md) — Cross-section synthesis, drop-in *Related Work* outline, single-sentence and single-paragraph positioning.

## Sections

| # | File | Topic | Refs |
|---|---|---|---|
| 01 | [`01_sensors.md`](01_sensors.md) | Non-vision sensor-based VARTM/LCM monitoring (dielectric, fiber-optic, ultrasonic, pressure, capacitive, ECT/EIT) | 22 |
| 02 | [`02_vision_flowfront.md`](02_vision_flowfront.md) | Vision-based flow-front detection (the most direct prior art) | 20 |
| 03 | [`03_flow_physics.md`](03_flow_physics.md) | Resin flow physics: Darcy, dual-scale, VARTM-specific phenomena, race-tracking | 24 |
| 04 | [`04_classical_cv.md`](04_classical_cv.md) | Classical CV building blocks: Otsu, background subtraction, codebook, morphology | 31 |
| 05 | [`05_annotation_engines.md`](05_annotation_engines.md) | Annotation engines and weak supervision: Snorkel, Cellpose, manufacturing bootstrapping | 35 |
| 06 | [`06_deep_detectors.md`](06_deep_detectors.md) | YOLO family, U-Net, Mask R-CNN, SAM/SAM2, deep detectors in composite manufacturing | 33 |
| 07 | [`07_defects_and_anomalies.md`](07_defects_and_anomalies.md) | Defects and anomalies: dry spots, race-tracking, voids, closed-loop control | 26 |
| 08 | [`08_process_monitoring.md`](08_process_monitoring.md) | Process monitoring, digital twins, Bayesian permeability inversion | 30 |
| 09 | [`09_anomaly_detection.md`](09_anomaly_detection.md) | Anomaly detection in manufacturing video and time-series (PatchCore, TranAD, MVTec AD) | 48 |
| 10 | [`10_reproducibility.md`](10_reproducibility.md) | Reproducibility, scikit-image / CellProfiler analogues, reporting standards | 21 |
| 11 | [`11_comparable_tools.md`](11_comparable_tools.md) | Comparable open-source tools (CoastSat, TrackMate, FlowFrontNet repo) | ~12 |
| 12 | [`12_optimization_and_metrics.md`](12_optimization_and_metrics.md) | Tuning, ablation methodology, multi-metric segmentation evaluation | 24 |

Total references across all sections: ~325 (with overlap on the highest-cited works).

## Items to verify before submission

Surfaced in section 02 and 06.

- *X. Li et al., "AI-Based Monitoring of Resin Flow Front Using YOLO," Materials Research Forum, 2023* could not be independently located. Replace, locate the original, or drop.
- The 2025 dielectric+YOLO LCM paper used as a closest analog was reachable only through ResearchGate; verify author list and venue.

## How this maps to the three draft variants

- **Variant A (Ablation)** — Pull from sections 04, 12, 03, 07. The classical-CV-block lineage and the multi-metric reporting argument are the strongest spine.
- **Variant B (Dataset)** — Pull from sections 05, 06, 02. Annotation-engine lineage is the spine; FlowFrontNet and the YOLO LCM paper are the closest learned-baseline anchors.
- **Variant C (Monitoring)** — Pull from sections 01, 07, 08, 09. Sensor-vs-vision contrast and the inverse-identification consumer story are the spine.

All three variants share the section-03 physics foundation and the section-10 reproducibility argument.
