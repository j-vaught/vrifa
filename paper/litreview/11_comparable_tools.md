# Comparable open-source tools

## VARTM/RTM-specific tools (or absence thereof)

A targeted search across GitHub, Zenodo, Hugging Face, Roboflow Universe, and Papers with Code returned **no open-source tool whose primary purpose is to segment a resin flow front in VARTM/RTM video and emit detector-ready supervision**. The queries used were: `site:github.com "VARTM" "flow front"`, `"resin transfer molding" segmentation github`, `"resin infusion" image processing python github`, `Zenodo VARTM dataset`, `huggingface "VARTM" OR "resin infusion" OR "flow front"`, `Roboflow Universe "resin" OR "VARTM" OR "infusion"`, and `Papers with Code resin flow front composite`. The closest hits are *simulation* codebases, not *image-analysis* codebases. **LCMsim** [1] and **LCMsim_v2.jl** [2] are Julia mold-filling solvers (GPL-2.0) that predict flow-front propagation from mesh, permeability, and pressure inputs and emit JLD2/contour fields; they consume no video and produce no annotations. **FlowFrontNet** [3] is a published CNN for sensor-grid-to-flow-front image regression, with no canonical public CLI repo for video ingest. Computer-vision-controlled infusion has been demonstrated in the literature (e.g., Mesogitis-style PID-vision rigs reported in Materials, MDPI, 2018) [4], but the supporting code was not released. Within the composite-manufacturing user base, the ecosystem is therefore dominated by closed lab scripts and commercial tools (PAM-RTM, RTM-Worx, LIMS), and VRIFA appears to be the first openly licensed CLI that turns infusion video into masks plus standard ML annotations.

## Closest analogues from adjacent scientific imaging

The closest architectural relatives are interactive bioimage and goniometry tools. **TrackMate** [5] (GPL-3.0, last release 2024) is a Fiji plugin that decouples segmentation from temporal linking and supports threshold, MorphoLibJ, ilastik, and StarDist detectors; it exports XML/CSV tracks but no detector-format labels. **CellProfiler** [6] (BSD-3, v4.2.8 2024) is a modular pixel-pipeline GUI with thresholding, morphology, and object-tracking modules and CSV/HDF5 measurement export; it is not a CLI annotation generator. **ilastik** [7] (random-forest pixel classification, BSD) is the canonical "draw on the image and threshold the probability map" tool; the user supplies labels and ilastik supplies a model, the inverse of VRIFA's reference-frame-against-itself approach. In hydrology and coastal science, **CoastSat** [8] (GPL-3.0, v3.2 2025) extracts moving water/sand boundaries from satellite imagery using sub-pixel thresholding inside a user-defined ROI and exports GeoJSON time-series; it is the strongest *moving-region-from-imagery* analogue but operates on georeferenced satellite tiles, not lab video. **Sessile.drop.analysis** [9] (GPL-3.0, 2024) and **DropPy** [10] track a contact line in droplet video with a baseline reference, mirroring VRIFA's "reference frame" concept at single-droplet scale. **PIVlab** [11] (Apache-2.0) and **OpenPIV** [12] (GPL) compute velocity fields in fluid video but never output a binary front mask. **CVAT** [13] and **Label Studio** [14] are *human* annotation engines that export COCO/YOLO; they do not segment flow fronts but they consume what VRIFA emits.

## Roboflow / HuggingFace / Papers with Code presence

A direct site search of Roboflow Universe, Hugging Face Datasets, and Papers with Code returned **no public dataset or model card for "VARTM," "resin infusion," or "flow front" in a composites context** as of May 2026. Generic Roboflow projects exist for "resin" in dental imaging and 3D-print SLA, none for liquid composite molding. The data-centric ML community has therefore not yet been seeded with this problem, which makes VRIFA's COCO/YOLO/Darknet exports a likely upstream supplier rather than a downstream consumer.

## Capability comparison

| Capability | VRIFA | TrackMate | CoastSat | LCMsim |
|---|---|---|---|---|
| ROI handling | CLI rectangular crop | GUI box/polygon | Lon/lat polygon | Mesh region |
| Reference strategy | Per-pixel peak brightness, darken-only | None (per-frame detector) | Per-image NIR/SWIR index | Pressure boundary |
| Illumination handling | Adaptive Otsu + temporal filter | Detector-dependent | Cloud mask + index | N/A (simulation) |
| Annotation export | **COCO + YOLOv5 + Darknet** | XML/CSV tracks | GeoJSON/PKL | JLD2 fields |
| Reproducibility log | `run_summary.yaml` (full config + timing) | Saved Fiji XML | Pickled config | Julia script |
| License | MIT | GPL-3.0 | GPL-3.0 | GPL-2.0 |
| Language / interface | Python CLI | Java/Fiji GUI | Python notebook | Julia |

## Positioning of VRIFA

VRIFA sits in an empty cell of the open-tool matrix: a *headless, video-first, CLI* segmentation pipeline targeted at a *composite-manufacturing* moving region, exporting *detector-ready supervision* in three concurrent formats with a fully serialized run config. Bioimage tools (TrackMate, CellProfiler, ilastik) are GUI-centric and produce measurement tables, not detector labels. CoastSat is the closest spiritual sibling but lives in geospatial coordinates and was never intended for indoor video. The simulation tools (LCMsim) solve the forward physics; they need experimental ground truth that something like VRIFA can supply.

VRIFA's distinct contribution is therefore the *combination*: a peak-brightness, darken-only reference strategy that suits VARTM's lighting; a CLI-driven ROI and Otsu-offset interface; concurrent COCO, YOLOv5, and Darknet emission so the same run feeds Detectron2, Ultralytics, and Darknet without conversion; and a `run_summary.yaml` that makes every annotation traceable. To our knowledge it is the first open infusion-video tool that closes the loop from raw lab footage to ML-trainable supervision without manual labeling.

## References / repos

1. **LCMsim** — https://github.com/obertscheiderfhwn/LCMsim — Julia, GPL-2.0, last release 2025-03-20. RTM/VARI mold-filling simulator.
2. **LCMsim_v2.jl** — https://github.com/LCMsim/LCMsim_v2.jl — Julia refactor with adaptable solver.
3. Stieber et al., **FlowFrontNet** (Springer, 2021) — https://link.springer.com/chapter/10.1007/978-3-030-67667-4_25 — sensor-grid-to-image CNN for CFRP; no canonical public CLI.
4. Matveev et al., *Materials* 11(12):2469, 2018 — https://www.mdpi.com/1996-1944/11/12/2469 — webcam-PID infusion control; code unreleased.
5. **TrackMate** — https://github.com/trackmate-sc/TrackMate — Java/Fiji, GPL-3.0, v7.13 2024-06-25.
6. **CellProfiler** — https://github.com/CellProfiler/CellProfiler — Python, BSD-3, v4.2.8 2024-09-27.
7. **ilastik** — https://github.com/ilastik/ilastik — Python, BSD, interactive RF pixel classifier.
8. **CoastSat** — https://github.com/kvos/CoastSat — Python, GPL-3.0, v3.2 2025-01-06.
9. **Sessile.drop.analysis** — https://github.com/mvgorcum/Sessile.drop.analysis — Python, GPL-3.0, 2024-04-21.
10. **DropPy** — https://github.com/michaelorella/droppy — Python contact-angle goniometry, MIT.
11. **PIVlab** — https://github.com/Shrediquette/PIVlab — MATLAB, Apache-2.0, actively maintained 2025.
12. **OpenPIV** — https://github.com/OpenPIV/openpiv-python — Python, GPL.
13. **CVAT** — https://github.com/cvat-ai/cvat — human annotation, MIT, exports COCO/YOLO.
14. **Label Studio** — https://github.com/HumanSignal/label-studio — human annotation, Apache-2.0, exports COCO/YOLO.
15. **DeePore** — https://github.com/ArashRabbani/DeePore — Python, deep learning for porous-media characterization (adjacent, not flow-front).
