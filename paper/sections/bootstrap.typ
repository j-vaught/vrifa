= Detector Bootstrap Demonstration

VRIFA is positioned in part as a label-generation engine for downstream learned detectors. To exhibit that pathway concretely, this section reports a single qualitative demonstration in which a YOLO segmentation model @Jocher2023YOLOv8 is trained from the COCO export produced by VRIFA's annotation pipeline and then applied to a held-out frame from one of the eleven samples. The training set consisted of the COCO export from the canonical reference videos `input_1` and `input_2`, comprising $4,!157$ and $205$ wet-region annotations respectively. Training was a single short run with default Ultralytics hyperparameters, no manual tuning, and no separate validation set tied to the human-labeled subset described in Section~5. The result is shown in Figure~@fig:yolo_demo.

#figure(
  // TODO image("../assets/yolo_overlay_demo.png")
  rect(width: 100%, height: 1.4in, stroke: 0.5pt, inset: 8pt)[
    _YOLO bootstrap placeholder._ A held-out frame with the
    YOLO-predicted wet-region overlay. Replaced once the bootstrap
    artifact is regenerated against the current Rust default
    configuration.
  ],
  caption: [
    Qualitative result of bootstrapping a YOLO segmentation model
    from VRIFA's COCO export. The demonstration is intentionally
    qualitative; quantitative downstream-detector evaluation is
    deferred to follow-on work with a separately-collected
    detector-validation set.
  ],
) <fig:yolo_demo>

The result should be read as a proof-of-concept for the bootstrap pathway, not as a benchmark of the trained detector. There is no quantitative claim about the YOLO model in this paper; in particular, no IoU or precision-recall metric is reported for the detector itself. Two reasons. First, the annotation export from VRIFA is uncalibrated supervision and the trained detector inherits whatever systematic error the upstream classical pipeline produces; reporting the detector's accuracy against the same fifty-five labels would conflate the labeling-pipeline error with the detector-architecture error. Second, a quantitative claim about the trained detector requires a held-out detector-validation set that does not overlap with the training videos, and that set has not been collected. Both gaps are followed up in dedicated work and are out of scope for this submission.
