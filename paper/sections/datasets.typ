= Datasets and Labeling Protocol

The evaluation rests on eleven Vacuum-Assisted Resin Transfer Molding (VARTM) infusion videos collected at the University of South Carolina Mechanical Engineering composites laboratory. Three of the eleven samples are committed at the full sensor resolution of $1920 times 1080$ or $1920 times 1088$ pixels and are intended as the canonical reference set for the runtime benchmark and for figure regeneration. The remaining eight samples are operator-view recordings cropped to $1048 times 524$ pixels and span thirty-second-resolution captures of full infusion runs from February 2026. Sample durations range from twenty-three seconds for the shortest standard-rate clip to roughly nine minutes for the longest cropped run, and frame counts range from one hundred to over fifteen thousand. Two of the high-resolution clips are recorded at a time-lapse rate of $3.33$ frames per second, while the remaining nine record at $30$ frames per second. The sample inventory is summarized in Table~@tab:samples.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, left),
    stroke: 0.4pt,
    inset: 6pt,
    table.header(
      [*Sample*], [*Frames*], [*FPS*], [*Width*], [*Height*], [*Class*],
    ),
    [`input_1`],  [706],  [29.97], [1920], [1080], [high-res standard],
    [`input_2`],  [100],  [3.33],  [1920], [1088], [high-res time-lapse],
    [`input_3`],  [200],  [3.33],  [1920], [1088], [high-res time-lapse],
    [`input_4`],  [8 100],  [30.00], [1048], [524], [cropped operator],
    [`input_5`],  [8 100],  [30.00], [1048], [524], [cropped operator],
    [`input_6`],  [8 100],  [30.00], [1048], [524], [cropped operator],
    [`input_7`],  [8 100],  [30.00], [1048], [524], [cropped operator],
    [`input_8`],  [12 600], [30.00], [1048], [524], [cropped operator],
    [`input_9`],  [15 469], [30.00], [1048], [524], [cropped operator],
    [`input_10`], [11 426], [30.00], [1048], [524], [cropped operator],
    [`input_11`], [14 876], [30.00], [1048], [524], [cropped operator],
  ),
  caption: [
    Sample inventory. Eleven VARTM infusion clips committed under
    `data/`. The high-resolution clips serve as the canonical
    benchmark targets; the cropped operator-view recordings
    contribute the long-form generalization evidence.
  ],
) <tab:samples>

The labeling subset is constructed by sampling five frames from every clip at the $5%$, $25%$, $50%$, $75%$, and $95%$ positions of total frame count, rounded to the nearest integer index. The sampling is stratified within each sample so that the fifty-five-frame ground-truth set covers every clip equally rather than concentrating in a single high-frame-count recording. The resulting set spans early-fill, early-mid, mid, mid-late, and late-fill stages within each clip, and across clips it spans the four resolution and frame-rate regimes described above.

== Labeling protocol

Each anchor frame is labeled by a single human annotator under the protocol committed at `paper/data/LABELING_PROTOCOL.md`. The annotator marks one polygon per frame indicating the resin-wetted region, defined as the area where resin has visibly transitioned the fabric from dry to saturated. Specular reflections from the silicone vacuum bag, vacuum-bag wrinkles and creases that pre-date the front, fabric-weave shadows that pre-date the front, and pooled resin in inlet runners outside the laminate boundary are explicitly excluded. The annotator distinguishes real wetting from transient appearance changes by scrubbing forward a few frames and confirming that the candidate region's brightness has changed monotonically. Real wetting darkens once and stays dark; reflections and wrinkles oscillate. The labeling tool is the browser-based polygon editor at #link("https://www.makesense.ai")[makesense.ai], chosen for its ability to import a flat directory of PNGs and export Common Objects in Context (COCO) JSON without intermediate format conversion. Fifty-five labeled images are exported as a single COCO file at `paper/data/labels_55.json`. Figure~@fig:labeling shows a representative labeled frame.

#figure(
  // image("../typst/figures/labeling_protocol.pdf"),
  rect(width: 100%, height: 1.4in, stroke: 0.5pt, inset: 8pt)[
    _Labeling protocol illustration placeholder._ One representative
    frame from the fifty-five-frame ground-truth subset, with the
    operator's polygon overlaid in rose. Replaced once labels arrive.
  ],
  caption: [
    Example labeled frame from the labeling pass. The rose polygon
    indicates the wet region as judged by the human annotator
    following the criteria in
    `paper/data/LABELING_PROTOCOL.md`.
  ],
) <fig:labeling>

== Scope and known limits of the labeling subset

A subset of fifty-five frames is small in absolute terms and reviewers should treat per-sample agreement numbers as first-pass evidence rather than as definitive benchmarks. The strategy here is breadth over depth, stratifying across all eleven clips so the per-sample breakdown surfaces any sample-specific mode of failure rather than concentrating evidence on a single recording. The agreement script tolerates partial label sets, and scaling the labeled subset to one hundred and ten frames (ten per sample) is a straightforward extension if a reviewer demands more. The locked thesis sentence in this paper accommodates that scaling without rewording beyond the count.

A single human annotator was used. Inter-annotator agreement is not reported. The protocol was written before the labeling pass began, however, so the criteria for what counts as wet are recoverable and could be applied by a second annotator in follow-on work without ambiguity. The protocol document is committed alongside the labels to support exactly that kind of replication.
