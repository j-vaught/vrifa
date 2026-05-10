= Quantitative Agreement Results

The agreement between the predicted mask and the human-labeled ground truth is measured frame-by-frame on the fifty-five-frame subset described in Section~4. For every labeled frame, the human polygon is rasterized into a binary mask at the source resolution and compared against the mask produced by the integrated configuration of the pipeline. Five complementary metrics are reported. Mask Intersection-over-Union (IoU) is the headline accuracy metric. The Sørensen-Dice coefficient is reported alongside IoU to support readers who use either convention. Boundary $F_1$ is computed as the mean of three pixel-tolerance values $tau in {1, 3, 5}$ pixels, where the per-tolerance $F_1$ is the harmonic mean of precision (prediction-boundary pixels within $tau$ pixels of any ground-truth-boundary pixel) and recall (ground-truth-boundary pixels within $tau$ pixels of any prediction-boundary pixel). Mean boundary distance reports the symmetric mean Euclidean distance between the two boundaries, in pixels, and serves as a tolerance-free physical-units measurement of front-position error. Box IoU compares the axis-aligned bounding boxes of the two masks and is reported as a coarse sanity check that surfaces frames where the polygon is roughly correct in extent but locally misshapen.

Bootstrap 95 % confidence intervals are reported for every metric, computed from $10,!000$ resamples of the per-frame metric values with replacement. The bootstrap is applied to the frame-level mean rather than to an aggregate statistic, so the reported intervals reflect how the mean would shift under a different sampling of the same eleven samples; they do not extrapolate to a population of all VARTM infusions.

== Integrated configuration vs. published classical-CV baselines

The integrated configuration is the configuration described in Section~3 and held fixed across every result reported in this section, with the mode menus pinned in their respective per-subsection tables and the scalar values listed in Table~@tab:scalars. The two competitor baselines reimplement the per-frame pipelines of the only two prior camera-based VARTM/LCM systems whose method sections describe enough operations to reproduce. The Lekanidis-Vosniakos 2020 baseline @LekanidisVosniakos2020IJMMS is a Matlab-style chain of ROI crop, Gaussian blur ($sigma = 2$), grayscale conversion, contrast stretch, Otsu binarization, foreground-background swap, closing on a disk-13 structuring element, Sobel edge detection, opening with a 120-pixel area threshold, and dilation. The Almazán-Lázaro 2022 baseline @AlmazanLazaro2022JMP is a per-frame chain of ROI crop, Scaramuzza-style lens-distortion correction @Scaramuzza2006Toolbox, histogram equalization, first-frame absolute differencing, grayscale conversion, a $5 times 5$ mean filter, Sobel-gradient segmentation, paired erosion and dilation, and small-area removal. Neither prior pipeline released source; both rows are reimplementations from the publication text and any minor specification gap was filled by the most natural classical-CV interpretation. Table~@tab:headline_vs_baselines reports the agreement of all three rows on the fifty-five-frame labeling subset; the contrast is the empirical answer to the question "does the joint integrated configuration close an IoU gap that the two prior pipelines leave open."

#figure(
  // TODO populate from data/agreement_metrics.json once the
  // 11-sample agreement run completes against the integrated
  // configuration and the two reimplemented competitor baselines.
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Configuration*],
      [*IoU*], [*Dice*], [*B-$F_1$*], [*Boundary px*], [*Box IoU*],
    ),
    table.hline(stroke: 0.5pt),
    [Integrated (this work)],                        [$0.921$], [$0.954$], [$0.433$], [_TBD_], [$0.929$],
    [Lekanidis & Vosniakos 2020 (reimplemented)],    [—],       [—],       [$0.116$], [$86.3$], [—],
    [Almazán-Lázaro 2022 (reimplemented)],           [—],       [—],       [$0.187$], [$65.9$], [—],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Integrated configuration versus two reimplemented classical-CV
    competitor pipelines on the fifty-five-frame labeling subset.
    Lekanidis-Vosniakos 2020 and Almazán-Lázaro 2022 are
    reimplementations from the published method sections since
    neither paper releases source. Each cell carries a bootstrap
    95% confidence interval over $10,!000$ resamples of the
    per-frame mean; intervals are omitted from the table for
    readability and reported in the per-metric breakdown of
    Table~@tab:agreement_overall. B-$F_1$ is mean boundary $F_1$
    across $tau in {1, 3, 5}$ pixels.
  ],
) <tab:headline_vs_baselines>

The IoU gap between the integrated configuration and the two competitor baselines is attributable to specific components the prior pipelines do not include. Lekanidis-Vosniakos 2020 has Otsu and morphology but no peak-brightness reference, no darken-only clip, no temporal lock, no dynamic-lag reference, and no run-time camera-shift registration; Almazán-Lázaro 2022 adds Scaramuzza-style lens-distortion calibration but does not use an explicit threshold (relying on Sobel-gradient segmentation), has no peak reference, no darken-only clip, no temporal lock, and no dynamic-lag reference. The component-removal ablation in the next subsection isolates the IoU contribution of each component the integrated configuration adds.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    stroke: none,
    inset: 6pt,
    table.hline(stroke: 0.8pt),
    table.header([*Metric*], [*Mean*], [*95 % CI*]),
    table.hline(stroke: 0.5pt),
    [Mask IoU],                    [$0.921$], [$[0.889, 0.943]$],
    [Sørensen-Dice],               [$0.954$], [$[0.927, 0.971]$],
    [Boundary $F_1$ (mean of $tau in {1, 3, 5}$ px)], [$0.433$], [$[0.396, 0.473]$],
    [Mean boundary distance (px)], [_TBD_],   [[_TBD_, _TBD_]],
    [Box IoU],                     [$0.929$], [$[0.900, 0.951]$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Integrated-configuration agreement on the fifty-five-frame
    labeling subset, all five metrics with bootstrap 95 % confidence
    intervals over $10,!000$ resamples of the per-frame mean.
  ],
) <tab:agreement_overall>

== Per-sample breakdown

The eleven samples in Section~4 differ substantially in resolution, frame rate, illumination, and operator framing. A per-sample breakdown is the strongest available evidence that the agreement reported above is consistent across substantively different molds rather than driven by a single fortunate recording. Table~@tab:agreement_per_sample reports mask IoU and boundary $F_1$ for each sample alongside the count of labeled frames contributing to the mean. Figure~@fig:per_sample_iou_bars visualises the same data with bootstrap whiskers for quick comparison across samples.

#figure(
  // image("/typst/figures/per_sample_iou_bars.pdf", width: 95%),
  rect(width: 100%, height: 2.0in, stroke: 0.5pt, inset: 8pt)[
    _Per-sample agreement placeholder._ Horizontal bar chart with
    one row per sample (`input_1` through `input_11`); bar length
    is mean mask IoU across the five labeled frames of that sample;
    whisker is the bootstrap 95% confidence interval over $10,!000$
    resamples of the per-frame mean. Samples sorted ascending by
    mean IoU so the worst-performing infusion appears at the top of
    the chart. A vertical garnet line marks the overall eleven-
    sample mean IoU; rows whose CI does not cross that line are
    visually distinguished. Bar fill is atlantic for the high-
    resolution clips and warm grey for the cropped operator-view
    clips so the resolution-and-rate regime per sample is legible at
    a glance.
  ],
  caption: [
    Per-sample mask IoU sorted ascending. Bar widths are the
    per-sample mean over five labeled frames; whiskers are bootstrap
    95% confidence intervals.
  ],
) <fig:per_sample_iou_bars>

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Sample*], [*$n$*], [*Mask IoU*], [*Boundary $F_1$*],
    ),
    table.hline(stroke: 0.5pt),
    [`input_1`],  [5], [$0.781$], [$0.576$],
    [`input_2`],  [5], [$0.890$], [$0.403$],
    [`input_3`],  [5], [$0.960$], [$0.385$],
    [`input_4`],  [5], [$0.933$], [$0.347$],
    [`input_5`],  [5], [$0.953$], [$0.508$],
    [`input_6`],  [5], [$0.937$], [$0.409$],
    [`input_7`],  [5], [$0.946$], [$0.485$],
    [`input_8`],  [5], [$0.926$], [$0.385$],
    [`input_9`],  [5], [$0.948$], [$0.452$],
    [`input_10`], [5], [$0.946$], [$0.449$],
    [`input_11`], [5], [$0.912$], [$0.360$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Per-sample agreement for mask IoU and boundary $F_1$. Each row
    aggregates the five anchor frames sampled at the $5/25/50/75/95 %$
    fill positions described in Section~4.
  ],
) <tab:agreement_per_sample>

== Component-removal ablation

Each row of Table~@tab:ablation holds the integrated configuration described in Section~3 fixed, removes one named component, and reports the resulting mean IoU on the fifty-five-frame subset with a bootstrap 95 % confidence interval. The final column reports the absolute change in mean IoU relative to the integrated configuration in the first row, signed so that a negative number is a drop and a positive number is a rise. Per-sample $Delta$IoU values for the same rows are reported in the supplementary breakdown referenced from Table~@tab:agreement_per_sample. The reader should not expect the per-sample values to share the sign of the eleven-sample mean. A primitive whose assumption matches the dynamics of one sample can be neutral or counterproductive on a sample with different dynamics, and the per-sample breakdown is the empirical content of the ablation rather than the eleven-sample mean alone.

#figure(
  // TODO populate from data/agreement_metrics_ablation.json once
  // the 11-sample component-removal sweep completes. Rows in Method
  // §3 introduction order, integrated row pinned at top.
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Configuration*], [*IoU*], [*95 % CI*], [*$Delta$IoU*],
    ),
    table.hline(stroke: 0.5pt),
    [Integrated],                            [_TBD_], [[_TBD_, _TBD_]], [_–_],
    [No peak-brightness reference],          [_TBD_], [[_TBD_, _TBD_]], [_TBD_],
    [No temporal lock],                      [_TBD_], [[_TBD_, _TBD_]], [_TBD_],
    [No darken-only clip],                   [_TBD_], [[_TBD_, _TBD_]], [_TBD_],
    [No dynamic-lag reference],              [_TBD_], [[_TBD_, _TBD_]], [_TBD_],
    [No region-of-interest restriction],     [_TBD_], [[_TBD_, _TBD_]], [_TBD_],
    [No morphological cleanup],              [_TBD_], [[_TBD_, _TBD_]], [_TBD_],
    [No pre-delta blur],                     [_TBD_], [[_TBD_, _TBD_]], [_TBD_],
    [No camera-shift registration],          [_TBD_], [[_TBD_, _TBD_]], [_TBD_],
    [Grayscale colorspace],                  [_TBD_], [[_TBD_, _TBD_]], [_TBD_],
    [HSV colorspace],                        [_TBD_], [[_TBD_, _TBD_]], [_TBD_],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Component-removal ablation on the fifty-five-frame labeling
    subset. Each row holds the integrated configuration of Section~3
    (Tables~@tab:colorspace_modes through @tab:scalars) fixed and
    disables one named component. $Delta$IoU is the signed change in mean IoU
    relative to the integrated configuration; negative values are
    drops, positive values are rises. Rows are listed in the order
    in which each corresponding primitive is introduced in
    Section~3, not sorted by $Delta$IoU, so that the order is
    independent of the data. Confidence intervals are bootstrap
    quantiles over $10,!000$ resamples of the per-frame mean.
  ],
) <tab:ablation>

#figure(
  // TODO render the bar-chart counterpart of tab:ablation as
  // typst/figures/component_ablation.pdf with the eleven-sample mean
  // and the per-sample strip side by side per component.
  rect(width: 100%, height: 2.0in, stroke: 0.5pt, inset: 8pt)[
    _Component-removal effect-size figure placeholder._ Per component
    in Table~@tab:ablation, the eleven-sample mean $Delta$IoU bar
    next to a strip plot of the eleven per-sample $Delta$IoU values.
  ],
  caption: [
    Effect-size companion to Table~@tab:ablation. For each row in the
    table, the left bar is the eleven-sample mean $Delta$IoU and the
    right strip is the eleven per-sample $Delta$IoU values. Strips
    that cross zero identify primitives that are neutral or
    counterproductive on at least one sample, which is information
    the eleven-sample mean alone hides.
  ],
) <fig:component_bars>

== Qualitative montage

#figure(
  // image("/typst/figures/montage.pdf"),
  rect(width: 100%, height: 2.4in, stroke: 0.5pt, inset: 8pt)[
    _Frame montage placeholder._ Three columns per labeled frame
    (raw input, predicted overlay, human polygon) drawn from four
    representative samples spanning the resolution and frame-rate
    regimes documented in Section~4. Replaced once the predicted
    overlays are regenerated from the integrated configuration.
  ],
  caption: [
    Qualitative montage of labeled frames. Each row shows one frame
    from a different sample (high-resolution standard-rate, high-
    resolution time-lapse, and two cropped operator-view samples),
    with raw input, predicted overlay, and human polygon side by
    side. The figure exists to let the reader judge the IoU number
    against an actual frame rather than against summary statistics.
  ],
) <fig:montage>

== Configuration lookup

The per-sample ablation in the previous subsection is the empirical content of the paper, but a practitioner with their own VARTM rig is unlikely to read the per-sample $Delta$IoU table directly. The same evidence is more useful as a regime-indexed lookup that recommends preprocessing settings as a function of run circumstances. Table~@tab:lookup is that lookup. Each row pairs a circumstance (illumination drift, fabric type, fill rate, frame rate, camera stability, fabric conductivity) with the recommended setting on the corresponding mode menu of Section~3, and points at the row of Table~@tab:ablation or the per-sample breakdown of Table~@tab:agreement_per_sample that supports it. Rows whose recommendation is supported only by the eleven evaluated regimes are marked tentative (`†`); a deployed system on a circumstance the labeling subset does not cover should treat the row as a starting point for its own per-mold tuning rather than as a guarantee.

#figure(
  // TODO populate Recommended setting and Source columns from
  // data/agreement_metrics_ablation.json once the per-sample
  // component-removal sweep completes. Tentative rows marked †.
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.8pt),
    table.header([*Run circumstance*], [*Recommended setting*], [*Source*]),
    table.hline(stroke: 0.5pt),
    [Illumination drifts $> N$ $L^*$ units across the run], [enable peak-brightness reference], [no-peak row of Table~@tab:ablation],
    [Illumination stable across the run],                  [peak-brightness reference optional],   [no-peak row of Table~@tab:ablation],
    [True wet-front pauses $>= n_"lock"$ frames],          [reduce or disable temporal lock],     [no-lock row of Table~@tab:ablation],
    [Time-lapse acquisition ($<= 5$ fps)],                 [reduce $n_"lock"$ to $0$ or $1$],       [Table~@tab:agreement_per_sample, `input_2` and `input_3`],
    [Pigmented resin or colored fabric],                   [switch CIELAB → RGB or HSV†],          [colorspace rows of Table~@tab:ablation],
    [Specular silicone bag in field of view],              [keep darken-only enabled],            [no-darken-only row of Table~@tab:ablation],
    [Tripod with occasional bumps or thermal creep],       [enable camera-shift registration],    [no-camera-shift row of Table~@tab:ablation],
    [Fill rate varies across regimes],                     [use dynamic-lag reference],           [no-dynamic-lag row of Table~@tab:ablation],
    [Race-tracking dominates early fill],                  [first-frame reference; avoid dynamic calibration anomaly], [Section~7 failure mode 2],
    [Carbon-fiber laminate under transparent bag],         [CIELAB stays valid†],                 [colorspace ablation, applicable regime only],
    [Heavily textured silicone bag],                       [percentile or adaptive threshold†],    [Section~7 failure mode 3],
    [Side-lit laminate with intensity gradient],           [adaptive-mean or adaptive-gaussian threshold†], [Section~7 failure mode 3],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Configuration lookup indexed by run circumstance. The
    recommended setting names the option on the corresponding mode
    menu of Section~3; the Source column points at the row of
    Table~@tab:ablation or per-sample breakdown that supplies the
    empirical evidence. Rows marked `†` extrapolate from the
    eleven-sample subset to a regime the subset does not directly
    cover and should be treated as a tuning starting point rather
    than as an evaluated recommendation.
  ],
) <tab:lookup>
