= Quantitative Agreement Results

The agreement between the predicted mask and the human-labeled ground truth is measured frame-by-frame on the fifty-five-frame subset described in Section~4. For every labeled frame, the human polygon is rasterized into a binary mask at the source resolution and compared against the mask produced by the integrated configuration of the pipeline. Five complementary metrics are reported. Mask Intersection-over-Union (IoU) is the headline accuracy metric. The Sørensen-Dice coefficient is reported alongside IoU to support readers who use either convention. Boundary $F_1$ is computed as the mean of three pixel-tolerance values $tau in {1, 3, 5}$ pixels, where the per-tolerance $F_1$ is the harmonic mean of precision (prediction-boundary pixels within $tau$ pixels of any ground-truth-boundary pixel) and recall (ground-truth-boundary pixels within $tau$ pixels of any prediction-boundary pixel). Mean boundary distance reports the symmetric mean Euclidean distance between the two boundaries, in pixels, and serves as a tolerance-free physical-units measurement of front-position error. Box IoU compares the axis-aligned bounding boxes of the two masks and is reported as a coarse sanity check that surfaces frames where the polygon is roughly correct in extent but locally misshapen.

Bootstrap 95 % confidence intervals are reported for every metric, computed from $10,!000$ resamples of the per-frame metric values with replacement. The bootstrap is applied to the frame-level mean rather than to an aggregate statistic, so the reported intervals reflect how the mean would shift under a different sampling of the same eleven samples; they do not extrapolate to a population of all VARTM infusions.

== Integrated configuration vs. naive baseline

The integrated configuration is the configuration described in Section~3 and held fixed across every result reported in this section, with values listed in Table~@tab:defaults. The naive baseline runs the same pipeline with every named component disabled, namely no peak-brightness reference, no darken-only clip, no temporal lock, no morphological cleanup, and no dynamic-lag reference selection. The naive baseline retains only the colorspace projection, the ROI restriction, and the Otsu threshold against the first-frame reference, which is the minimum that any prior camera-based VARTM system reports doing. The contrast between the two rows is the empirical answer to the question "does the joint configuration matter for IoU on this benchmark."

#figure(
  // TODO populate from paper/data/agreement_metrics.json once the
  // 11-sample agreement run completes against the integrated
  // configuration and the naive baseline.
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    stroke: 0.4pt,
    inset: 5pt,
    table.header(
      [*Configuration*],
      [*IoU*], [*Dice*], [*B-$F_1$*], [*Boundary px*], [*Box IoU*],
    ),
    [Integrated], [_TBD_], [_TBD_], [_TBD_], [_TBD_], [_TBD_],
    [Naive baseline], [_TBD_], [_TBD_], [_TBD_], [_TBD_], [_TBD_],
  ),
  caption: [
    Integrated configuration vs. naive baseline on the
    fifty-five-frame labeling subset. The naive baseline runs the
    same pipeline with every named component disabled, retaining only
    the colorspace projection, ROI restriction, and Otsu threshold
    against the first-frame reference, which is the minimum any prior
    camera-based VARTM system reports doing. Each cell carries a
    bootstrap 95 % confidence interval over $10,!000$ resamples of
    the per-frame mean; intervals omitted from the table for
    readability and reported in the per-metric breakdown of
    Table~@tab:agreement_overall. B-$F_1$ is mean boundary $F_1$
    across $tau in {1, 3, 5}$ pixels.
  ],
) <tab:headline_vs_naive>

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    stroke: 0.4pt,
    inset: 6pt,
    table.header([*Metric*], [*Mean*], [*95 % CI*]),
    [Mask IoU], [_TBD_], [[_TBD_, _TBD_]],
    [Sørensen-Dice], [_TBD_], [[_TBD_, _TBD_]],
    [Boundary $F_1$], [_TBD_], [[_TBD_, _TBD_]],
    [Mean boundary distance (px)], [_TBD_], [[_TBD_, _TBD_]],
    [Box IoU], [_TBD_], [[_TBD_, _TBD_]],
  ),
  caption: [
    Integrated-configuration agreement on the fifty-five-frame
    labeling subset, all five metrics with bootstrap 95 % confidence
    intervals over $10,!000$ resamples of the per-frame mean.
  ],
) <tab:agreement_overall>

== Per-sample breakdown

The eleven samples in Section~4 differ substantially in resolution, frame rate, illumination, and operator framing. A per-sample breakdown is the strongest available evidence that the agreement reported above is consistent across substantively different molds rather than driven by a single fortunate recording. Table~@tab:agreement_per_sample reports mask IoU and boundary $F_1$ for each sample alongside the count of labeled frames contributing to the mean.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: 0.4pt,
    inset: 5pt,
    table.header(
      [*Sample*], [*$n$*], [*Mask IoU*], [*Boundary $F_1$*],
    ),
    [`input_1`],  [5], [_TBD_], [_TBD_],
    [`input_2`],  [5], [_TBD_], [_TBD_],
    [`input_3`],  [5], [_TBD_], [_TBD_],
    [`input_4`],  [5], [_TBD_], [_TBD_],
    [`input_5`],  [5], [_TBD_], [_TBD_],
    [`input_6`],  [5], [_TBD_], [_TBD_],
    [`input_7`],  [5], [_TBD_], [_TBD_],
    [`input_8`],  [5], [_TBD_], [_TBD_],
    [`input_9`],  [5], [_TBD_], [_TBD_],
    [`input_10`], [5], [_TBD_], [_TBD_],
    [`input_11`], [5], [_TBD_], [_TBD_],
  ),
  caption: [
    Per-sample agreement for mask IoU and boundary $F_1$. Each row
    aggregates the five anchor frames sampled at the $5/25/50/75/95 %$
    fill positions described in Section~4.
  ],
) <tab:agreement_per_sample>

== Component-removal ablation

The empirical claim of this paper is that the joint configuration matters and every named component contributes a measurable IoU delta to the joint. Table~@tab:ablation reports that claim directly. Each row holds the integrated configuration of Table~@tab:defaults fixed and removes one named component, the column reports the resulting mean IoU on the fifty-five-frame subset with a bootstrap 95 % confidence interval, and the final column reports the absolute IoU drop relative to the integrated configuration in the first row.

#figure(
  // TODO populate from paper/data/agreement_metrics_ablation.json once
  // the 11-sample component-removal sweep completes. Rows in order of
  // descending IoU drop; integrated row pinned at top.
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: 0.4pt,
    inset: 5pt,
    table.header(
      [*Configuration*], [*IoU*], [*95 % CI*], [*$Delta$IoU*],
    ),
    [Integrated],                            [_TBD_], [[_TBD_, _TBD_]], [—],
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
  ),
  caption: [
    Component-removal ablation on the fifty-five-frame labeling
    subset. Each row holds Table~@tab:defaults fixed and disables one
    named component. $Delta$IoU is absolute drop relative to the
    integrated configuration. Rows ordered by descending $Delta$IoU
    so that the components with the largest individual contribution
    appear first. Confidence intervals are bootstrap quantiles over
    $10,!000$ resamples of the per-frame mean.
  ],
) <tab:ablation>

#figure(
  // TODO render the bar-chart counterpart of tab:ablation as
  // typst/figures/component_ablation.pdf, sorted by descending IoU
  // drop with bootstrap CI whiskers, brand palette.
  rect(width: 100%, height: 2.0in, stroke: 0.5pt, inset: 8pt)[
    _Component-removal bar chart placeholder._ One bar per component
    in Table~@tab:ablation, IoU mean with bootstrap 95 % CI whiskers,
    sorted by descending $Delta$IoU.
  ],
  caption: [
    Bar-chart counterpart of Table~@tab:ablation. Bars are sorted by
    descending IoU drop relative to the integrated configuration, so
    the leftmost bars are the components whose removal hurts the
    most. The integrated bar is highlighted in garnet for reference.
  ],
) <fig:component_bars>

== Hyperparameter sensitivity around the operating point

The component-removal table reports the cost of disabling each named primitive completely. A separate set of single-axis sweeps reports the IoU response inside the neighborhood of the values held fixed in Table~@tab:defaults, so a reader can judge how flat the operating point is and how steep the cliffs are along each axis. Figure~@fig:ablation_curves shows the response on the high-resolution canonical reference video for the threshold offset $delta_tau$, the minimum component area $a_"min"$, the morphology kernel $k_m$, the delta-blur kernel $k_b$, the lock window $n_"lock"$, and a categorical panel for colorspace, darken-only, peak-reference, blur, and reference-mode. The curves are not the headline numbers, which average over all eleven samples, but they show how each axis behaves around the operating point on a single sample where the response is well isolated.

#figure(
  image("/typst/figures/ablation_curves.pdf", width: 95%),
  caption: [
    Single-axis IoU response on the canonical reference video as each
    parameter is varied with all other parameters held at the values
    in Table~@tab:defaults. Dashed grey marks the IoU at the held-
    fixed operating point; dotted green marks the joint optimum
    located by the wider sweep. The categorical panel reports IoU
    when each of the named binary or discrete components is toggled
    individually. Joint multi-component effects are reported in the
    component-removal ablation in Table~@tab:ablation.
  ],
) <fig:ablation_curves>

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
