= Quantitative Agreement Results

The agreement between the VRIFA mask and the human-labeled ground truth is measured frame-by-frame on the fifty-five-frame subset described in Section~5. For every labeled frame, the human polygon is rasterized into a binary mask at the source resolution and compared against the mask produced by the default configuration of the Rust binary. Five complementary metrics are reported. Mask Intersection-over-Union (IoU) is the headline accuracy metric. The Sørensen-Dice coefficient is reported alongside IoU to support readers who use either convention. Boundary $F_1$ is computed at the mean of three pixel tolerances $tau in {1, 3, 5}$ pixels, where the per-tolerance $F_1$ is the harmonic mean of precision (prediction-boundary pixels within $tau$ pixels of any ground-truth-boundary pixel) and recall (ground-truth-boundary pixels within $tau$ pixels of any prediction-boundary pixel). Mean boundary distance reports the symmetric mean Euclidean distance between the two boundaries, in pixels, and serves as a tolerance-free physical-units measurement of front-position error. Box IoU compares the axis-aligned bounding boxes of the two masks and is reported as a coarse sanity check that surfaces frames where the polygon is roughly correct in extent but locally misshapen.

Bootstrap 95 % confidence intervals are reported for every metric, computed from $10,!000$ resamples of the per-frame metric values with replacement. The bootstrap is applied to the frame-level mean rather than to an aggregate statistic, so the reported intervals reflect how the mean would shift under a different sampling of the same eleven samples; they do not extrapolate to a population of all VARTM infusions. The script that produces these numbers is committed at `_dev/validation/agreement.py`, and the resulting machine-readable JSON is committed at `paper/data/agreement_metrics.json`.

== Overall agreement

#figure(
  // TODO populate from paper/data/agreement_metrics.json once labels arrive.
  // Use #table(...) with mean ± 95% CI per metric, columns:
  // Metric | Mean | 95% CI
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
    Overall agreement across the fifty-five-frame labeling subset.
    Confidence intervals are bootstrap quantiles over
    ten-thousand resamples of the per-frame mean.
  ],
) <tab:agreement_overall>

// TODO replace once labels arrive: brief prose paragraph
// interpreting the overall numbers against the locked thesis
// (mask IoU X.XXX, boundary F1 Y.YYY).

== Per-sample breakdown

The eleven samples described in Section~5 differ substantially in resolution, frame rate, illumination, and operator framing, and the per-sample breakdown is the strongest available evidence that the agreement reported above is consistent across substantively different molds rather than driven by a single fortunate recording. Table~@tab:agreement_per_sample reports mask IoU and boundary $F_1$ for each sample alongside the count of labeled frames contributing to the mean.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    stroke: 0.4pt,
    inset: 5pt,
    table.header(
      [*Sample*], [*$n$*], [*Mask IoU*], [*Boundary $F_1$*],
    ),
    [`input_1`], [5], [_TBD_], [_TBD_],
    [`input_2`], [5], [_TBD_], [_TBD_],
    [`input_3`], [5], [_TBD_], [_TBD_],
    [`2026-02-17_…`], [5], [_TBD_], [_TBD_],
    [`2026-02-18_…`], [5], [_TBD_], [_TBD_],
    [`2026-02-19_…`], [5], [_TBD_], [_TBD_],
    [`2026-02-20_…`], [5], [_TBD_], [_TBD_],
    [`2026-02-23_…`], [5], [_TBD_], [_TBD_],
    [`2026-02-24_…`], [5], [_TBD_], [_TBD_],
    [`2026-02-25_…`], [5], [_TBD_], [_TBD_],
    [`2026-02-26_…`], [5], [_TBD_], [_TBD_],
  ),
  caption: [
    Per-sample agreement for mask IoU and boundary $F_1$. Each row
    aggregates the five anchor frames sampled at the
    $5/25/50/75/95 %$ fill positions described in Section~5.
  ],
) <tab:agreement_per_sample>

// TODO replace once labels arrive: prose paragraph interpreting
// per-sample variance, naming any sample where the default
// configuration produces meaningfully lower agreement than the
// overall mean and (per the protocol of Section~3) whether the
// contingency tuning pass was applied.

== Hyperparameter sensitivity around the operating point

The component-removal ablation reports the cost of disabling each named primitive completely. A separate set of single-axis sweeps reports the IoU response inside the neighborhood of the values held fixed in Table~@tab:defaults, so a reader can judge how flat the operating point is and how steep the cliffs are along each axis. Figure~@fig:ablation_curves shows the response on the high-resolution canonical reference video for the threshold offset $delta_tau$, the minimum component area $a_"min"$, the morphology kernel $k_m$, the delta-blur kernel $k_b$, the lock window $n_"lock"$, and a categorical panel for colorspace, darken-only, peak-reference, blur, and reference-mode. The curves are not the headline numbers, which average over all eleven samples, but they show how each axis behaves around the operating point on a single sample where the response is well isolated.

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
    component-removal ablation table earlier in this section.
  ],
) <fig:ablation_curves>

== Qualitative montage

#figure(
  // image("/typst/figures/montage.pdf"),
  rect(width: 100%, height: 2.4in, stroke: 0.5pt, inset: 8pt)[
    _Frame montage placeholder._ Three columns per labeled frame
    (raw input, predicted overlay, human polygon) drawn from four
    representative samples spanning the resolution and frame-rate
    regimes documented in Section~3. Replaced once the predicted
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
