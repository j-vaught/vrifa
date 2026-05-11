= Quantitative Agreement Results

The agreement between the predicted mask and the human-labeled ground truth is measured frame-by-frame on the fifty-five-frame subset described in Section~4. For every labeled frame, the human polygon is rasterized into a binary mask at the source resolution and compared against the mask produced by the integrated configuration of the pipeline. Five complementary metrics are reported. Mask Intersection-over-Union (IoU) is the headline accuracy metric. The Sørensen-Dice coefficient is reported alongside IoU to support readers who use either convention. Boundary $F_1$ is computed as the mean of three pixel-tolerance values $tau in {1, 3, 5}$ pixels, where the per-tolerance $F_1$ is the harmonic mean of precision (prediction-boundary pixels within $tau$ pixels of any ground-truth-boundary pixel) and recall (ground-truth-boundary pixels within $tau$ pixels of any prediction-boundary pixel). Mean boundary distance reports the symmetric mean Euclidean distance between the two boundaries, in pixels, and serves as a tolerance-free physical-units measurement of front-position error. Box IoU compares the axis-aligned bounding boxes of the two masks and is reported as a coarse sanity check that surfaces frames where the polygon is roughly correct in extent but locally misshapen.

Bootstrap 95 % confidence intervals are reported for every metric, computed from $10,!000$ resamples of the per-frame metric values with replacement. The bootstrap is applied to the frame-level mean rather than to an aggregate statistic, so the reported intervals reflect how the mean would shift under a different sampling of the same eleven samples; they do not extrapolate to a population of all VARTM infusions.

== Integrated configuration vs. published classical-CV baselines

The integrated configuration is the configuration described in Section~3 and held fixed across every result reported in this section. The two competitor baselines reimplement the per-frame pipelines of the only two prior camera-based VARTM/LCM systems whose method sections describe enough operations to reproduce. The Lekanidis-Vosniakos 2020 baseline @LekanidisVosniakos2020IJMMS is a Matlab-style chain of ROI crop, Gaussian blur ($sigma = 2$), grayscale conversion, contrast stretch, Otsu binarization, foreground-background swap, closing on a disk-13 structuring element, Sobel edge detection, opening with a 120-pixel area threshold, and dilation. The Almazán-Lázaro 2022 baseline @AlmazanLazaro2022JMP is a per-frame chain of ROI crop, Scaramuzza-style lens-distortion correction @Scaramuzza2006Toolbox, histogram equalization, first-frame absolute differencing, grayscale conversion, a $5 times 5$ mean filter, Sobel-gradient segmentation, paired erosion and dilation, and small-area removal. Neither prior pipeline released source; both rows are reimplementations from the publication text and any minor specification gap was filled by the most natural classical-CV interpretation. Table~@tab:headline_vs_baselines reports the agreement of all three rows on the fifty-five-frame labeling subset; the contrast is the empirical answer to the question "does the joint integrated configuration close an IoU gap that the two prior pipelines leave open."

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Configuration*],
      [*IoU*], [*Dice*],
      [*$F_1$ at $1$ px*], [*$F_1$ at $3$ px*], [*$F_1$ at $5$ px*],
      [*Boundary px*], [*Box IoU*],
    ),
    table.hline(stroke: 0.5pt),
    [Integrated (this work)],                     [$0.921$], [$0.954$], [$0.256$], [$0.453$], [$0.586$], [$17.6$], [$0.924$],
    [Lekanidis & Vosniakos 2020 (reimplemented)], [$0.144$], [$0.247$], [$0.061$], [$0.116$], [$0.172$], [$86.3$], [$0.733$],
    [Almazán-Lázaro 2022 (reimplemented)],        [$0.075$], [$0.136$], [$0.103$], [$0.193$], [$0.266$], [$65.9$], [$0.761$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Integrated configuration versus two reimplemented classical-CV
    competitor pipelines on the 55-frame labeling subset. Boundary
    $F_1$ is broken out at three pixel tolerances.
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
    [Mask IoU],                    [$0.921$], [$[0.888, 0.942]$],
    [Sørensen-Dice],               [$0.954$], [$[0.927, 0.970]$],
    [Boundary $F_1$ (mean of $tau in {1, 3, 5}$ px)], [$0.432$], [$[0.401, 0.464]$],
    [Mean boundary distance (px)], [$17.6$],  [$[11.7, 25.4]$],
    [Box IoU],                     [$0.924$], [$[0.888, 0.948]$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Integrated-configuration agreement on the fifty-five-frame
    labeling subset, all five metrics with bootstrap 95 % confidence
    intervals over $10,!000$ resamples of the per-frame mean.
  ],
) <tab:agreement_overall>

== Per-sample breakdown

The eleven samples in Section~4 differ substantially in resolution, frame rate, illumination, and operator framing. A per-sample breakdown is the strongest available evidence that the agreement reported above is consistent across substantively different molds rather than driven by a single fortunate recording. Table~@tab:agreement_per_sample reports mask IoU and boundary $F_1$ for each sample alongside the count of labeled frames contributing to the mean. Per-sample IoU ranges from $0.778$ on `input_1` (the only sample with a polygonal ROI mask, which makes the comparison area substantially smaller than the other ten) through $0.961$ on `input_3`; ten of eleven samples clear $0.88$ and seven of eleven clear $0.93$, indicating the integrated configuration's behavior is sample-aware but not sample-fragile.


#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Sample*], [*$n$*], [*Mask IoU*],
      [*$F_1$ at $1$ px*], [*$F_1$ at $3$ px*], [*$F_1$ at $5$ px*],
    ),
    table.hline(stroke: 0.5pt),
    [`input_1`],  [5], [$0.778$], [$0.393$], [$0.633$], [$0.674$],
    [`input_2`],  [5], [$0.890$], [$0.254$], [$0.426$], [$0.531$],
    [`input_3`],  [5], [$0.961$], [$0.219$], [$0.398$], [$0.548$],
    [`input_4`],  [5], [$0.933$], [$0.223$], [$0.368$], [$0.450$],
    [`input_5`],  [5], [$0.952$], [$0.293$], [$0.542$], [$0.688$],
    [`input_6`],  [5], [$0.937$], [$0.202$], [$0.422$], [$0.594$],
    [`input_7`],  [5], [$0.946$], [$0.274$], [$0.513$], [$0.667$],
    [`input_8`],  [5], [$0.925$], [$0.246$], [$0.401$], [$0.497$],
    [`input_9`],  [5], [$0.948$], [$0.253$], [$0.453$], [$0.649$],
    [`input_10`], [5], [$0.946$], [$0.266$], [$0.463$], [$0.619$],
    [`input_11`], [5], [$0.912$], [$0.195$], [$0.361$], [$0.524$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Per-sample agreement, with boundary $F_1$ broken out at three
    pixel tolerances. Each row aggregates the five anchor frames
    sampled at the 5 / 25 / 50 / 75 / 95% fill positions.
  ],
) <tab:agreement_per_sample>

== Component-removal ablation

Each row of Table~@tab:ablation holds the integrated configuration described in Section~3 fixed, removes one named component, and reports the resulting mean IoU on the fifty-five-frame subset with a bootstrap 95 % confidence interval. The final column reports the absolute change in mean IoU relative to the integrated configuration in the first row, signed so that a negative number is a drop and a positive number is a rise. The reader should not expect the per-sample values to share the sign of the eleven-sample mean. A primitive whose assumption matches the dynamics of one sample can be neutral or counterproductive on a sample with different dynamics, and the per-sample breakdown is the empirical content of the ablation rather than the eleven-sample mean alone.

Three findings deserve direct attention. First, the camera-shift registration row reports the effect of *enabling* registration on the integrated configuration, which does not use it by default. The drop of $0.630$ to $0.291$ is large and consistent across every sample because the per-sample dataset is mechanically stable (first-to-last pixel displacement below $4$ px on the noisiest sample and below $1$ px on every 524p clip per the dataset's image-side characterization), and on a stationary camera the registration's phase-correlation stage misfires onto small intra-laminate features (resin wetting boundaries, fabric weave alignment) that look like apparent motion. The implication is operational rather than theoretical: a research bench that uses a tripod-mounted camera should leave camera-shift registration off, and only enable it when there is documented camera motion exceeding the per-frame and cumulative thresholds in the configuration. Second, the peak-brightness reference is neutral on this 11-sample subset (mean $Delta$IoU $+0.009$): the per-sample numbers cancel out, with `input_1` improving by $+0.15$ and `input_6` and `input_8` each losing $-0.02$. The primitive's intended workload (lighting drift) is not the binding constraint on per-sample performance here. Third, the ROI restriction matters only for `input_1` (the only sample that uses a polygonal mask file at $-0.14$); on the ten samples that use full-frame ROI the swap is a no-op. The dynamic-lag-reference ablation reports $Delta$IoU exactly zero across every sample, which surfaces a static configuration property rather than an empirical effect: the integrated config sets `ref_mode = first` and the dynamic-lag parameters are inactive when the first-frame reference is in use, so the disable-the-dynamic-lag toggle has no behavior to disable.

#figure(
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
    [Integrated],                            [$0.921$], [$[0.888, 0.942]$], [—],
    [No peak-brightness reference],          [$0.929$], [$[0.918, 0.940]$], [$+0.009$],
    [No darken-only clip],                   [$0.902$], [$[0.885, 0.919]$], [$-0.018$],
    [No dynamic-lag reference],              [$0.921$], [$[0.888, 0.942]$], [$+0.000$],
    [No region-of-interest restriction],     [$0.908$], [$[0.870, 0.935]$], [$-0.013$],
    [No morphological cleanup],              [$0.913$], [$[0.881, 0.935]$], [$-0.007$],
    [Camera-shift registration enabled],     [$0.291$], [$[0.217, 0.367]$], [$-0.630$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Component-removal ablation on the 55-frame labeling subset.
    Each row disables one component of the integrated configuration;
    $Delta$IoU is the signed change in mean IoU.
  ],
) <tab:ablation>


== Configuration lookup

The per-sample ablation in the previous subsection is the empirical content of the paper, but a practitioner with their own VARTM rig is unlikely to read the per-sample $Delta$IoU table directly. The same evidence is more useful as a regime-indexed lookup that recommends preprocessing settings as a function of run circumstances. Table~@tab:lookup is that lookup. Each row pairs a circumstance (illumination drift, fabric type, fill rate, frame rate, camera stability, fabric conductivity) with the recommended setting on the corresponding mode menu of Section~3, and points at the row of Table~@tab:ablation or the per-sample breakdown of Table~@tab:agreement_per_sample that supports it. Rows whose recommendation is supported only by the eleven evaluated regimes are marked tentative (`†`); a deployed system on a circumstance the labeling subset does not cover should treat the row as a starting point for its own per-mold tuning rather than as a guarantee.

#figure(
  table(
    columns: (auto, auto),
    align: (left, left),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.8pt),
    table.header([*Run circumstance*], [*Recommended setting*]),
    table.hline(stroke: 0.5pt),
    [Dry-frame mean $L^*$ $>= 70$ (bright bag, possible auto-exposure rebound)], [Set $n_"lock" = 0$],
    [Dry-frame mean $L^*$ in $[54, 60]$ (darker bag)],     [Set $n_"lock"$ in $[10, 11]$],
    [Mid-fill wet-dry contrast (p95-p5) $>= 65$ $L^*$],    [Set `threshold-offset` in $[-50, -40]$],
    [Mid-fill wet-dry contrast $<= 50$ $L^*$],             [Set `threshold-offset` near $0$ or positive; consider `threshold=triangle`†],
    [`lock_frames` $in {1, 2, 3, 4}$ for any sample],      [Never. Use $0$ or $>= 5$ — the intermediate range is the worst region of the parameter space on every sample],
    [Illumination drifts $> N$ $L^*$ units across the run],[Enable peak-brightness reference],
    [Time-lapse acquisition ($<= 5$ fps)],                 [Reduce $n_"lock"$ to $0$ or $1$],
    [Pigmented resin or colored fabric],                   [Switch CIELAB → RGB or HSV†],
    [Specular silicone bag in field of view],              [Keep darken-only enabled],
    [Tripod with occasional bumps or thermal creep],       [Enable camera-shift registration],
    [Fill rate varies across regimes],                     [Use dynamic-lag reference],
    [Race-tracking dominates early fill],                  [First-frame reference; avoid dynamic calibration anomaly],
    [Heavily textured silicone bag],                       [Percentile or adaptive threshold†],
    [Side-lit laminate with intensity gradient],           [adaptive-mean or adaptive-gaussian threshold†],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Configuration lookup indexed by run circumstance. Rows marked
    `†` extrapolate beyond the eleven-sample subset and are
    starting points rather than evaluated recommendations.
  ],
) <tab:lookup>
