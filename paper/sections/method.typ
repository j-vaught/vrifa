#import "../lib.typ": td

= Method

== Pipeline overview

Unlike contemporary ML models, the detection algorithm introduced here treats each video frame as a comparison against a reference image, often of the dry preform. Where resin has wetted the fabric, the local appearance darkens, and the absolute change is largest near the advancing flow front. The pipeline first checks for camera motion against the previous frame and, when motion crosses a configured threshold, registers the live frame back into the reference coordinate system before any difference is computed. The registered frame is optionally smoothed with a pre-delta blur whose output feeds both the running peak map and the difference computation, so the reference and the comparison see the same bandwidth-limited input. The pipeline computes the per-pixel difference inside a rectangular region of interest, clips it to admit only darkening so that specular highlights from the vacuum bag are rejected, normalizes the resulting field, thresholds it into a binary candidate mask, cleans the mask with morphological closing and opening, removes small connected components, and finally locks pixels whose wet label has persisted for several consecutive frames. The locked mask is the canonical output, the optional heatmap renders the underlying delta field for visual diagnosis, and the optional overlay draws the mask boundary onto the original Vacuum-Assisted Resin Transfer Molding (VARTM) frame for human review. Figure~@fig:pipeline shows the sixteen stages in order.

#figure(
  image("/typst/figures/pipeline.pdf", width: 95%),
  caption: [
    Sixteen-stage integrated pipeline.
  ],
) <fig:pipeline>

The pipeline is structured as a configurable framework rather than a fixed algorithm. The sixteen stages collectively expose four colorspaces, three region-of-interest forms, five reference-selection modes, six blur kernels, six threshold modes, three structuring-element shapes, and a handful of other binary or numeric options. The "integrated configuration" referenced throughout this paper is one specific point in that menu space. The regime-indexed configuration lookup in Section~7 recommends alternative settings under operating conditions where the integrated point does not transfer.

#figure(
  image("/typst/figures/colorspace_projection.pdf", width: 95%),
  caption: [
    The nine channel projections of the three supported colorspaces.
  ],
) <fig:colorspace_projection>

== Frame decode and colorspace conversion

The first stage decodes each Blue-Green-Red (BGR) frame from the input video. The second projects that frame into a working colorspace. The pipeline supports four options, namely the International Commission on Illumination (CIE) 1976 $L^* a^* b^*$ colorspace (CIELAB), Red-Green-Blue (RGB), Hue-Saturation-Value (HSV), and 8-bit grayscale. The integrated configuration uses CIELAB because for the resin-and-fabric combinations in the eleven samples evaluated here, wetting primarily darkens the lightness channel $L^*$ rather than shifting chrominance, so the single-channel $L^*$ projection used in the difference computation captures most of the signal. The choice is regime-dependent rather than universal. A pigmented resin or a colored fabric could shift the balance of evidence into chrominance and make RGB or HSV preferable, which is why the alternatives remain selectable. The eleven-sample mean and per-sample breakdown of the colorspace effect are reported in Table~@tab:ablation. We denote the converted frame at index $t$ by $F_t$.



== Region of interest

The region of interest is a binary mask $R$ that is one inside the laminate and zero elsewhere; all subsequent stages operate only on pixels where $R = 1$. The pipeline supports three forms for constructing $R$, mutually exclusive, selected by which configuration parameter is supplied. The rectangular form takes four fractional margins, one per edge, each clamped to the closed interval from zero to forty-nine hundredths of the corresponding side, and builds the rectangle bounded by those margins. A single configuration parameter sets all four margins symmetrically, with per-edge overrides available. The rectangular form is appropriate for laminates that fit a rectangular bounding box with no internal fixtures, which describes every sample in the labeling subset of Section~4. The imported-PNG form reads a single-channel grayscale image at the source video's resolution from a path supplied via `--roi-mask`, thresholds it at 127, and uses the resulting binary image directly as $R$. The imported-PNG form is appropriate when the laminate is non-rectangular (for instance a curved part boundary) or when fixtures, sensors, or labels inside the bag must be excluded from the difference computation but lie inside any axis-aligned rectangle that contains the laminate. The imported-COCO form reads a Common Objects in Context (COCO) JSON file from the same `--roi-mask` flag, locates the image entry whose `file_name` matches the input video, and rasterizes every polygon annotation on that image into $R$. The imported-COCO form is appropriate when the laminate boundary is already available as a polygon in an existing labeling project. The integrated configuration uses the rectangular form with `roi-margin = 0.15`; the imported forms are exercised in the per-mold tuning regimes described in Section~7.

#figure(
  image("/typst/figures/roi_crop.pdf", width: 100%),
  caption: [
    Three forms of the ROI mask $R$ on the input frame.  ],
) <fig:roi_crop>

== Camera-shift detection and registration

A bumped tripod, a thermal expansion of the rig, or a hand brushing the camera in mid-run shifts the projected position of every laminate pixel by some vector that has nothing to do with wetting. The reference frame stops aligning with the live frame, the difference field lights up along high-contrast laminate edges, and the threshold catches that as wet. The pipeline detects the shift and corrects it before the difference is computed. For each frame, a single phase-correlation step @KuglinHines1975PhaseCorrelation on a fixed-resolution downsample of the working channel of the previous and current frames returns a translation $(d_x, d_y)$ and a confidence score. When either the per-frame magnitude $sqrt(d_x^2 + d_y^2)$ or the cumulative drift across a five-frame rolling window exceeds the configured threshold, an iterative Enhanced Correlation Coefficient refinement @EvangelidisPsarakis2008ECC fits a translation or affine warp $W_t$ on a static-edge mask of the current ROI, and the live frame is warped through $W_t$ into the reference coordinate system before all downstream stages. The static-edge mask is recomputed at each shift event so the registration is driven by mold and frame edges that do not move with the wet front rather than by the wet region itself. The peak map is either discarded (`peak-on-shift = reset`, used by the integrated configuration), so the post-warp pixels do not accumulate against pre-warp brightness, or warped through $W_t$ into the new coordinate system (`peak-on-shift = warp`), so the running maximum is preserved at the cost of carrying any residual registration error into the peak.

#figure(
  image("/typst/figures/camera_shift_pair.pdf", width: 100%),
  caption: [
    Pipeline delta field across the bumped-tripod event in input_1
    (frame 10 reference, frame 75 current, 4.7 px shift). The corrected panel preserves the real wetting while
    the registration warp removes the edge artifacts.
  ],
) <fig:camera_shift_pair>

== Peak-brightness reference

The pipeline accumulates a running maximum of the working channel across all frames seen so far. When the pre-delta blur is enabled the maximum is updated from the blurred working channel rather than the raw channel so the peak map and the delta input share the same bandwidth, which prevents a one-pixel halo of speckle from shifting the peak above the eventual delta input and creating a phantom darkening signal. The motivation for the running maximum is that, on infusions where the dry preform is the brightest state the fabric reaches in the working channel, a per-pixel maximum tracks lighting drift caused by lamp warm-up and bag deformation rather than treating that drift as wetting evidence. Figure~@fig:peak illustrates the regime on one tracked pixel of the canonical input clip. The raw $L^*$ value drifts upward as the lamps stabilize, the running peak $P$ tracks that drift, and the front arrival separates as a sharp drop of more than thirty units below the peak. A fixed reference at frame zero would have included the post-warm-up lift in its difference. The assumption that wetting is the only event that lowers the peak does not hold universally. A specular flash that drives the peak above its long-term value, or a fabric whose wet state is brighter than its dry state for some channel, can produce a peak map that misrepresents subsequent wetting evidence. The IoU effect of disabling the primitive on the eleven-sample subset is reported in Table~@tab:ablation.

#figure(
  image("/typst/figures/peak_reference.pdf", width: 95%),
  caption: [
    Four pixels of input_1 tracked across all 706 frames. Each pixel drifts upward during lamp warm-up,
    the running peak absorbs that drift, and the front arrival
    collapses $L^*$ sharply below $P$.
  ],
) <fig:peak>

The peak map can be disabled through a single configuration parameter. The "no peak-brightness reference" row of Table~@tab:ablation reports the IoU effect of doing so on each of the eleven samples, with the per-sample breakdown showing that the effect is not constant across infusion regimes.

== Reference selection

The reference image $G_t$ used for the difference computation has five modes. The first-frame mode pins $G_t = F_0$ for every $t$. The running mode updates an exponential moving average $G_t = (1 - alpha) G_(t-1) + alpha F_t$ with $alpha = 0.05$. The previous-frame mode uses a fixed-offset history $G_t = F_(t - k)$, with the buffer bootstrapped from $F_0$ until $t > k$. The absolute mode pins $G_t$ to a user-specified absolute frame index. The dynamic mode adapts the lag online from a square-root-area growth model fit to the early frames. The integrated configuration uses the first-frame mode, which, combined with the peak map above, gives a piecewise-static reference whose only adaptation is the peak update.

The dynamic mode warrants a closer look because the integrated configuration does not use it for the headline numbers and disabling it is one of the rows in Table~@tab:ablation. For the first ten frames after detection bootstrapping (the calibration window), the pipeline records the wet area $a_t$ inside the region of interest and the elapsed time $tau_t$ at frame $t$. It then estimates a growth-rate factor $kappa$ as the median of $a_t / tau_t^(3/2)$ across the calibration frames. The three-halves exponent is motivated by the area-growth law for radial Darcy infusion, where wetted area scales approximately as time to the three-halves power once flow is well established; the assumption holds approximately for the early-fill regime of the eleven samples in the labeling subset and breaks for race-tracking-dominated runs, which is why the dynamic-calibration window is sensitive to anomalous early frames as discussed in Section~7. Given $kappa$ and a target wet fraction $rho$ of the region-of-interest area $|R|$ (held at $0.2$ in the integrated configuration), the dynamic lag $Delta tau_t$ in seconds that places the reference at the moment when the wet area was a fraction $rho$ of the current area is

$ Delta tau_t = lambda dot.c [ ( (rho |R|) / kappa + sqrt(tau_t) )^2 - tau_t ], $ <eq:dynlag>

clipped to be non-negative and scaled by a user lag factor $lambda$. The reference frame is then read from a small cache at the integer index whose elapsed time is closest to $tau_t - Delta tau_t$, falling back to the first frame whenever the calibration has not yet produced a finite $kappa$. A linear-mode override replaces the sqrt-area growth fit with a linear lag schedule parameterised by `dynamic-lag-linear-start` and `dynamic-lag-linear-max`, which steps the reference frame back at a constant rate independent of the calibration estimate. The override is intended for diagnostic comparisons where the sqrt-area assumption is suspect, and a per-frame log of the chosen lag is written to `dynamic-lag-log` for post-hoc inspection.

#figure(
  image("/typst/figures/reference_modes.pdf", width: 100%),
  caption: [
    The five reference-selection modes plus linear-lag dynamic
    override, all evaluated on input_1 frame 352. 
  ],
) <fig:reference_modes>

== Delta computation

Stage seven produces the per-pixel scalar field $D_t$ that drives all downstream decisions. The integrated configuration's darken-only mode operates on the working (first) channel and records how much darker the frame is than its reference,

$ D_t (y, x) = R(y, x) dot.c max(0, w_0 dot.c (G_t (y, x) - F_t (y, x, 0))), $ <eq:delta_darken>

where the reference $G_t$ is the running peak map (per-pixel maximum of the working channel across all prior frames) when peak-reference is enabled, and the working-channel slice of the frame chosen by the reference-selection mode otherwise. The clip to non-negative values discards every pixel that becomes brighter than the reference, which removes specular flashes from the silicone vacuum bag and from condensation, neither of which are wetting events. A full-color mode replaces the difference with the channel-weighted Euclidean distance across all channels and is intended for HSV and RGB workflows where chrominance shifts are diagnostic. The per-channel weights $w_0, w_1, w_2$ are exposed via `channel-weights` and equal $1, 1, 1$ in the integrated configuration; non-uniform weights are appropriate when one channel of the working colorspace carries the wetting signal more strongly than the others. Figure~@fig:darken_only shows the effect of the clip on a single frame; Table~@tab:ablation reports the IoU cost of removing it across the eleven-sample subset.

#figure(
  image("/typst/figures/darken_only_compare.pdf", width: 100%),
  caption: [
    Input frame from input_2 (frame 15 against frame 0 reference)
    under two delta computations. The naive Euclidean delta lights up
    the left side.
    The darken-only delta clips that brightening to zero.
  ],
) <fig:darken_only>

== Post-delta blur

The delta field is smoothed by the post-delta blur stage, which is a single function exposed by the `blur.rs` module and shared with the optional pre-delta blur of stage four. The user selects a kernel kind from {flat, gaussian, triangle, median, bilateral, none} together with a kernel size, written as a single specification of the form KIND[:SIZE]. The integrated configuration uses `gaussian:9`, a separable Gaussian of size $k_b = 9$ pixels (forced odd at runtime). Gaussian is appropriate when speckle is approximately white noise around the underlying response field; flat and triangle are exposed because some camera-and-bag combinations produce structured noise that the corresponding box or tent filter handles with less bias. Routing both the pre- and post-delta blurs through the same module keeps their behavior identical at matched specifications and ensures that the bandwidth-limited input the peak map sees in stage four is the same kind of bandwidth-limited input the threshold sees in stage nine.

#figure(
  image("/typst/figures/pre_post_blur.pdf", width: 95%),
  caption: [
    Pre- and post-delta blur on the input_1 bump event.  
    Top row, working channel $L^*$ after pre-delta blur.
  ],
) <fig:pre_post_blur>

== Normalization and threshold

The smoothed field is rescaled to the byte range using a min-max linear rescaling, producing a $tilde(D)_t$ that fits in eight bits and feeds both the threshold-selection stage and the heatmap renderer. Six thresholding modes are exposed through a single specification of the form KIND[:ARGS]. The four global modes share a single offset $delta_tau$ that is added before binarization. Otsu's between-class variance method @Otsu1979Threshold recovers an automatic threshold $tau_"otsu"$ over the full $tilde(D)_t$ histogram and adds the offset, which is the integrated configuration's choice and works on infusions whose histograms are roughly bimodal. The Triangle method recovers $tau_"tri"$ by the geometric construction of Zack and co-workers on the histogram @Zack1977Triangle, which is appropriate when one class dominates the histogram (typically early fill, where most pixels are dry). Manual mode uses a user-supplied absolute byte value $tau_"man"$ plus $delta_tau$; percentile mode sorts the ROI pixels and recovers the $p$-th percentile by linear interpolation. The two adaptive modes, adaptive-mean and adaptive-gaussian, compute a per-pixel threshold from a $b times b$ neighborhood mean (or Gaussian-weighted mean) minus a constant $C$, and bypass $delta_tau$ because $C$ already serves the same role. Adaptive thresholding is appropriate when the delta retains a low-frequency intensity gradient that the reference stage did not fully cancel, for instance under uneven side-lighting or vignette artifacts. The integrated configuration uses Otsu with $delta_tau = -30$, which biases the global threshold toward the wet class on infusions where the partially-wetted halo around the front sits below the bimodal split that Otsu finds. The bias is appropriate for the early-fill regime of the eleven labeled samples and not appropriate for every infusion. The Triangle, manual, percentile, and adaptive variants remain selectable for infusions whose histograms or intensity gradients fit those modes better; the integrated configuration does not exercise them, and the headline numbers of Section~5 are reported under Otsu plus offset.

#figure(
  // image("/typst/figures/threshold_modes.pdf", width: 95%),
  rect(width: 100%, height: 2.0in, stroke: 0.5pt, inset: 8pt)[
    _Threshold-mode comparison placeholder._ Six-panel grid of
    binary masks produced by each threshold mode on the same
    normalized response field $tilde(D)_t$. Panels: Otsu plus offset
    (used by the integrated configuration), Triangle, manual at
    $tau_"man" = 64$, percentile at $p = 70$, adaptive-mean with
    $b = 21$ and $C = 10$, adaptive-gaussian with $b = 21$ and
    $C = 10$. Above the grid, the histogram of the response is
    plotted with the Otsu and Triangle thresholds annotated. The
    integrated mode is bordered in garnet.

  ],
  caption: [
    The six threshold modes the pipeline supports applied to the
    same normalized response field.
  ],
) <fig:threshold_modes>

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.8pt),
    table.header([*Option*], [*Description*]),
    table.hline(stroke: 0.5pt),
    [otsu],              [Between-class variance threshold over the histogram of $tilde(D)_t$ @Otsu1979Threshold, plus offset $delta_tau$.],
    [triangle],          [Geometric construction of Zack and co-workers on the histogram @Zack1977Triangle, plus offset; appropriate when one class dominates the histogram.],
    [manual],            [User-supplied absolute byte value $tau_"man"$, plus offset.],
    [percentile],        [$p$-th percentile of ROI pixels of $tilde(D)_t$ by linear interpolation, plus offset.],
    [adaptive-mean],     [Per-pixel threshold from a $b times b$ neighborhood mean minus constant $C$. Bypasses $delta_tau$.],
    [adaptive-gaussian], [Per-pixel threshold from a $b times b$ Gaussian-weighted neighborhood mean minus constant $C$. Bypasses $delta_tau$.],
    table.hline(stroke: 0.8pt),
  ),
  caption: [Threshold modes selectable via the `KIND[:ARGS]` specification of `--threshold`.],
) <tab:threshold_modes>

== Morphological close

Stage ten passes the binary mask through morphological closing with an elliptical structuring element of size $k_m$ (default thirteen pixels). Closing welds neighbouring wet patches into a single front and fills small gaps where transient bag wrinkles muted the local response. The kernel size is the parameter that sets the spatial scale at which gaps are considered noise rather than real disconnections in the front; on the resolutions in the eleven labeled samples a $13 times 13$ ellipse is large enough to bridge bag-wrinkle gaps and small enough to leave genuinely separate wet regions separate.

== Morphological open and area filter

Stage eleven passes the closed mask through morphological opening with the same kernel and shape, removing specks below the kernel size that survived the closing pass. A connected-components labelling pass then discards any region whose pixel area is below $a_"min"$ pixels (default $400$), which removes the small islands that the morphology kernels are too small to suppress. Figure~@fig:cleanup shows the same frame at every step of stages nine through eleven.

#figure(
  // image("/typst/figures/mask_cleanup.pdf", width: 100%),
  rect(width: 100%, height: 1.4in, stroke: 0.5pt, inset: 8pt)[
    _Mask-cleanup montage placeholder._ Five panels for one frame:
    the normalized response, the threshold output, after closing,
    after opening, and after the connected-components area filter.
    Regenerated from the new ablation runs.
  ],
  caption: [
    Mask cleanup on the first canonical input clip at frame 200.
    The normalized response field (1) is thresholded into a noisy
    binary mask (2). Closing welds neighbouring wet patches and
    fills small gaps (3). Opening removes specks the closing
    kernel could not fill (4). Connected-components labelling
    removes the few residual islands below the four-hundred-pixel
    area floor, leaving the final mask on which the temporal lock
    operates (5).
  ],
) <fig:cleanup>

The kernel parities are forced odd at runtime so that anchor handling is symmetric. The structuring-element shape is elliptical by default with optional rectangular and cross-shaped alternatives, exposed because the front shape changes with infusion geometry.

#figure(
  // image("/typst/figures/morph_kernels.pdf", width: 95%),
  rect(width: 100%, height: 1.4in, stroke: 0.5pt, inset: 8pt)[
    _Morphological-kernel comparison placeholder._ Three panels of
    the cleaned mask under each structuring-element shape with
    $k_m = 13$. Left: ellipse (used by the integrated configuration),
    isotropic, appropriate for round fronts. Center: rectangle,
    asymmetric, appropriate for stripe-shaped wet regions. Right:
    cross, minimal, preserves corners. The integrated panel is
    bordered in garnet.
  ],
  caption: [
    Morphological structuring-element shape applied to the same
    threshold output. The integrated configuration uses ellipse.
  ],
) <fig:morph_kernels>

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.8pt),
    table.header([*Option*], [*Description*]),
    table.hline(stroke: 0.5pt),
    [ellipse], [Disk-shaped structuring element. Isotropic; appropriate for round fronts.],
    [rect],    [Rectangular structuring element. Asymmetric; appropriate for stripe-shaped wet regions.],
    [cross],   [Plus-shaped structuring element. Minimal; preserves corners.],
    table.hline(stroke: 0.8pt),
  ),
  caption: [Structuring-element shapes selectable via `--morph-shape`. Used by both the closing and opening passes.],
) <tab:morph_modes>

== Temporal locking

Stage twelve imposes hysteresis along the time axis. Each pixel keeps a small per-pixel counter that increments while the cleaned mask reports the pixel as wet and resets to zero on any frame where the cleaned mask says dry. Once the counter reaches the threshold $n_"lock"$ frames, a sticky locked-pixel map is set true at that location and never resets. The output of the stage is the elementwise OR of the cleaned mask with the sticky locked map, so a pixel that has ever been wet for $n_"lock"$ consecutive frames stays wet for the remainder of the run. Figure~@fig:lock illustrates the bookkeeping with a twelve-frame example for $n_"lock" = 3$.

#figure(
  image("/typst/figures/lock_timeline.pdf", width: 95%),
  caption: [
    Lock state for one pixel across twelve frames with
    $n_"lock" = 3$. The detection row records what the cleaned mask
    said this frame. The counter row tracks consecutive wet frames
    and resets on dry. The locked row latches at the first frame
    where the counter reaches three (frame eight in this example)
    and never returns to false. A flicker that survives for fewer
    than three frames is treated as a transient and not committed.
  ],
) <fig:lock>

The integrated configuration uses $n_"lock" = 3$, which holds a positive detection in the mask for three subsequent frames. The latched-after-three-frames behavior is appropriate for infusions whose true pauses are shorter than the lock window and inappropriate for infusions whose pauses are longer, where the latch produces phantom regions that look like artifacts as discussed in Section~7. Setting the lock window to zero disables the stage altogether. The "no temporal lock" row of Table~@tab:ablation reports the per-sample effect, which depends on whether the live infusion contains pauses on the order of three frames or longer.

== Heatmap, overlay, and contour export

The last two display stages exist for inspection and label export. The heatmap renderer maps the normalized delta $tilde(D)_t$ through the Turbo colormap to produce a three-channel pseudocolor image. The overlay renderer extracts the boundary of the locked mask via a five-by-five rectangular morphological gradient and paints those boundary pixels red on a copy of the original BGR frame. Neither renderer is part of the detection logic; they expose, in human-readable form, the field on which the threshold acted and the boundary that the locked mask encloses.

For machine-readable export, the contour-extraction stage emits Common Objects in Context (COCO) and YOLO-format polygons for every connected component of the locked mask. Polygon segmentation is computed with the Suzuki-Abe topological border-following algorithm @Suzuki1985Border, optionally simplified by the Douglas-Peucker algorithm @DouglasPeucker1973 with tolerance $epsilon$, and optionally densified to a maximum edge length so that downstream rasterization preserves curvature. The annotation-sampling utility selects which frames receive labels using one of three modes, namely all-frame, evenly-spaced count, or fixed-stride, with deduplication of consecutive ties so that the integer-truncated linear-spacing exactly reproduces the standard reference behaviour.

#figure(
  // image("/typst/figures/heatmap_overlay_contour.pdf", width: 100%),
  rect(width: 100%, height: 1.8in, stroke: 0.5pt, inset: 8pt)[
    _Render-output placeholder._ Four panels for one frame of the
    canonical reference video. Panel 1: raw BGR input. Panel 2:
    heatmap render of the normalized response field $tilde(D)_t$
    through the Turbo colormap. Panel 3: overlay of the locked-mask
    boundary in red on the original BGR frame. Panel 4: COCO-format
    polygon export drawn over the input with the polygon vertices
    marked. Together the four panels show what the pipeline emits
    for human review and machine-readable export.
  ],
  caption: [
    The three render outputs for one frame: heatmap (panel 2),
    overlay (panel 3), COCO contour export (panel 4). The raw
    input (panel 1) is shown for reference.
  ],
) <fig:heatmap_overlay_contour>

== Scalar parameters held fixed

Table~@tab:scalars collects the scalar parameter values held fixed across every experiment reported in this paper. The mode menus that each subsection above exposes are documented in their respective tables and are pinned in prose at "the integrated configuration uses..."; the scalars are listed here in one compact table so a reader can recover the exact operating point without scraping ten subsections. The values were chosen during development on a held-out subset disjoint from the labeled evaluation set and were not retuned per video or per metric.

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (left, right, left),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.8pt),
    table.header([*Symbol*], [*Value*], [*Where used*]),
    table.hline(stroke: 0.5pt),
    [`roi-margin`],                     [$0.15$],         [Symmetric fractional region-of-interest margin per edge.],
    [`motion-per-frame-threshold`],     [$1.5$ px],       [Per-frame translation magnitude that triggers registration.],
    [`cumulative-motion-threshold`],    [$3$ px],         [Five-frame rolling cumulative drift that triggers registration.],
    [$alpha$],                          [$0.05$],         [Exponential-moving-average weight for the running reference.],
    [$k_p$],                            [$0$],            [Pre-delta blur kernel size (zero disables the stage).],
    [$k_b$],                            [$9$],            [Post-delta blur kernel size in pixels (Gaussian).],
    [$delta_tau$],                      [$-30$],          [Threshold offset added to Otsu / Triangle / manual / percentile.],
    [$k_m$],                            [$13$],           [Morphological structuring-element size in pixels.],
    [`morph-close-iterations`],         [$1$],            [Closing iterations applied at stage ten.],
    [`morph-open-iterations`],          [$1$],            [Opening iterations applied at stage eleven.],
    [$a_"min"$],                        [$400$ px],       [Minimum connected-component area accepted by the area filter.],
    [$n_"lock"$],                       [$3$],            [Consecutive-wet frames required to latch a pixel in the lock map.],
    [`dynamic-calibration-frames`],     [$10$],           [Frames used to fit the dynamic-reference factor $kappa$.],
    [$rho$],                            [$0.2$],          [Target wet-area fraction for the dynamic-reference lag.],
    [$lambda$],                         [$1.0$],          [Multiplicative lag-scale factor in Eq.~@eq:dynlag.],
    [`dynamic-ref-cache-size`],         [$32$ frames],    [Frames cached for the dynamic-reference reader.],
    [`frame-step`],                     [$1$],            [Frame stride at decode time.],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Scalar parameter values held fixed in the integrated configuration
    across every experiment reported in this paper. The only equation
    that depends on them explicitly is the dynamic-mode lag of
    Eq.~@eq:dynlag. Mode-menu choices (colorspace, motion-model, peak-
    on-shift, ref-mode, blur kind, threshold kind, morph-shape) are
    documented in the per-subsection tables.
  ],
) <tab:scalars>
