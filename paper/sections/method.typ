#import "../lib.typ": td

= Method

== Pipeline overview

Unlike contemporary ML models, the detection algorithm introduced here treats each video frame as a comparison against a reference image, often of the dry preform. Where resin has wetted the fabric, the local appearance darkens, and the absolute change is largest near the advancing flow front. The pipeline first checks for camera motion against the previous frame and, when motion crosses a configured threshold, registers the live frame back into the reference coordinate system before any difference is computed. The registered frame is optionally smoothed with a pre-delta blur whose output feeds both the running peak map and the difference computation, so the reference and the comparison see the same bandwidth-limited input. The pipeline computes the per-pixel difference inside a rectangular region of interest, clips it to admit only darkening so that specular highlights from the vacuum bag are rejected, normalizes the resulting field, thresholds it into a binary candidate mask, cleans the mask with morphological closing and opening, removes small connected components, and finally locks pixels whose wet label has persisted for several consecutive frames. The locked mask is the output, the optional heatmap renders the underlying delta field for visual diagnosis, and the optional overlay draws the mask boundary onto the original VARTM frame for human review. Figure~@fig:pipeline shows the sixteen stages in order.

#figure(
  image("/typst/figures/pipeline.pdf", width: 95%),
  caption: [
    Sixteen-stage integrated pipeline.
  ],
) <fig:pipeline>

The pipeline is structured as a configurable framework rather than a fixed algorithm. The sixteen stages collectively expose four colorspaces, three region-of-interest forms, five reference-selection modes, six blur kernels, six threshold modes, three structuring-element shapes, and a handful of other binary or numeric options. The integrated configuration referenced throughout this paper is one specific point in that menu space. The regime-indexed configuration lookup reported alongside the results recommends alternative settings under operating conditions where the integrated point does not transfer.

#figure(
  image("/typst/figures/colorspace_projection.pdf", width: 95%),
  caption: [
    The nine channel projections of the three supported colorspaces.
  ],
) <fig:colorspace_projection>

== Frame decode and colorspace conversion

The first stage decodes each Blue-Green-Red (BGR) frame from the input video. The second projects that frame into a working colorspace. The pipeline supports four options, namely the International Commission on Illumination (CIE) 1976 $L^* a^* b^*$ colorspace (CIELAB), Red-Green-Blue (RGB), Hue-Saturation-Value (HSV), and 8-bit grayscale. The integrated configuration uses CIELAB because for the resin-and-fabric combinations in the eleven samples evaluated here, wetting primarily darkens the lightness channel $L^*$ rather than shifting chrominance, so the single-channel $L^*$ projection used in the difference computation captures most of the signal. The choice is regime-dependent rather than universal. A pigmented resin or a colored fabric could shift the balance of evidence into chrominance and make RGB or HSV preferable, which is why the alternatives remain selectable. The eleven-sample mean and per-sample breakdown of the colorspace effect are reported in~@tab:ablation. We denote the converted frame at index $t$ by $F_t$.



== Region of interest

The region of interest is a binary mask $R$ that is one inside the laminate and zero elsewhere; all subsequent stages operate only on pixels where $R = 1$. The pipeline supports three forms for constructing $R$, mutually exclusive, selected by which configuration parameter is supplied. The rectangular form takes four fractional margins, one per edge, each clamped to the closed interval from zero to forty-nine hundredths of the corresponding side, and builds the rectangle bounded by those margins. A single configuration parameter sets all four margins symmetrically, with per-edge overrides available. The rectangular form is appropriate for laminates that fit a rectangular bounding box with no internal fixtures, which describes every sample in the labeling subset described below. The imported-PNG form reads a single-channel grayscale image at the source video's resolution from a path supplied via the CLI, thresholds it at 127, and uses the resulting binary image directly as $R$. The imported-PNG form is appropriate when the laminate is non-rectangular (for instance a curved part boundary) or when fixtures, sensors, or labels inside the bag must be excluded from the difference computation but lie inside any axis-aligned rectangle that contains the laminate. The imported-COCO form reads a Common Objects in Context (COCO) JSON file from the same CLI flag, locates the image entry whose file name matches the input video, and rasterizes every polygon annotation on that image into $R$. The imported-COCO form is appropriate when the laminate boundary is already available as a polygon in an existing labeling project.

#figure(
  image("/typst/figures/roi_crop.pdf", width: 100%),
  caption: [
    Three forms of the ROI mask $R$ on the input frame.  ],
) <fig:roi_crop>

== Camera-shift detection and registration

A bumped tripod, a thermal expansion of the rig, or a hand brushing the camera in mid-run shifts the projected position of every laminate pixel by some vector that has nothing to do with wetting. The reference frame stops aligning with the live frame, the difference field lights up along high-contrast laminate edges, and the threshold catches that as wet. The pipeline detects the shift and corrects it before the difference is computed. For each frame, a single phase-correlation step @KuglinHines1975PhaseCorrelation on a fixed-resolution downsample of the working channel of the previous and current frames returns a translation $(d_x, d_y)$ and a confidence score. When either the per-frame magnitude $sqrt(d_x^2 + d_y^2)$ or the cumulative drift across a five-frame rolling window exceeds the configured threshold, an iterative Enhanced Correlation Coefficient refinement @EvangelidisPsarakis2008ECC fits a translation or affine warp $W_t$ on a static-edge mask of the current ROI, and the live frame is warped through $W_t$ into the reference coordinate system before all downstream stages. The static-edge mask is recomputed at each shift event so the registration is driven by mold and frame edges that do not move with the wet front rather than by the wet region itself. The peak map is either discarded, so the post-warp pixels do not accumulate against pre-warp brightness, or warped through $W_t$ into the new coordinate system, so the running maximum is preserved at the cost of carrying any residual registration error into the peak.

#figure(
  image("/typst/figures/camera_shift_pair.pdf", width: 100%),
  caption: [
    Pipeline delta field across the bumped-tripod event in input_1
    (frame 10 reference, frame 75 current, 4.7 px shift). The corrected panel preserves the real wetting while
    the registration warp removes the edge artifacts.
  ],
) <fig:camera_shift_pair>

== Peak-brightness reference

The pipeline accumulates a running maximum of the working channel across all frames seen so far. When the pre-delta blur is enabled the maximum is updated from the blurred working channel rather than the raw channel so the peak map and the delta input share the same bandwidth, which prevents a one-pixel halo of speckle from shifting the peak above the eventual delta input and creating a phantom darkening signal. The motivation for the running maximum is that, on infusions where the dry preform is the brightest state the fabric reaches in the working channel, a per-pixel maximum tracks lighting drift caused by lamp warm-up and bag deformation rather than treating that drift as wetting evidence. ~@fig:peak illustrates the regime on one tracked pixel of the canonical input clip. The raw $L^*$ value drifts upward as the lamps stabilize, the running peak $P$ tracks that drift, and the front arrival separates as a sharp drop of more than thirty units below the peak. A fixed reference at frame zero would have included the post-warm-up lift in its difference. The assumption that wetting is the only event that lowers the peak does not hold universally. A specular flash that drives the peak above its long-term value, or a fabric whose wet state is brighter than its dry state for some channel, can produce a peak map that misrepresents subsequent wetting evidence. The IoU effect of disabling the primitive on the eleven-sample subset is reported in~@tab:ablation.

#figure(
  image("/typst/figures/peak_reference.pdf", width: 95%),
  caption: [
    Four pixels of input_1 tracked across all 706 frames. Each pixel drifts upward during lamp warm-up,
    the running peak absorbs that drift, and the front arrival
    collapses $L^*$ sharply below $P$.
  ],
) <fig:peak>



== Reference selection

The reference image $G_t$ used for the difference computation has five modes. The first-frame mode pins $G_t = F_0$ for every $t$. The running mode updates an exponential moving average $G_t = (1 - alpha) G_(t-1) + alpha F_t$ with $alpha = 0.05$. The previous-frame mode uses a fixed-offset history $G_t = F_(t - k)$, with the buffer bootstrapped from $F_0$ until $t > k$. The absolute mode pins $G_t$ to a user-specified absolute frame index. The dynamic mode adapts the lag online from a square-root-area growth model fit to the early frames. The integrated configuration uses the first-frame mode, which, combined with the peak map above, gives a piecewise-static reference whose only adaptation is the peak update.

Reference selection is computed but does not influence the delta when both peak-reference and darken-only are enabled, as in the integrated configuration. The peak map replaces $G_t$ in the darken-only delta formula below, so the menu choice above becomes load-bearing only when peak-reference or darken-only is disabled.

The dynamic mode warrants a closer look because the integrated configuration does not use it for the headline numbers and disabling it is one of the rows in~@tab:ablation. For the first ten frames after detection bootstrapping (the calibration window), the pipeline records the wet area $a_t$ inside the region of interest and the elapsed time $tau_t$ at frame $t$. It then estimates a growth-rate factor $kappa$ as the median of $a_t / tau_t^(1.5)$ across the calibration frames. The three-halves exponent is motivated by the area-growth law for radial Darcy infusion, where wetted area scales approximately as time to the three-halves power once flow is well established; the assumption holds approximately for the early-fill regime of the eleven samples in the labeling subset and breaks for race-tracking-dominated runs, which is why the dynamic-calibration window is sensitive to anomalous early frames as discussed in the failure-modes analysis. Given $kappa$ and a target wet fraction $rho$ of the region-of-interest area $|R|$ (held at $0.2$ in the integrated configuration), the dynamic lag $Delta tau_t$ in seconds that places the reference at the moment when the wet area was a fraction $rho$ of the current area is

$ Delta tau_t = lambda dot.c [ ( (rho |R|) / kappa + sqrt(tau_t) )^2 - tau_t ], $ <eq:dynlag>

This is clipped to be non-negative and scaled by a user lag factor $lambda$. The reference frame is then read from a small cache at the integer index whose elapsed time is closest to $tau_t - Delta tau_t$, falling back to the first frame whenever the calibration has not yet produced a finite $kappa$. A linear-mode override replaces the sqrt-area growth fit with a parameterized linear lag schedule, which steps the reference frame back at a constant rate independent of the calibration estimate. The override is intended for diagnostic comparisons where the sqrt-area assumption is suspect, and a per-frame log of the chosen lag is written to a log for post-hoc inspection.

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

#figure(
  image("/typst/figures/darken_only_compare.pdf", width: 100%),
  caption: [
    Input frame from input_2 (frame 15 against frame 0 reference)
    under two delta computations. The naive Euclidean delta lights up
    the left side.
    The darken-only delta clips that brightening to zero.
  ],
) <fig:darken_only>

where the reference $G_t$ is the running peak map (per-pixel maximum of the working channel across all prior frames) when peak-reference is enabled, and the working-channel slice of the frame chosen by the reference-selection mode otherwise. The clip to non-negative values discards every pixel that becomes brighter than the reference, which removes specular flashes from the vacuum bag and from condensation, neither of which are wetting events. A full-color mode replaces the difference with the channel-weighted Euclidean distance across all channels and is intended for HSV and RGB workflows where chrominance shifts are diagnostic. The per-channel weights $w_0, w_1, w_2$ are exposed via the CLI and equal $1, 1, 1$ in the integrated configuration; non-uniform weights are appropriate when one channel of the working colorspace carries the wetting signal more strongly than the others. ~@fig:darken_only shows the effect of the clip on a single frame; ~@tab:ablation reports the IoU cost of removing it across the eleven-sample subset.

== Post-delta blur

#figure(
  image("/typst/figures/pre_post_blur.pdf", width: 95%),
  caption: [
    Pre- and post-delta blur on the input_1 bump event.  
    Top row, working channel $L^*$ after pre-delta blur.
  ],
) <fig:pre_post_blur>
The delta field is smoothed by the post-delta blur stage, which is a single function module and shared with the optional pre-delta blur of stage four. The user selects a kernel kind from {flat, gaussian, triangle, median, bilateral, none} together with a kernel size, written as a single specification of the form KIND[:SIZE]. The integrated configuration uses a separable Gaussian of size $k_b = 9$ pixels (forced odd at runtime). Gaussian is appropriate when speckle is approximately white noise around the underlying response field; flat and triangle are exposed because some camera-and-bag combinations produce structured noise that the corresponding box or tent filter handles with less bias.



== Normalization and threshold

The smoothed field is rescaled to the byte range using a min-max linear rescaling, producing a $tilde(D)_t$ that fits in eight bits and feeds both the threshold-selection stage and the heatmap renderer. Six thresholding modes are exposed through a single specification of the form KIND[:ARGS]. The four global modes share a single offset $delta_tau$ that is added before binarization. Otsu's between-class variance method @Otsu1979Threshold recovers an automatic threshold $tau_"otsu"$ over the full $tilde(D)_t$ histogram and adds the offset, which is the integrated configuration's choice and works on infusions whose histograms are roughly bimodal. The Triangle method recovers $tau_"tri"$ by the geometric construction on the histogram @Zack1977Triangle, which is appropriate when one class dominates the histogram (typically early fill, where most pixels are dry). Manual mode uses a user-supplied absolute byte value $tau_"man"$ plus $delta_tau$; percentile mode sorts the ROI pixels and recovers the $p$-th percentile by linear interpolation. The two adaptive modes, adaptive-mean and adaptive-gaussian, compute a per-pixel threshold from a $b times b$ neighborhood mean (or Gaussian-weighted mean) minus a constant $C$, and bypass $delta_tau$ because $C$ already serves the same role. Adaptive thresholding is appropriate when the delta retains a low-frequency intensity gradient that the reference stage did not fully cancel, for instance under uneven side-lighting or vignette artifacts. The integrated configuration uses Otsu with $delta_tau = -30$, which biases the global threshold toward the wet class on infusions where the partially-wetted halo around the front sits below the bimodal split that Otsu finds. The bias is appropriate for the early-fill regime of the eleven labeled samples and not appropriate for every infusion. The Triangle, manual, percentile, and adaptive variants remain selectable for infusions whose histograms or intensity gradients fit those modes better; the integrated configuration does not exercise them, and the headline numbers reported below are reported under Otsu plus offset.

#figure(
  image("/typst/figures/threshold_modes.pdf", width: 95%),
  caption: [
    The six threshold modes applied to the same normalized response
    field from input_1 frame 352. Top strip is the
    histogram of response inside the ROI with the Otsu and
    Triangle threshold values marked.
  ],
) <fig:threshold_modes>

== Morphological cleanup

Stages ten and eleven clean the thresholded mask. Stage ten passes the binary mask through morphological closing with an elliptical structuring element of size $k_m$ (default thirteen pixels). Closing welds neighbouring wet patches into a single front and fills small gaps where transient bag wrinkles muted the local response. The kernel size sets the spatial scale at which gaps are considered noise rather than real disconnections in the front; on the resolutions in the eleven labeled samples a $13 times 13$ ellipse is large enough to bridge bag-wrinkle gaps and small enough to leave genuinely separate wet regions separate. Stage eleven passes the closed mask through morphological opening with the same kernel and shape, removing specks below the kernel size that survived the closing pass. A connected-components labelling pass then discards any region whose pixel area is below $a_"min"$ pixels (default $400$), which removes the small islands that the morphology kernels are too small to suppress. ~@fig:cleanup shows the same frame at every step of stages nine through eleven.

#figure(
  image("/typst/figures/mask_cleanup.pdf", width: 100%),
  caption: [
    Mask cleanup on input_1 frame 200. Normalized response (1),
    threshold (2), closing (3), opening (4), and area filter (5).
  ],
) <fig:cleanup>

The kernel parities are forced odd at runtime so that anchor handling is symmetric. The structuring-element shape is elliptical by default with optional rectangular and cross-shaped alternatives, exposed because the front shape changes with infusion geometry.

== Temporal locking

Stage twelve imposes hysteresis along the time axis. Each pixel keeps a small per-pixel counter that increments while the cleaned mask reports the pixel as wet and resets to zero on any frame where the cleaned mask says dry. Once the counter reaches the threshold $n_"lock"$ frames, a sticky locked-pixel map is set true at that location and never resets. The output of the stage is the elementwise OR of the cleaned mask with the sticky locked map, so a pixel that has ever been wet for $n_"lock"$ consecutive frames stays wet for the remainder of the run.

The integrated configuration uses $n_"lock" = 3$, which holds a positive detection in the mask for three subsequent frames. The latched-after-three-frames behavior is appropriate for infusions whose true pauses are shorter than the lock window and inappropriate for infusions whose pauses are longer, where the latch produces phantom regions that look like artifacts as discussed in the failure-modes analysis. Setting the lock window to zero disables the stage altogether. The "no temporal lock" row of Table~@tab:ablation reports the per-sample effect, which depends on whether the live infusion contains pauses on the order of three frames or longer.

== Heatmap, overlay, and contour export

The last two display stages exist for inspection and label export. The heatmap renderer maps the normalized delta $tilde(D)_t$ through the Turbo colormap to produce a three-channel pseudocolor image. The overlay renderer extracts the boundary of the locked mask via a five-by-five rectangular morphological gradient and paints those boundary pixels red on a copy of the original BGR frame. Neither renderer is part of the detection logic; they expose, in human-readable form, the field on which the threshold acted and the boundary that the locked mask encloses.

For machine-readable export, the contour-extraction stage emits Common Objects in Context (COCO) and YOLO-format polygons for every connected component of the locked mask. Polygon segmentation is computed with the Suzuki-Abe topological border-following algorithm @Suzuki1985Border, optionally simplified by the Douglas-Peucker algorithm @DouglasPeucker1973 with tolerance $epsilon$, and optionally densified to a maximum edge length so that downstream rasterization preserves curvature. The annotation-sampling utility selects which frames receive labels using one of three modes, namely all-frame, evenly-spaced count, or fixed-stride, with deduplication of consecutive ties so that the integer-truncated linear-spacing exactly reproduces the standard reference behaviour.

#figure(
  image("/typst/figures/heatmap_overlay_contour.pdf", width: 100%),
  caption: [
    The render outputs of the pipeline on input_1 frame 352: raw
    input, $tilde(D)_t$ heatmap, locked-mask overlay, and COCO
    contour export.
  ],
) <fig:heatmap_overlay_contour>

