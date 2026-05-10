= Method

// Three-layer Method section. Layer 1 gives plain-English
// intuition with a pipeline diagram. Layer 2 walks each stage
// in prose with a small number of load-bearing equations and
// stage-specific diagrams. Layer 3 maps the stages onto the
// Rust workspace.

== Pipeline overview

The detection algorithm treats each video frame as a comparison against a quiescent reference image of the dry preform. Where resin has wetted the fabric, the local appearance darkens, and the absolute change is largest near the advancing flow front. The pipeline first checks for camera motion against the previous frame and, when motion crosses a configured threshold, registers the live frame back into the reference coordinate system before any difference is computed. The registered frame is then optionally smoothed with a pre-delta blur whose output feeds both the running peak map and the difference computation, so the reference and the comparison see the same bandwidth-limited input. The pipeline then computes the per-pixel difference inside a rectangular region of interest, clips it to admit only darkening so that specular highlights from the vacuum bag are rejected, normalizes the resulting field, thresholds it into a binary candidate mask, cleans the mask with morphological closing and opening, removes small connected components, and finally locks pixels whose wet label has persisted for several consecutive frames. The locked mask is the canonical output, the optional heatmap renders the underlying delta field for visual diagnosis, and the optional overlay draws the mask boundary onto the original Vacuum-Assisted Resin Transfer Molding (VARTM) frame for human review. Figure~@fig:pipeline shows the fourteen stages in order.

#figure(
  image("/typst/figures/pipeline.pdf", width: 95%),
  caption: [
    Fourteen-stage VRIFA pipeline. Each frame flows from decode
    through an optional camera-shift registration and pre-delta
    blur, then through an appearance-difference comparison against
    a chosen reference, into a thresholded and morphologically
    cleaned mask, and finally through a temporal lock that
    stabilizes the boundary across frames. The same mask drives
    the binary, heatmap, and overlay renderers.
  ],
) <fig:pipeline>

The pipeline is deliberately classical, which is what makes it suitable as a reference implementation for downstream learning systems. Every stage is an explicit function of one frame, the chosen reference image, and a small set of scalar parameters, so the output is reproducible bit-for-bit at the stage boundary and the algorithm can be audited end-to-end without recourse to a black-box model.

== Frame decode and colorspace conversion

The first stage decodes each Blue-Green-Red (BGR) frame from the input video. The second converts that frame into a working colorspace. VRIFA exposes four options, namely the International Commission on Illumination (CIE) 1976 $L^* a^* b^*$ colorspace (CIELAB), Red-Green-Blue (RGB), Hue-Saturation-Value (HSV), and 8-bit grayscale, all routed through standard color converters. CIELAB is the default because resin wetting darkens the lightness channel $L^*$ much more than it shifts chrominance, which makes the single-channel $L^*$ projection used in the difference computation both sufficient and robust. We denote the converted frame at index $t$ by $F_t in bb(R)^(H times W times C)$, where $C$ is the channel count of the chosen colorspace.

== Region of interest

A rectangular region of interest excludes the part frame, manifold flange, and bag wrinkles outside the laminate. The user supplies four fractional margins, one per edge, each clamped to the closed interval from zero to forty-nine hundredths of the corresponding side, and the algorithm builds a binary mask $R$ that is one inside the resulting rectangle and zero elsewhere. A single configuration parameter sets all four margins symmetrically, with per-edge overrides available. All subsequent stages operate only on pixels where $R = 1$.

== Camera-shift detection and registration

A bumped tripod, a thermal expansion of the rig, or a hand brushing the camera in mid-run shifts the projected position of every laminate pixel by some vector that has nothing to do with wetting. The reference frame stops aligning with the live frame, the difference field lights up along high-contrast laminate edges, and the threshold catches that as wet. VRIFA detects the shift and corrects it before the difference is computed. For each frame, a single phase-correlation step on a fixed-resolution downsample of the working channel of the previous and current frames returns a translation $(d_x, d_y)$ and a confidence score. When either the per-frame magnitude $sqrt(d_x^2 + d_y^2)$ or the cumulative drift since the last reset exceeds an operator-set threshold, an iterative-coplanar-correlation refinement fits a translation or affine warp $W_t$ on a static-edge mask of the current ROI, and the live frame is warped through $W_t$ into the reference coordinate system before all downstream stages. The static-edge mask is recomputed at each shift event so the registration is driven by mold and frame edges that do not move with the wet front rather than by the wet region itself. The peak map is reset on registration so the post-warp pixels do not accumulate against pre-warp brightness.

== Peak-brightness reference

VRIFA accumulates a running maximum of the working channel across all frames seen so far. When the pre-delta blur is enabled the maximum is updated from the blurred working channel rather than the raw channel so the peak map and the delta input share the same bandwidth, which prevents a one-pixel halo of speckle from shifting the peak above the eventual delta input and creating a phantom darkening signal. The motivation for the running maximum is that the dry preform is, by definition, the brightest the fabric will ever appear in the working channel. Treating that running maximum as the per-pixel reference instead of a single early frame absorbs the slow lighting drift caused by lamp warm-up and bag deformation, leaving the difference field free to respond to wetting alone. Figure~@fig:peak shows the effect at one tracked pixel of the canonical input clip. The raw $L^*$ value drifts upward as the lamps stabilize, the running peak $P$ tracks that drift monotonically, and the front arrival cleanly separates as a sharp drop of more than thirty units below the peak. A fixed reference would have started at the value reached on frame zero and missed the post-warm-up lift entirely.

#figure(
  image("/typst/figures/peak_reference.pdf", width: 95%),
  caption: [
    One pixel of the first canonical input clip tracked across all
    706 frames. The raw CIELAB lightness $L^*$ drifts up by roughly
    twenty units before the front arrives, and the running peak $P$
    absorbs that drift. When the resin reaches the pixel, $L^*$
    collapses by more than thirty units below the peak, which is
    the signal the difference stage detects. A single fixed
    reference at frame zero would have treated the warm-up
    brightening as a (negative) wetting event.
  ],
) <fig:peak>

The peak map can be disabled through a single configuration parameter for users who want a strictly fixed reference, or who are diagnosing a lighting failure where the assumption of monotone brightness no longer holds.

== Reference selection

The reference image $G_t$ used for the difference computation has five modes. The first-frame mode pins $G_t = F_0$ for every $t$. The running mode updates an exponential moving average $G_t = (1 - alpha) G_(t-1) + alpha F_t$ with default $alpha = 0.05$. The previous-frame mode uses a fixed-offset history $G_t = F_(t - k)$, with the buffer bootstrapped from $F_0$ until $t > k$. The absolute mode pins $G_t$ to a user-specified absolute frame index. The dynamic mode adapts the lag online from a square-root-area growth model fit to the early frames. The default selection is the first-frame mode, which, combined with the peak map above, gives a piecewise-static reference whose only adaptation is the peak update.

The dynamic mode warrants a closer look because it is what enables comparable behavior across runs of different fill speeds. For the first $N_"calib"$ frames after detection bootstrapping, VRIFA records the wet area $a_t$ inside the region of interest and the elapsed time $tau_t = (t-1)/f$ in seconds, where $f$ is the video frame rate. It then estimates a growth-rate factor $kappa$ as the median of $a_t / tau_t^(3/2)$ across the calibration frames; the three-halves exponent comes from the area-growth law observed for radial Darcy infusion, where wetted area is approximately proportional to time at the three-halves power once flow is well established. Given $kappa$ and a target wet fraction $rho$ of the region-of-interest area $|R|$ (default $0.2$), the dynamic lag $Delta tau_t$ in seconds that places the reference at the moment when the wet area was a fraction $rho$ of the current area is

$ Delta tau_t = lambda dot.c [ ( (rho |R|) / kappa + sqrt(tau_t) )^2 - tau_t ], $ <eq:dynlag>

clipped to be non-negative and scaled by a user lag factor $lambda$ (default $1.0$). The reference frame is then read from a small cache at the integer index closest to $t - Delta tau_t f$, falling back to the first frame whenever the calibration has not yet produced a finite $kappa$. A linear-mode override is available for diagnostic comparisons.

== Delta computation

Stage eight produces the per-pixel scalar field $D_t$ that drives all downstream decisions. The default darken-only mode operates on the working (first) channel and records how much darker the frame is than its reference,

$ D_t (y, x) = R(y, x) dot.c max(0, w_0 dot.c (G_t^star (y, x) - F_t (y, x, 0))), $ <eq:delta_darken>

where $G_t^star$ equals the peak map $P_t$ when the peak-reference mode is enabled and otherwise equals the channel-zero slice of $G_t$. The clip to non-negative values discards every pixel that becomes brighter than the reference, which removes specular flashes from the silicone vacuum bag and from condensation, neither of which are wetting events. A full-color mode replaces the difference with the channel-weighted Euclidean distance across all $C$ channels, exposed through a single configuration parameter and intended for HSV and RGB workflows where chrominance shifts are diagnostic. Figure~@fig:darken_only shows the effect of the clip.

#figure(
  image("/typst/figures/darken_only_compare.pdf", width: 100%),
  caption: [
    Same input frame, two delta computations. The naive Euclidean
    delta (centre) lights up bright on the vacuum-bag specular
    highlight to the upper right of the laminate. The darken-only
    delta (right) discards the highlight and isolates the front.
    The early-fill front in the lower part of the panel is visible
    in both, though more cleanly in darken-only.
  ],
) <fig:darken_only>

== Smoothing, normalization, and threshold

The delta field is smoothed with a separable Gaussian kernel of size $k_b times k_b$ pixels (default $9$, forced odd at runtime) so that the dynamic range of the downstream normalized image is set by structure rather than by isolated speckle. The smoothed field is then rescaled to the byte range using a min-max linear rescaling, producing a $tilde(D)_t$ that fits in eight bits and feeds both the threshold-selection stage and the heatmap renderer.

Three thresholding modes share a single offset. If the user passes a manual value $tau_"man"$, the algorithm uses $tau = tau_"man" + delta_tau$. If the user passes a percentile $p in [0, 100]$, the algorithm sorts the region-of-interest pixels of $tilde(D)_t$, recovers the $p$-th percentile by linear interpolation between adjacent ranks, and adds $delta_tau$. Otherwise, Otsu's between-class variance method recovers an automatic threshold $tau_"otsu"$ over the full $tilde(D)_t$ histogram, and the offset is again added. In all three cases the final threshold is clipped to the byte range. The default offset $delta_tau = -30$ biases Otsu slightly toward the wet class to capture the partially-wetted halo that classical Otsu would otherwise miss; manual and percentile modes are reserved for parameter studies and are not used for the headline numbers reported in this paper.

== Mask cleanup

The binary mask leaving the threshold stage is rough. It contains gaps where transient bag wrinkles muted the local response, isolated specks where the threshold caught a single noisy pixel, and components small enough that they are obviously noise rather than wetted fabric. The morphological cleanup and the connected-components filter remove those artefacts in a fixed sequence. Morphological closing with an elliptical structuring element of size $k_m$ (default thirteen pixels) fills the gaps and welds neighbouring patches into one front. Morphological opening with the same kernel removes specks below the kernel size. Connected-components labelling discards any region whose pixel area is below $a_"min"$ pixels (default $400$). Figure~@fig:cleanup shows the same frame at every step of that sequence.

#figure(
  image("/typst/figures/mask_cleanup.pdf", width: 100%),
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

== Temporal locking

Stage thirteen imposes hysteresis along the time axis. Each pixel keeps a small per-pixel counter that increments while the cleaned mask reports the pixel as wet and resets to zero on any frame where the cleaned mask says dry. Once the counter reaches the threshold $n_"lock"$ frames, a sticky locked-pixel map is set true at that location and never resets. The output of the stage is the elementwise OR of the cleaned mask with the sticky locked map, so a pixel that has ever been wet for $n_"lock"$ consecutive frames stays wet for the remainder of the run. Figure~@fig:lock illustrates the bookkeeping with a twelve-frame example for $n_"lock" = 3$.

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

The default $n_"lock" = 3$ trades a small number of frames of recovery latency for boundary stability that downstream consumers need. Setting the lock window to zero disables the stage altogether for users who want raw per-frame masks.

== Heatmap, overlay, and contour export

The last two display stages exist for inspection and label export. The heatmap renderer maps the normalized delta $tilde(D)_t$ through the Turbo colormap to produce a three-channel pseudocolor image. The overlay renderer extracts the boundary of the locked mask via a five-by-five rectangular morphological gradient and paints those boundary pixels red on a copy of the original BGR frame. Neither renderer is part of the detection logic; they expose, in human-readable form, the field on which the threshold acted and the boundary that the locked mask encloses.

For machine-readable export, the contour-extraction stage emits Common Objects in Context (COCO) and YOLO-format polygons for every connected component of the locked mask. Polygon segmentation is computed with a standard contour-extraction routine, optionally simplified by the Douglas-Peucker algorithm with tolerance $epsilon$, and optionally densified to a maximum edge length so that downstream rasterization preserves curvature. The annotation-sampling utility selects which frames receive labels using one of three modes, namely all-frame, evenly-spaced count, or fixed-stride, with deduplication of consecutive ties so that the integer-truncated linear-spacing exactly reproduces the standard reference behaviour.

== Hyperparameter defaults

Table~@tab:defaults collects the parameters exposed by the configuration interface together with the values used for every result reported in this paper. Defaults were chosen during development and held fixed across all runs in the evaluation set, so the reported metrics reflect the algorithm as a user would see it on a fresh checkout rather than the outcome of a per-video search.

#figure(
  table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    stroke: (x: none, y: 0.5pt),
    table.header[*Symbol / parameter*][*Default*][*Description*],
    [colorspace], [CIELAB], [Working colorspace for the difference computation.],
    [roi-margin], [0.15], [Symmetric fractional ROI margin per edge.],
    [ref-mode], [first], [Reference selection mode.],
    [ref-running-alpha, $alpha$], [0.05], [EMA weight for the running reference.],
    [peak-reference], [true], [Use the running peak map in the working channel.],
    [darken-only], [true], [Clip the delta to non-negative wetting deltas.],
    [camera-stable], [false], [Enable phase-correlation camera-shift detection.],
    [motion-per-frame-threshold], [1.5], [Per-frame translation magnitude that triggers a registration.],
    [cumulative-motion-threshold], [3], [Accumulated drift, in pixels, that triggers a registration.],
    [motion-model], [affine], [Warp model fit on a static-edge mask when a shift is detected.],
    [peak-on-shift], [reset], [How the peak map is treated when a shift is registered.],
    [pre-delta-blur-kernel, $k_p$], [0], [Gaussian blur applied to the working frame before peak update and delta. Disabled when zero.],
    [blur-kernel, $k_b$], [9], [Gaussian blur kernel size in pixels, applied to the delta field.],
    [threshold-offset, $delta_tau$], [-30], [Offset added to Otsu/percentile/manual threshold.],
    [morph-kernel, $k_m$], [13], [Morphology structuring-element size.],
    [morph-shape], [ellipse], [Structuring-element shape.],
    [morph-close-iterations], [1], [Morphological closing iterations.],
    [morph-open-iterations], [1], [Morphological opening iterations.],
    [min-area, $a_"min"$], [400], [Minimum connected-component area, in pixels.],
    [lock-frames, $n_"lock"$], [3], [Temporal lock window, in frames.],
    [frame-step], [1], [Frame stride at decode time.],
    [dynamic-calibration-frames, $N_"calib"$], [10], [Frames used to fit the dynamic-reference factor $kappa$.],
    [dynamic-target-fraction, $rho$], [0.2], [Target wet-area fraction for dynamic-reference lag.],
    [dynamic-lag-scale, $lambda$], [1.0], [Multiplicative scale on the dynamic-mode lag.],
    [dynamic-ref-cache-size], [32], [Frames cached for the dynamic-reference reader.],
  ),
  caption: [
    Hyperparameter defaults shipped in the VRIFA configuration
    interface and used for every result in this paper. Symbols
    match the variables introduced above; the only equation that
    depends on them explicitly is the dynamic-mode lag of
    Eq.~@eq:dynlag.
  ],
) <tab:defaults>

== Implementation pointer

VRIFA is organized as a Cargo workspace with five crates. The core algorithm crate hosts the stage logic, with one Rust source file per stage. The file names track the stage names used above, namely the colorspace conversion, region-of-interest, motion-estimation, registration, warp-application, peak, reference, delta, morphology, threshold, lock, heatmap, overlay, contour-extraction, and sampling stages, with a small conversion shim that mediates between native array containers and the underlying matrix views used by the computer-vision routines. The input/output crate wraps video decode and encode through standard video-capture and video-writer interfaces, plus an asynchronous PNG writer used for per-frame artifact dumps. The command-line crate is the user-facing binary; it owns argument parsing, run orchestration, the dynamic-reference cache, and the per-frame loop that ties the fourteen stages together. The annotations crate handles COCO and YOLO export of the contours produced by the contour-extraction stage. The Python-bindings crate exposes the same stage functions as a PyO3 module, used only by internal-development tooling rather than by end-user workflows.

The Implementation section returns to the same crate layout and describes the buffer-reuse strategy, the computer-vision bindings, and the bit-exact stage-parity result that motivates VRIFA being framed as a reference implementation rather than as one detector among many.
