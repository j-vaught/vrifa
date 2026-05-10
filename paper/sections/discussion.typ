= Discussion

The integrated pipeline is a working method on the eleven samples evaluated in this paper, not a universal segmenter for arbitrary VARTM video. The failure modes described below are concrete, mechanism-level, and traceable to the specific component whose assumption the input violates.

== Failure Modes

The temporal-locking window is the primary diagnostic when results appear noisy on long pauses. The integrated configuration sets the window to three frames, which holds a positive detection in the mask for three subsequent frames so transient single-frame dropouts do not flicker the boundary. On infusions whose true wet-front pauses exceed three frames, transient detections from earlier in the run persist past the moment the front recedes and the mask grows phantom regions that look like artifacts. The mechanism is not a defect. It is the locking semantics applied to a sequence whose pauses violate the assumption baked into a fixed three-frame window. The operational consequence is that the window is a per-mold parameter, not a constant suitable for every infusion, and the failure is visible only on long-pause sequences.

#figure(
  // image("/typst/figures/failure_lock_phantom.pdf", width: 95%),
  rect(width: 100%, height: 1.8in, stroke: 0.5pt, inset: 8pt)[
    _Long-pause lock-artifact placeholder._ Three-frame sequence
    on an infusion whose true wet front pauses for ten frames mid-
    run with the lock window left at $n_"lock" = 3$. Frame at
    $t = 20$ (mid-pause) shows phantom regions held by the lock
    from earlier detections. Frame at $t = 30$ (after the pause
    resolves) shows the true front receded but the lock-held mask
    still showing the phantom. Frame at $t = 40$ shows that the
    phantom region persists, because the locked-pixel map is sticky.
    A strip plot below shows the per-pixel detection state over
    time for one phantom-region pixel with the lock latch
    annotated.
  ],
  caption: [
    Phantom regions persist past the moment the front recedes when
    the lock window is shorter than the true pause duration.
  ],
) <fig:failure_lock_phantom>

The dynamic-reference calibration window is the second sensitive surface. The integrated configuration uses ten calibration frames to fit a square-root-of-area growth model that subsequently lags the reference frame behind the live frame. The assumption baked into that fit is that early fill is representative of the growth law. If the first ten processed frames contain race-tracking, an air pocket, or some other anomaly, the fit picks up a poor reference factor and every downstream frame inherits that error. The remedy is to choose a calibration window that excludes the anomaly or to fall back to one of the static reference modes, both of which are configuration choices rather than numerical defects.

#figure(
  // image("/typst/figures/failure_calibration.pdf", width: 95%),
  rect(width: 100%, height: 1.6in, stroke: 0.5pt, inset: 8pt)[
    _Dynamic-reference calibration anomaly placeholder._ Two
    side-by-side trajectories of the dynamic-reference fit. Left:
    clean infusion where early fill obeys the sqrt-area growth
    law, $kappa$ converges, downstream lag is well-behaved, mask
    aligns with the true front. Right: race-tracking-dominated
    infusion where the first ten calibration frames violate the
    growth law, $kappa$ is fit to a poor model, the downstream lag
    is wrong, and the resulting mask is offset from the true front.
    Beneath both trajectories, side-by-side overlays at the same
    fill percentage show the offset visually. The mismatched panel
    is bordered in garnet.
  ],
  caption: [
    The dynamic-reference fit is sensitive to the early-fill
    regime. Race-tracking inside the calibration window distorts
    $kappa$ and propagates an offset reference frame through the
    rest of the run.
  ],
) <fig:failure_calibration>

The third failure mode is the inherent ceiling of any tuned classical-CV pipeline. The integrated configuration does not generalize across substantially different fabric types, lighting setups, or vacuum-bag textures without re-tuning. The mode menus and scalar values held fixed in Section~3 (Tables~@tab:colorspace_modes, @tab:threshold_modes, @tab:scalars) assume a transparent vacuum bag, top-down LED lighting, and a dark mold background, because those are the conditions of the eleven samples in the labeling subset. Off-distribution conditions, such as a heavily textured bag, side lighting, or a bright mold surface, can break the Otsu threshold or push the response distribution outside the range the threshold offset was tuned for. The pipeline exposes a percentile-threshold mode, a manual contrast threshold, and an RGB colorspace switch precisely because no single configuration covers every regime, and a per-mold reconfiguration is the appropriate response. The component-removal ablation in Section~5 reports on conditions where the integrated configuration as evaluated holds.

#figure(
  // image("/typst/figures/failure_offdistribution.pdf", width: 95%),
  rect(width: 100%, height: 1.8in, stroke: 0.5pt, inset: 8pt)[
    _Off-distribution failure placeholder._ Three-column panel.
    Left: heavily-textured silicone vacuum bag where bag patterns
    create persistent dark features that the threshold catches as
    wet. Center: side-lit laminate with a strong illumination
    gradient across the field of view that the reference stage
    does not fully cancel, leaving a low-frequency intensity
    gradient in the response. Right: bright reflective mold surface
    that drives Otsu's bimodal split in the wrong direction. For
    each column, the top row is the raw input and the bottom row is
    the integrated-configuration mask, showing visibly broken
    segmentation. The integrated configuration as evaluated does
    not address any of these regimes; each requires a per-mold
    reconfiguration via the percentile-threshold mode, manual
    contrast threshold, or adaptive-mean threshold from
    Table~@tab:threshold_modes.
  ],
  caption: [
    Three off-distribution conditions where the integrated
    configuration as evaluated breaks. Each requires per-mold
    reconfiguration of the threshold or colorspace mode.
  ],
) <fig:failure_offdistribution>

== Tradeoffs

The integrated configuration is built on a deliberate set of tradeoffs that a reader should weigh alongside the benchmark numbers.

We trade an end-to-end accuracy ceiling for component-level explainability. A learned segmenter trained on a sufficiently large in-domain corpus would plausibly exceed the boundary $F_1$ reported in this paper, but no such corpus exists in the public VARTM literature and a research group that wants to characterize a single infusion run next week cannot collect one in time. The component-removal ablation in Section~5 reports the IoU consequence of disabling each named primitive on the eleven-sample subset, which is the property the integrated configuration is designed to expose. A monolithic learned segmenter would conflate those deltas, so a reader interested only in the headline number could replace this pipeline with a sufficiently large convolutional model and lose the per-component evidence. The empirical contribution of this paper is the per-component evidence, not the headline.

We trade automated hyperparameter selection for component-level legibility. The pipeline has no learned end-to-end loss and no automated tuner sweeping fifty parameters in the background. Every parameter that controls the result is named in Section~3 and either documented in a per-subsection mode-menu table or pinned in Table~@tab:scalars. A reader who wants to understand why a result looks the way it does on a particular infusion can answer that question by reading the configuration values rather than by introspecting weights. The cost is that an unfamiliar reader who picks an inappropriate value for a per-mold parameter can ruin a run, which is the same risk the temporal-locking failure mode above describes.

The pipeline runs on a CPU on the host described in Section~6 and on a GPU when one is present. Neither implementation is part of the empirical claim; both are reported so that a reader can locate the throughput available on a typical research bench and scale that estimate to longer runs without re-deriving it.

== Scope and Limits

Several setups fall outside the regime evaluated in this paper. Opaque vacuum bags break the entire pipeline, because the front is no longer visually observable from above and there is nothing for any image-domain method to detect. Multi-resin systems where dye contrast is intentionally absent fall in the same category, since the response image is ultimately a contrast measurement. Very fast infusions, where the front passes through a region of interest in well under a second, push the temporal-locking and morphology kernels past the regime they were tuned for, and the resulting masks lag the true front. Camera motion during infusion is handled by the optional registration stage, which warps the live frame back into the reference coordinate system whenever per-frame or rolling-window drift exceeds the configured threshold. The registration absorbs the bumped-tripod and thermal-creep regimes that otherwise break the comparison between live and reference frame. Setups where the camera moves continuously and at high amplitude, such as a handheld phone walked around the rig, exceed the small-shift assumption the affine warp is fit under and remain out of scope. Lens-distortion is not corrected in the integrated pipeline; for wide-angle or fisheye-mounted cameras, a one-time Scaramuzza-style calibration @Scaramuzza2006Toolbox would be a complementary preprocessing step, applied to every frame before the colorspace conversion, that the integrated pipeline does not currently provide.

These are not weaknesses to be patched. They are the boundary of the application domain. A practitioner facing any of them requires a different tool, and stating that boundary explicitly is more useful than overstating the method's operating envelope.

== Implications for the Field

The integrated pipeline is intended to be useful to two audiences. A practitioner running a VARTM rig obtains a flow-front segmentation method whose effect on each captured infusion is interpretable component by component, so a per-mold problem (specular bag, off-distribution lighting, fast-fill geometry) can be traced to a specific stage rather than buried in an end-to-end loss. A researcher building permeability-inversion or process-monitoring studies that depend on per-pixel front geometry obtains a method whose individual components are characterized against a labeled benchmark, so substituting any one of them for an alternative method can be evaluated against a fixed comparison point rather than against an opaque baseline.
