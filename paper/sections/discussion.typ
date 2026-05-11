= Discussion

The integrated pipeline is a working method on the eleven samples evaluated in this paper, not a universal segmenter for arbitrary VARTM video.

== Failure Modes

On every sample tested, the temporal-locking window is bimodal. The cleanest configurations either disable locking entirely or set the window to at least five frames, while a window of one to four frames is generally the worst region of the parameter space, often dropping IoU by $0.25$ relative to the bimodal modes. The most likely root cause is a startup transient in the locking heuristic, where the partial-reference accumulation during the first few frames seeds the locked-pixel map with stale evidence that subsequent frames cannot dislodge. A practitioner should therefore either disable the window or set it to at least five frames, and avoid the intermediate range entirely.

The temporal-locking window is also the primary diagnostic when results appear noisy on long pauses. The integrated configuration sets the window to three frames, which holds a positive detection in the mask for three subsequent frames so transient single-frame dropouts do not flicker the boundary. On infusions whose true wet-front pauses exceed three frames, transient detections from earlier in the run persist past the moment the front recedes and the mask grows phantom regions that look like artifacts. The window is therefore a per-mold parameter, not a constant suitable for every infusion.

The dynamic-reference calibration window is the second sensitive surface. The integrated configuration uses ten calibration frames to fit a square-root-of-area growth model that subsequently lags the reference frame behind the live frame. If the first ten processed frames contain race-tracking, an air pocket, or some other anomaly, the fit picks up a poor reference factor and every downstream frame inherits that error. The remedy is to choose a calibration window that excludes the anomaly or to fall back to one of the static reference modes.

The third failure mode is the inherent ceiling of any tuned classical-CV pipeline. The integrated configuration does not generalize across substantially different fabric types, lighting setups, or vacuum-bag textures without re-tuning. The mode menus and scalar values held fixed in the method above assume a transparent vacuum bag, top-down LED lighting, and a dark mold background, because those are the conditions of the eleven samples in the labeling subset. Off-distribution conditions, such as a heavily textured bag, side lighting, or a bright mold surface, can break the Otsu threshold or push the response distribution outside the range the threshold offset was tuned for.

#figure(
  image("/typst/figures/failure_offdistribution.pdf", width: 95%),
  caption: [
    Three off-distribution regimes that stress the integrated
    configuration. Each calls for a different sample-aware setting
    from~@tab:lookup rather than a structural fix.
  ],
) <fig:failure_offdistribution>

== Scope and Limits

Several setups fall outside the regime evaluated in this paper. Opaque vacuum bags break the entire pipeline, because the front is no longer visually observable from above and there is nothing for any image-domain method to detect. Multi-resin systems where dye contrast is intentionally absent fall in the same category, since the response image is ultimately a contrast measurement. Very fast infusions, where the front passes through a region of interest in well under a second, push the temporal-locking and morphology kernels past the regime they were tuned for, and the resulting masks lag the true front. Camera motion during infusion is handled by the optional registration stage, which warps the live frame back into the reference coordinate system whenever per-frame or rolling-window drift exceeds the configured threshold. The registration absorbs the bumped-tripod and thermal-creep regimes that otherwise break the comparison between live and reference frame. Setups where the camera moves continuously and at high amplitude, such as a handheld phone walked around the rig, exceed the small-shift assumption the affine warp is fit under and remain out of scope. Lens-distortion is not corrected in the integrated pipeline; for wide-angle or fisheye-mounted cameras, a one-time Scaramuzza-style calibration @Scaramuzza2006Toolbox would be a complementary preprocessing step, applied to every frame before the colorspace conversion, that the integrated pipeline does not currently provide.
