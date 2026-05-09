= Runtime Characterization

The runtime story is captured by a three-tier benchmark introduced in the Implementation section. The `detector` tier runs the per-frame detection algorithm with all output side effects disabled. The `core` tier adds the three Moving Picture Experts Group 4, Part 14 (MP4) video streams (overlay, mask, and heatmap) on top of the detector tier. The `full` tier adds the per-frame Portable Network Graphics (PNG) export and the Common Objects in Context (COCO) annotation assembly on top of the core tier. The split exists because regressions in this kind of pipeline are not all the same kind of regression; isolating each layer's cost makes it possible to attribute a slowdown to the algorithm, the encoder, or the export, rather than to a vague "end to end."

Three back-to-back invocations of the `cargo bench -p vrifa-cli --bench perf_gate` target produced the medians reported in Figure~@fig:runtime. On the canonical reference video `input_1` (706 frames at $1920 times 1080$, ROI fraction 0.49), the medians are $23.6$ seconds for the detector tier, $27.2$ seconds for the core tier, and $70.3$ seconds for the full tier. On the time-lapse reference video `input_2` (100 frames at $1920 times 1088$, ROI fraction 1.0), the medians are $3.6$, $4.4$, and $8.4$ seconds respectively. Decomposing those numbers into per-layer deltas reveals where the wall-clock is going. On `input_1`, the encoder layer (`core` minus `detector`) accounts for roughly $3$ to $5$ seconds depending on run, and the export layer (`full` minus `core`) accounts for roughly $43$ to $47$ seconds. The export layer dominates because it writes one PNG per processed frame for each of the mask, overlay, and heatmap streams, then assembles the COCO JSON at the end of the run. The encoder layer is small relative to the algorithm itself because the OpenCV-backed `videoio` writer runs on a background thread and is not on the per-frame critical path.

#figure(
  // image("../typst/figures/runtime.pdf"),
  rect(width: 100%, height: 1.8in, stroke: 0.5pt, inset: 8pt)[
    _Runtime breakdown placeholder._ Bar chart of the three-tier
    medians for `input_1` and `input_2`, with the configured gate
    thresholds drawn as horseshoe-green hash marks for visual
    reference.
  ],
  caption: [
    Three-tier perf-gate measurements. Bars show wall-clock medians
    across three back-to-back runs; hash marks indicate the
    configured gate thresholds at $32 / 35 / 90$ seconds for
    `input_1` and $5 / 6 / 11$ seconds for `input_2`. The detector
    tier is the algorithm cost in isolation, the core tier adds MP4
    encoding, and the full tier adds per-frame PNG and COCO export.
  ],
) <fig:runtime>

The benchmark gates encode a hard pass or fail contract. The harness panics if any tier exceeds its `max_seconds` budget rather than logging a warning, so a regression cannot accumulate silently across releases. Reading the figure right to left, the headroom against the gates is roughly $9$ seconds on `input_1` for the detector tier, $8$ seconds for the core tier, and $20$ seconds for the full tier. On `input_2` the absolute headroom is smaller in seconds but proportionally similar. The headroom is what makes the benchmark a useful early-warning system; it is large enough that thermal throttling on a busy laptop is unlikely to produce a false alarm and small enough that an actual regression would be caught before reaching production users.

The throughput numbers translate to roughly $30$ frames per second on `input_1` for the algorithm itself, falling to roughly $10$ frames per second once full export is enabled. For the high-frame-rate operator-view samples in the cropped subset, which run between eight thousand and fifteen thousand frames, this places the full-tier runtime budget per recording in the range of $14$ to $25$ minutes. Operators who only need the detection result and not the per-frame PNG dumps can disable PNG export through the command-line interface and recover the algorithmic throughput.
