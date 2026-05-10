= Runtime and Systems

The integrated pipeline runs at video rate on a single CPU thread, and the same algorithm has been ported to a CUDA implementation that processes the same eleven samples at substantially higher throughput. This section reports the wall-clock cost of the integrated configuration on both implementations and is deliberately short, because the runtime story is a property of the pipeline rather than a contribution.

== Hardware

All runtimes are reported on a single host. The CPU is an Apple M-series with $N_"cores"$ performance cores at $f_"cpu"$ GHz, $N_"ram"$ GB unified memory, running macOS $V_"os"$. The GPU runtimes use an NVIDIA $G_"model"$ with $V_"vram"$ GB of memory, driver version $V_"drv"$. Decode and encode use the system FFmpeg with the libx264 and libopenh264 codecs, both invoked through the OpenCV `videoio` interface. No frame is decoded twice and no result is cached between runs.

== Throughput

Table~@tab:runtime reports per-sample wall-clock for the integrated configuration on the CPU implementation, on the CUDA implementation, and on a Python reference implementation that mirrors the algorithm stage-for-stage and is included as a runtime baseline rather than as a science contribution. Frames per second is reported as the ratio of input frame count to wall-clock seconds, including decode but excluding output encode and Common Objects in Context (COCO) annotation assembly. The CUDA implementation reaches $K$ frames per second aggregated across the eleven samples, an $S$-fold speedup over the CPU implementation at the same parameter values.

#figure(
  // TODO populate from _dev/validation/bench_3way_11videos.sh once the
  // CUDA close-out lands. Columns: sample, frames, python (s),
  // CPU (s), CUDA (s), CPU speedup over python, CUDA speedup over python.
  rect(width: 100%, height: 1.6in, stroke: 0.5pt, inset: 8pt)[
    _Three-implementation runtime placeholder._ Aggregate per-sample
    wall-clock and frames per second for the Python reference, the
    CPU implementation, and the CUDA implementation across the eleven
    VARTM samples. Replaced once the CUDA close-out completes.
  ],
  caption: [
    Per-sample wall-clock for the integrated configuration on three
    implementations of the same algorithm. Wall-clock includes video
    decode but excludes output encode and annotation export. Speedups
    are reported relative to the Python reference at matched
    parameter values.
  ],
) <tab:runtime>

#figure(
  // image("/typst/figures/runtime_bars.pdf", width: 95%),
  rect(width: 100%, height: 1.8in, stroke: 0.5pt, inset: 8pt)[
    _Three-implementation runtime bar chart placeholder._ Three
    grouped horizontal bars per sample (Python reference in warm
    grey, CPU implementation in atlantic, CUDA implementation in
    garnet) for the eleven-sample subset. Bar length is wall-clock
    seconds, with a secondary axis showing frames per second.
    Samples ordered top-to-bottom by frame count so the longest
    runs are at the top of the chart and the speedup pattern is
    visible across regimes. Companion visualization to
    Table~@tab:runtime; cut from the final paper if the table alone
    is sufficient.
  ],
  caption: [
    Wall-clock per sample on three implementations of the same
    algorithm. Bars are grouped by sample so the per-sample
    speedup pattern is visible at a glance.
  ],
) <fig:runtime_bars>

== Validation

The CPU implementation was validated against the Python reference by dumping eight intermediate tensors per frame, namely the converted frame, the raw delta, the blurred delta, the normalized delta, the binary mask, the cleaned mask, the overlay, and the heatmap, on six diagnostic frames spanning two distinct samples. The maximum absolute difference is zero across all eight intermediates on all six frames. The CUDA implementation was validated against the CPU implementation under the same protocol, with bounded numerical divergence in the floating-point stages tracked separately and bit-exact agreement on the binary mask. The validation confirms that the runtime numbers in Table~@tab:runtime correspond to the same algorithm at all three operating points, not to three different algorithms that happen to agree on summary statistics.
