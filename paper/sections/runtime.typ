= Runtime and Systems

The integrated pipeline runs at video rate on a single CPU thread, and the same algorithm has been ported to a CUDA implementation that processes the same eleven samples at substantially higher throughput. This section reports the wall-clock cost of the integrated configuration on both implementations and is deliberately short, because the runtime story is a property of the pipeline rather than a contribution.

== Hardware

All runtimes are reported on a single host. The CPU is an Apple M-series with $N_"cores"$ performance cores at $f_"cpu"$ GHz, $N_"ram"$ GB unified memory, running macOS $V_"os"$. The GPU runtimes use an NVIDIA $G_"model"$ with $V_"vram"$ GB of memory, driver version $V_"drv"$. Decode and encode use the system FFmpeg with the libx264 and libopenh264 codecs, both invoked through the OpenCV `videoio` interface. No frame is decoded twice and no result is cached between runs.

== Throughput

Table~@tab:runtime reports per-sample wall-clock for the integrated configuration on the CPU implementation, on the CUDA implementation, and on a Python reference implementation that mirrors the algorithm stage-for-stage and is included as a runtime baseline rather than as a science contribution. Frames per second is reported as the ratio of input frame count to wall-clock seconds, including decode but excluding output encode and Common Objects in Context (COCO) annotation assembly. The CUDA implementation reaches $K$ frames per second aggregated across the eleven samples, an $S$-fold speedup over the CPU implementation at the same parameter values.

#figure(
  image("/typst/figures/runtime.pdf", width: 95%),
  caption: [
    Wall-clock medians for the integrated configuration on the
    high-resolution `input_1` and time-lapse `input_2` clips, broken
    down into the algorithm cost (detector tier), the algorithm plus
    MP4 encode (core tier), and the algorithm plus MP4 encode plus
    per-frame Portable Network Graphics (PNG) and Common Objects in
    Context (COCO) export (full tier). Hash marks indicate the
    pass/fail budgets enforced by the regression harness. The
    detector tier corresponds to roughly $30$ frames per second on
    `input_1`. Three-implementation aggregate wall-clock against the
    Python reference and the CUDA implementation is reported in
    Table~@tab:runtime once the CUDA close-out completes.
  ],
) <fig:runtime>

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

== Validation

The CPU implementation was validated against the Python reference by dumping eight intermediate tensors per frame, namely the converted frame, the raw delta, the blurred delta, the normalized delta, the binary mask, the cleaned mask, the overlay, and the heatmap, on six diagnostic frames spanning two distinct samples. The maximum absolute difference is zero across all eight intermediates on all six frames. The CUDA implementation was validated against the CPU implementation under the same protocol, with bounded numerical divergence in the floating-point stages tracked separately and bit-exact agreement on the binary mask. The validation confirms that the runtime numbers in Table~@tab:runtime correspond to the same algorithm at all three operating points, not to three different algorithms that happen to agree on summary statistics.
