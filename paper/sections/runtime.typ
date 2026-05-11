= Runtime and Systems

The integrated pipeline runs at single-CPU video rate on the host described below, and the same algorithm has been ported to a CUDA implementation that processes the same eleven samples at substantially higher throughput. This section reports the wall-clock cost of the integrated configuration on both implementations and is deliberately short, because the runtime story is a property of the pipeline rather than a contribution.

== Hardware

All runtimes are reported on a single Linux host. The CPU is an Intel Xeon w9-3495X with 112 cores at a maximum boost frequency of $4.8$ GHz, $512$ GB DDR5 memory, running Ubuntu $22.04$ LTS. The GPU runtimes use an NVIDIA RTX $6000$ Ada Generation with $48$ GB of memory, CUDA $12.4$. Decode and encode use the OpenCV `videoio` interface.

== Throughput

Table~@tab:runtime reports per-sample wall-clock for the integrated configuration on the CPU implementation and on the CUDA implementation. Each per-sample wall-clock is captured as a sequential single-process run, so the reported CPU figures reflect the throughput a research bench observes when the integrated pipeline is the only thing using the host's cores. Frames per second is reported as the ratio of input frame count to wall-clock seconds, including decode but excluding output encode and Common Objects in Context (COCO) annotation assembly. Aggregated across the eleven samples, the CUDA implementation processes $207$ frames per second on average, a $10$-fold speedup over the CPU implementation at the same parameter values.

// INTERNAL: CUDA numbers will be refreshed before final publication
// once the fast-vrifa-rs CUDA close-out lands. Regenerate by running
//   tmux new -d -s paper "cd ~/bench_vrifa && ... python3 /tmp/run_paper_data.py --workers 1"
// on COMECH-2422 and re-running scripts/build_paper_tables.py.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, left, right, right, right),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.8pt),
    table.vline(x: 4, stroke: 0.5pt),
    table.header(
      [*Sample*], [*Frames*], [*CPU fps*], [*CUDA fps*],
      [*Sample*], [*Frames*], [*CPU fps*], [*CUDA fps*],
    ),
    table.hline(stroke: 0.5pt),
    [`input_1`], [706], [$10.8$], [$109.7$],  [`input_7`],  [542],  [$24.4$], [$243.8$],
    [`input_2`], [100], [$11.5$], [$65.7$],   [`input_8`],  [842],  [$23.1$], [$253.3$],
    [`input_3`], [200], [$12.0$], [$96.6$],   [`input_9`],  [1037], [$25.3$], [$274.7$],
    [`input_4`], [542], [$24.5$], [$249.6$],  [`input_10`], [767],  [$23.7$], [$249.6$],
    [`input_5`], [542], [$26.2$], [$238.6$],  [`input_11`], [997],  [$24.3$], [$260.6$],
    [`input_6`], [542], [$23.7$], [$244.0$],  [],          [],     [],       [],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Per-sample frames-per-second of the integrated configuration on
    the CPU (`vrifa-rs`) and CUDA (`fast-vrifa-rs`) implementations.
  ],
) <tab:runtime>

== Validation

The CPU implementation was validated against the Python reference by dumping eight intermediate tensors per frame, namely the converted frame, the raw delta, the blurred delta, the normalized delta, the binary mask, the cleaned mask, the overlay, and the heatmap, on six diagnostic frames spanning two distinct samples. The maximum absolute difference is zero across all eight intermediates on all six frames. The CUDA implementation was validated against the CPU implementation under the same protocol, with bounded numerical divergence in the floating-point stages tracked separately and bit-exact agreement on the binary mask. The validation confirms that the runtime numbers in Table~@tab:runtime correspond to the same algorithm at all three operating points, not to three different algorithms that happen to agree on summary statistics.

== Output formats

Wall-clock numbers above are reported with output side effects disabled. The pipeline supports six independent artifact-export flags for downstream consumption: `--write-mask-pngs`, `--write-overlay-pngs`, and `--write-heatmap-pngs` write per-frame Portable Network Graphics images of the locked mask, the red-edge overlay on the BGR frame, and the Turbo heatmap of the normalized response respectively; `--write-mask-video`, `--write-overlay-video`, and `--write-heatmap-video` emit the same three streams as Moving Picture Experts Group 4 video (libx264 or libopenh264) for human review. None of these flags affects the algorithm or the metrics reported above; they determine only which renders are persisted to disk. The PNG path is bit-exact across runs because it bypasses the video codec; the MP4 path picks up slice-thread nondeterminism from libavcodec at the encode step, which is why the PNG outputs are the canonical machine-readable artifact and the MP4 outputs are reserved for human review. Two diagnostic flags, `--debug-dump-frames` and `--debug-dump-dir`, control the per-stage tensor dump used by the validation step above.
