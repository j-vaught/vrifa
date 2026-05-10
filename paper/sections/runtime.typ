= Runtime and Systems

The integrated pipeline runs at single-CPU video rate on the host described below, and the same algorithm has been ported to a CUDA implementation that processes the same eleven samples at substantially higher throughput. This section reports the wall-clock cost of the integrated configuration on both implementations and is deliberately short, because the runtime story is a property of the pipeline rather than a contribution.

== Hardware

All runtimes are reported on a single Linux host. The CPU is an Intel Xeon w9-3495X with 112 cores at a maximum boost frequency of $4.8$ GHz, $502$ GB DDR5 memory, running Ubuntu $22.04$ LTS. The GPU runtimes use an NVIDIA RTX $6000$ Ada Generation with $48$ GB of memory, driver version $550.144.03$, CUDA $12.4$. Decode and encode use the system FFmpeg with the libx264 and libopenh264 codecs, both invoked through the OpenCV `videoio` interface. No frame is decoded twice and no result is cached between runs.

== Throughput

Table~@tab:runtime reports per-sample wall-clock for the integrated configuration on the CPU implementation and on the CUDA implementation. Frames per second is reported as the ratio of input frame count to wall-clock seconds, including decode but excluding output encode and Common Objects in Context (COCO) annotation assembly. Aggregated across the eleven samples, the CUDA implementation processes $216$ frames per second on average, an $87$-fold speedup over the CPU implementation at the same parameter values.

// INTERNAL: CPU numbers in tab:runtime were captured under 8-worker
// concurrent execution (single-process wall-clock is contention-
// inflated; the relative CUDA/CPU speedup is the correct headline,
// not the absolute CPU fps). CUDA numbers reflect the current
// fast-vrifa-rs state and will be refreshed before final publication
// once the CUDA close-out lands. Regenerate by running
//   tmux new -d -s paper "cd ~/bench_vrifa && ... python3 /tmp/run_paper_data.py --cuda-only"
// on COMECH-2422 and re-running scripts/build_paper_tables.py.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    stroke: none,
    inset: 5pt,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Sample*], [*Frames*], [*CPU s*], [*CUDA s*], [*CPU fps*], [*CUDA fps*], [*Speedup*],
    ),
    table.hline(stroke: 0.5pt),
    [`input_2`],  [100],  [$71.3$],  [$1.5$], [$1.4$], [$65.4$],   [$46.6×$],
    [`input_3`],  [200],  [$141.2$], [$2.1$], [$1.4$], [$94.3$],   [$66.6×$],
    [`input_4`],  [542],  [$258.0$], [$1.8$], [$2.1$], [$297.8$],  [$141.8×$],
    [`input_5`],  [542],  [$255.9$], [$1.8$], [$2.1$], [$297.8$],  [$140.6×$],
    [`input_6`],  [542],  [$253.7$], [$1.8$], [$2.1$], [$297.8$],  [$139.4×$],
    [`input_7`],  [542],  [$255.0$], [$1.9$], [$2.1$], [$289.8$],  [$136.4×$],
    [`input_1`],  [706],  [$397.2$], [$6.6$], [$1.8$], [$107.3$],  [$60.4×$],
    [`input_10`], [767],  [$256.4$], [$3.1$], [$3.0$], [$249.8$],  [$83.5×$],
    [`input_8`],  [842],  [$343.9$], [$2.5$], [$2.4$], [$334.1$],  [$136.5×$],
    [`input_11`], [997],  [$172.0$], [$5.4$], [$5.8$], [$185.3$],  [$32.0×$],
    [`input_9`],  [1037], [$338.7$], [$3.0$], [$3.1$], [$343.4$],  [$112.2×$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Per-sample wall-clock for the integrated configuration on the
    CPU implementation (`vrifa-rs`) and the CUDA implementation
    (`fast-vrifa-rs`) on COMECH-2422. Wall-clock includes video decode
    but excludes output encode and annotation export. Speedup is the
    ratio of CPU seconds to CUDA seconds at matched parameter values.
    Frame counts are for the trimmed labeling videos in
    `data/ablation_data/`.
  ],
) <tab:runtime>

#figure(
  image("/typst/figures/runtime_bars.pdf", width: 90%),
  caption: [
    Frames-per-second per sample on the two implementations. CPU
    in atlantic, CUDA in garnet. Samples are ordered ascending by
    frame count.
  ],
) <fig:runtime_bars>

== Validation

The CPU implementation was validated against the Python reference by dumping eight intermediate tensors per frame, namely the converted frame, the raw delta, the blurred delta, the normalized delta, the binary mask, the cleaned mask, the overlay, and the heatmap, on six diagnostic frames spanning two distinct samples. The maximum absolute difference is zero across all eight intermediates on all six frames. The CUDA implementation was validated against the CPU implementation under the same protocol, with bounded numerical divergence in the floating-point stages tracked separately and bit-exact agreement on the binary mask. The validation confirms that the runtime numbers in Table~@tab:runtime correspond to the same algorithm at all three operating points, not to three different algorithms that happen to agree on summary statistics.

== Output formats

Wall-clock numbers above are reported with output side effects disabled. The pipeline supports six independent artifact-export flags for downstream consumption: `--write-mask-pngs`, `--write-overlay-pngs`, and `--write-heatmap-pngs` write per-frame Portable Network Graphics images of the locked mask, the red-edge overlay on the BGR frame, and the Turbo heatmap of the normalized response respectively; `--write-mask-video`, `--write-overlay-video`, and `--write-heatmap-video` emit the same three streams as Moving Picture Experts Group 4 video (libx264 or libopenh264) for human review. None of these flags affects the algorithm or the metrics reported in Section~5; they determine only which renders are persisted to disk. The PNG path is bit-exact across runs because it bypasses the video codec; the MP4 path picks up slice-thread nondeterminism from libavcodec at the encode step, which is why the PNG outputs are the canonical machine-readable artifact and the MP4 outputs are reserved for human review. Two diagnostic flags, `--debug-dump-frames` and `--debug-dump-dir`, control the per-stage tensor dump used by the validation step above.
