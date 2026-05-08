# Problem 1 stage diagnosis

This branch adds debug-only stage dumping for the Rust CLI and matching Python-side dump tooling:

- Rust: `vrifa --debug-dump-frames ... --debug-dump-dir ...`
- Python: `tools/dump_python_stages.py`
- Comparison: `tools/compare_stage_dumps.py`

## Requested stage comparison

The requested frame sets were compared with `np.max(np.abs(py - rs))` for each stage.

### input_1

Frames: `50`, `200`, `500`

For every requested stage below, `max_abs_diff = 0` on all three frames:

1. `frame_converted`
2. `delta`
3. `delta_blur`
4. `delta_norm`
5. `binary`
6. `mask`
7. `overlay`
8. `heatmap`

There is no CPU-stage divergence on the sampled `input_1` frames.

### input_2

Frames: `30`, `60`, `90`

For every requested stage below, `max_abs_diff = 0` on all three frames:

1. `frame_converted`
2. `delta`
3. `delta_blur`
4. `delta_norm`
5. `binary`
6. `mask`
7. `overlay`
8. `heatmap`

There is no CPU-stage divergence on the sampled `input_2` frames.

## Additional check

`compare_runs.py` still fails on the full video outputs for `input_2`:

- `overlay.mp4 mean per-pixel L2 = 4.250483`
- `heatmap.mp4 mean per-pixel L2 = 3.534062`

Because the raw `overlay.npy` and `heatmap.npy` stage dumps are bit-exact, the remaining divergence must be downstream of stage 8, in the MP4 encode/decode path rather than the CPU detection pipeline.

To confirm that, the decoded MP4 frames were compared against the raw dumped arrays on `input_2`.

### overlay raw vs decoded MP4 L2

- Frame `30`: raw→Python MP4 `6.5341`, raw→Rust MP4 `5.5240`, Python MP4→Rust MP4 `4.1129`
- Frame `60`: raw→Python MP4 `7.0142`, raw→Rust MP4 `5.8991`, Python MP4→Rust MP4 `4.5420`
- Frame `90`: raw→Python MP4 `7.3263`, raw→Rust MP4 `6.1121`, Python MP4→Rust MP4 `4.7136`

### heatmap raw vs decoded MP4 L2

- Frame `30`: raw→Python MP4 `5.6601`, raw→Rust MP4 `4.9375`, Python MP4→Rust MP4 `3.6061`
- Frame `60`: raw→Python MP4 `5.8917`, raw→Rust MP4 `5.0681`, Python MP4→Rust MP4 `3.9034`
- Frame `90`: raw→Python MP4 `6.0361`, raw→Rust MP4 `5.1563`, Python MP4→Rust MP4 `3.8836`

## Diagnosis

The first requested stage whose `max_abs_diff` exceeds `1` does not exist on the sampled frames. Every intermediate through final raw `overlay` and `heatmap` is bit-exact. The parity failure is introduced after those stages, in the video write path.
