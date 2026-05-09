# VRIFA Workspace

This directory contains the Rust workspace that builds the VRIFA CLI, core algorithm crates, fixtures, and benchmarks.

## Workspace Layout

- `crates/vrifa-core` contains the pure algorithm stages and shared pipeline types.
- `crates/vrifa-io` contains video readers, MP4 writers, and PNG writers.
- `crates/vrifa-annotations` contains COCO, YOLOv5, and Darknet exporters.
- `crates/vrifa-cli` contains the binary entrypoint, configuration parsing, and end-to-end orchestration.
- `crates/vrifa-py` contains the optional binding crate layered over the Rust pipeline.
- `tests/fixtures` contains frozen test assets and stage goldens.
- `docs/archive` contains closed investigation notes that are not required for day-to-day work.

## Stage Modules

- `colorspace.rs` converts BGR frames into the configured working colorspace.
- `roi.rs` resolves fractional margins and builds the ROI mask.
- `reference.rs` implements first, running, previous, absolute, and dynamic reference selection helpers.
- `peak.rs` updates the per-pixel peak-brightness map.
- `delta.rs` computes the channel-weighted response image.
- `threshold.rs` selects scalar thresholds with Otsu, manual, or percentile modes.
- `morphology.rs` blurs, thresholds, filters, and produces the cleaned mask.
- `lock.rs` applies persistence-based temporal locking.
- `overlay.rs` renders the red-edge overlay over the source BGR frame.
- `heatmap.rs` maps the normalized response through the Turbo colormap.
- `contours.rs` extracts bounding boxes and segmentation polygons.
- `sampling.rs` selects annotation frames for `all`, `count`, and `stride` modes.
- `cvutil.rs` holds OpenCV and ndarray conversion helpers shared across stages.

## Build

```bash
cargo build --release
```

## Run

```bash
./target/release/vrifa \
  --video-path ../data/input_1.mp4 \
  --output-dir ../outputs \
  --write-videos
```

## Verify

```bash
cargo test --workspace --release
../_dev/validation/compare_runs.py /tmp/py_run /tmp/rs_run
cargo bench -p vrifa-cli --bench perf_gate
```

## Tests and Fixtures

The files under `tests/fixtures/` are frozen goldens that lock the stage behavior. The parity tests in `crates/vrifa-core/tests/` load those fixtures directly and fail as soon as a stage drifts from the expected result. When behavior intentionally changes, regenerate the affected fixtures with the tooling under `../_dev/validation/` and commit the code and refreshed goldens together.

## Benchmarks

`cargo bench -p vrifa-cli --bench perf_gate` runs a three-tier gate for each bundled sample video:

- `detector` measures the per-frame detection pipeline without output side effects.
- `core` adds MP4 writing on top of the detector tier.
- `full` adds PNG and annotation export on top of the core tier.

The split makes regressions easier to localize. If only `full` regresses, the detector is still stable and the slowdown is in output handling rather than the stage pipeline.
