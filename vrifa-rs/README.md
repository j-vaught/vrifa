# VRIFA Rust workspace

This workspace contains the Rust port scaffold and runnable implementation split by crate:

- `vrifa-core` contains ndarray-based algorithm stages with no file or video I/O.
- `vrifa-io` contains video and PNG readers/writers.
- `vrifa-annotations` exports COCO, YOLOv5, and Darknet layouts.
- `vrifa-cli` provides the `vrifa` binary and the pipeline orchestration.
- `vrifa-py` exposes the thin PyO3/maturin extension.

Build and run from this directory:

```bash
cargo check --workspace
cargo run -p vrifa-cli --bin vrifa -- --video-path ../data/input_1.mp4 --output-dir ../outputs_rs_1 --write-videos --roi-margin 0.15 --annotation-formats coco
```

The workspace uses OpenCV through Homebrew on this machine for parity-critical operations. If Cargo cannot locate OpenCV, set `PKG_CONFIG_PATH` to the directory containing `opencv4.pc`.
