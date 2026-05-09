# fast-vrifa-rs

This workspace is the bring-up area for a GPU-oriented VRIFA implementation. The current milestone keeps the default CLI path delegated to the locked CPU binary, and adds a `wgpu` path for the first three stages: BGR upload, CIELAB conversion, ROI mask construction, and darken-only delta.

## Layout

- `crates/fast-vrifa-core` defines the backend trait and the CPU fallback implementation.
- `crates/fast-vrifa-cuda` is the future CUDA backend crate.
- `crates/fast-vrifa-wgpu` holds the Metal/Vulkan/DX12 compute path for the first staged GPU bring-up.
- `crates/fast-vrifa-cli` builds the `fast-vrifa` binary.
- `crates/fast-vrifa-py` builds the `fast_vrifa` Python binding surface.
- `tests/parity` holds the local parity smoke harness for bring-up work.

## Build

Build the delegated CLI:

```bash
cargo build --release -p fast-vrifa-cli
```

On macOS with Homebrew OpenCV, set `PKG_CONFIG_PATH=/opt/homebrew/opt/opencv/lib/pkgconfig` before running Cargo.

Build the `wgpu` path:

```bash
cargo build --release -p fast-vrifa-cli --features wgpu
```

Build the CUDA placeholder:

```bash
cargo build --release -p fast-vrifa-cli --features cuda
```

## Run

The default binary forwards all CLI arguments to the locked CPU implementation. Build the reference binary first:

```bash
cd ../vrifa-rs
cargo build --release -p vrifa-cli
cd ../fast-vrifa-rs
./target/release/fast-vrifa \
  --video-path ../data/input_1.mp4 \
  --output-dir /tmp/fast_vrifa_scaffold \
  --write-videos
```

Run the staged `wgpu` path:

```bash
./target/release/fast-vrifa \
  --backend wgpu \
  --video-path ../data/input_1.mp4 \
  --output-dir /tmp/fast_vrifa_wgpu \
  --write-mask-pngs true \
  --write-overlay-pngs true \
  --write-heatmap-pngs true \
  --roi-margin 0.15 \
  --annotation-formats coco
```

Run the hybrid CPU backend without delegation:

```bash
./target/release/fast-vrifa \
  --backend cpu \
  --video-path ../data/input_2.mp4 \
  --output-dir /tmp/fast_vrifa_cpu_backend \
  --write-mask-pngs true \
  --write-overlay-pngs true \
  --write-heatmap-pngs true
```

If the CPU binary lives somewhere else, set `VRIFA_BIN=/path/to/vrifa` before running `fast-vrifa`.

## Verify

```bash
PKG_CONFIG_PATH=/opt/homebrew/opt/opencv/lib/pkgconfig cargo test --workspace
PKG_CONFIG_PATH=/opt/homebrew/opt/opencv/lib/pkgconfig cargo test --workspace --features wgpu
PKG_CONFIG_PATH=/opt/homebrew/opt/opencv/lib/pkgconfig cargo test --workspace --features cuda
tests/parity/run_smoke.sh
```

## Current Status

The default path still delegates to the locked CPU binary. `--backend cpu` runs the same hybrid orchestration locally through the backend trait, which makes the non-delegated path testable without Metal. `--backend wgpu` runs colorspace, ROI, and darken-only delta on Metal via `wgpu`, downloads the delta plane, and finishes the remaining stages on the CPU. No intentional divergences are documented for this increment.
