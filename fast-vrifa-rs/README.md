# fast-vrifa-rs

This workspace is the bring-up area for a GPU-oriented VRIFA implementation. The first milestone keeps parity trivial by delegating execution to the locked CPU binary while the backend trait, crate layout, and verification path are established.

## Layout

- `crates/fast-vrifa-core` defines the backend trait and shared device-side placeholders.
- `crates/fast-vrifa-cuda` is the future CUDA backend crate.
- `crates/fast-vrifa-wgpu` is the future wgpu backend crate.
- `crates/fast-vrifa-cli` builds the `fast-vrifa` binary.
- `crates/fast-vrifa-py` builds the `fast_vrifa` Python binding surface.
- `tests/parity` holds the local parity smoke harness for bring-up work.

## Build

Build the delegated scaffold:

```bash
cargo build --release -p fast-vrifa-cli
```

On macOS with Homebrew OpenCV, set `PKG_CONFIG_PATH=/opt/homebrew/opt/opencv/lib/pkgconfig` before running Cargo.

Build the placeholder backend variants:

```bash
cargo build --release -p fast-vrifa-cli --features cuda
cargo build --release -p fast-vrifa-cli --features wgpu
```

## Run

The scaffold binary forwards all CLI arguments to the locked CPU implementation. Build the reference binary first:

```bash
cd ../vrifa-rs
cargo build --release -p vrifa-cli
cd ../fast-vrifa-rs
./target/release/fast-vrifa \
  --video-path ../data/input_1.mp4 \
  --output-dir /tmp/fast_vrifa_scaffold \
  --write-videos
```

If the CPU binary lives somewhere else, set `VRIFA_BIN=/path/to/vrifa` before running `fast-vrifa`.

## Verify

```bash
PKG_CONFIG_PATH=/opt/homebrew/opt/opencv/lib/pkgconfig cargo test --workspace
tests/parity/run_smoke.sh
```

## Current Status

This branch is the workspace scaffold only. The binary and binding surface are wired, the backend crates compile, and parity is guaranteed by delegation. GPU kernels and backend execution land in later increments.
