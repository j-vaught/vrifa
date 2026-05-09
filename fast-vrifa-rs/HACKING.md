# Hacking on fast-vrifa-rs

This workspace is separate from `vrifa-rs/` on purpose. Treat the CPU tree as a locked baseline and keep all experimental GPU work contained here.

## Add or reshape a backend trait

Edit `crates/fast-vrifa-core/src/backend.rs`. The trait in that crate is the contract between the CLI and the eventual CUDA or wgpu backends.

## Bring up a backend crate

Edit either `crates/fast-vrifa-cuda` or `crates/fast-vrifa-wgpu`. The `wgpu` crate already carries the stage-1 path: upload, exact BGR->CIELAB lookup, ROI mask fill, and darken-only delta.

## Change the CLI

Edit `crates/fast-vrifa-cli`. The default path still forwards to the locked CPU binary. `--backend cpu` exercises the same hybrid pipeline locally through the trait, `--backend wgpu` moves the staged GPU work onto Metal, and later `--backend cuda` will slot into the same dispatch point.

## Change the binding

Edit `crates/fast-vrifa-py` and keep the surface aligned with `fast_vrifa.run(...)` and `fast_vrifa.Config(...)`.

## Verify

```bash
PKG_CONFIG_PATH=/opt/homebrew/opt/opencv/lib/pkgconfig cargo test --workspace
PKG_CONFIG_PATH=/opt/homebrew/opt/opencv/lib/pkgconfig cargo test --workspace --features wgpu
tests/parity/run_smoke.sh
```

If you intentionally change output behavior later in the bring-up, record the reason and a reproduction case in `docs/divergences.md` before merging.
