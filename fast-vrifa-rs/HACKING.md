# Hacking on fast-vrifa-rs

This workspace is separate from `vrifa-rs/` on purpose. Treat the CPU tree as a locked baseline and keep all experimental GPU work contained here.

## Add or reshape a backend trait

Edit `crates/fast-vrifa-core/src/lib.rs`. The trait in that crate is the contract between the CLI and the eventual CUDA or wgpu backends.

## Bring up a backend crate

Edit either `crates/fast-vrifa-cuda` or `crates/fast-vrifa-wgpu`. Keep the crate independently buildable and expose a backend type that implements the trait from `fast-vrifa-core`.

## Change the CLI

Edit `crates/fast-vrifa-cli`. During scaffold bring-up the binary is a passthrough wrapper, so CLI-side changes should preserve the exact argument flow expected by the locked CPU implementation.

## Change the binding

Edit `crates/fast-vrifa-py` and keep the surface aligned with `fast_vrifa.run(...)` and `fast_vrifa.Config(...)`.

## Verify

```bash
PKG_CONFIG_PATH=/opt/homebrew/opt/opencv/lib/pkgconfig cargo test --workspace
tests/parity/run_smoke.sh
```

If you intentionally change output behavior later in the bring-up, record the reason and a reproduction case in `docs/divergences.md` before merging.
