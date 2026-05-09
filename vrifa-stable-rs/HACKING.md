# Hacking on VRIFA

This workspace is organized so most algorithm changes stay inside `vrifa-stable-core`, while the CLI and exporters mostly wire stages together.

## Fix a bug in stage X

Edit `crates/vrifa-stable-core/src/<stage>.rs` and keep the change local to that stage when possible. Run:

```bash
cargo build --release --workspace
cargo test --workspace --release
```

For end-to-end behavior verification, run a full pipeline against one of the sample videos and inspect the outputs against the artifacts under `../reference_outputs/`.

## Intentionally change the algorithm

Edit the Rust stage, refresh any saved reference outputs that move as a result, and run:

```bash
cargo bench -p vrifa-stable-cli --bench perf_gate
```

so any performance change is recorded with the behavior change.

## Add a new stage or module

Add the source file under `crates/vrifa-stable-core/src/`, expose it from `crates/vrifa-stable-core/src/lib.rs`, and wire it through `vrifa-stable-cli` where appropriate.

## Common Workflow

```bash
cargo fmt
cargo build --workspace --release
cargo bench -p vrifa-stable-cli --bench perf_gate
```

If you need artifact-level validation, the internal harness under `../_dev/validation/` provides run-level comparison and stage-dumping utilities.
