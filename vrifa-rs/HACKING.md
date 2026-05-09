# Hacking on VRIFA

This workspace is organized so most algorithm changes stay inside `vrifa-core`, while the CLI and exporters mostly wire stages together.

## Fix a bug in stage X

Edit `crates/vrifa-core/src/<stage>.rs` and keep the change local to that stage when possible. Run:

```bash
cargo test --workspace --release
```

The matching `crates/vrifa-core/tests/<stage>_parity.rs` test will compare the stage output against the frozen golden in `tests/fixtures/`.

## Intentionally change the algorithm

Edit the Rust stage, then regenerate only the affected fixtures and commit the code and goldens together. The one-line entrypoint is:

```bash
../_dev/validation/generate_stage_fixtures.py
```

After refreshing the fixtures, rerun:

```bash
cargo test --workspace --release
cargo bench -p vrifa-cli --bench perf_gate
```

The fixture files are the contract. If the behavior changes intentionally, the updated frozen golden must land in the same change set as the code.

## Add a new stage or module

Add the source file under `crates/vrifa-core/src/`, expose it from `crates/vrifa-core/src/lib.rs`, and add a matching integration test under `crates/vrifa-core/tests/`. Reuse the helpers in `crates/vrifa-core/tests/common/mod.rs` for fixture loading, assertion helpers, and shared config parsing.

## Common Workflow

```bash
cargo fmt
cargo test --workspace --release
cargo bench -p vrifa-cli --bench perf_gate
```

If you need artifact-level validation beyond the stage tests, the internal harness under `../_dev/validation/` provides run comparison, stage dumping, and fixture regeneration utilities.
