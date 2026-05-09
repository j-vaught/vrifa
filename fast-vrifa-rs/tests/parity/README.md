# Parity Smoke Test

The smoke harness covers two execution paths:

- the default delegated path, which should stay byte-identical to the locked CPU binary
- the `--backend wgpu` path, which runs colorspace, ROI, and darken-only delta on the GPU and must still satisfy the existing parity thresholds

Run:

```bash
./tests/parity/run_smoke.sh
```

The script builds the reference CPU binary, builds `fast-vrifa --features wgpu`, runs the CPU reference plus both fast-vrifa modes on `data/input_1.mp4` and `data/input_2.mp4`, and compares the output directories with `_dev/validation/compare_runs.py`.
