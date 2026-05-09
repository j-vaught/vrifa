# Parity Smoke Test

The scaffold milestone delegates `fast-vrifa` to the locked CPU implementation, so parity should pass exactly.

Run:

```bash
./tests/parity/run_smoke.sh
```

The script builds the reference CPU binary, builds `fast-vrifa`, runs both on `data/input_2.mp4`, and compares the output directories with `_dev/validation/compare_runs.py`.
