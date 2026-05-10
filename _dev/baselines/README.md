# Classical-CV baselines

This directory contains two standalone OpenCV reimplementations of the
published flow-front segmentation pipelines used as VRIFA baselines.
They do not import from `vrifa-rs` or `vrifa-core`.

Scripts:

- `lekanidis_vosniakos_2020.py`
- `almazan_lazaro_2022.py`
- `run_all.sh`

Example:

```bash
python3 _dev/baselines/lekanidis_vosniakos_2020.py \
  --video data/input_1.mp4 \
  --out outputs_baseline_lekanidis_vosniakos_2020/input_1
```

Each script writes masks to:

```text
<out>/masks/frame_NNNNNN.png
```

By default, the scripts write every frame. For the 55-frame labeled
subset only, add `--frame-selection labeled`.

If the videos live outside the current checkout, `run_all.sh` accepts a
`DATA_DIR` override:

```bash
DATA_DIR=/path/to/data FRAME_SELECTION=labeled ./_dev/baselines/run_all.sh
```

To evaluate a completed run directory:

```bash
python3 _dev/validation/agreement.py \
  --runs-dir outputs_baseline_lekanidis_vosniakos_2020 \
  --output data/agreement_metrics_lekanidis.json
```
