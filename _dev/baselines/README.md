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
  --out outputs_baseline_lekanidis_vosniakos_2020/input_1 \
  --roi-mask data/roi_masks/input_1.png \
  --write-overlay-video \
  --write-mask-video
```

Each script writes masks to:

```text
<out>/masks/frame_NNNNNN.png
```

When requested, videos are written to:

```text
<out>/videos/overlay.mp4
<out>/videos/mask.mp4
```

By default, the scripts write every frame. For the 55-frame labeled
subset only, add `--frame-selection labeled`.

If the videos live outside the current checkout, `run_all.sh` accepts a
`DATA_DIR` override:

```bash
DATA_DIR=/path/to/data FRAME_SELECTION=labeled ./_dev/baselines/run_all.sh
```

`run_all.sh` uses the repository's sample-specific ROI convention by
default:

- `input_1` uses `data/roi_masks/input_1.png` when present.
- `input_2` through `input_11` run with `--roi-margin 0`.

To evaluate a completed run directory:

```bash
python3 _dev/validation/agreement.py \
  --runs-dir outputs_baseline_lekanidis_vosniakos_2020 \
  --output data/agreement_metrics_lekanidis.json
```
