# VRIFA Usage Reference

After `cargo build --release` from `vrifa-rs/`, the CLI binary is at `./target/release/vrifa`.

## Detection Modes

VRIFA provides specialized detection modes optimized for VARTM process monitoring.

### Darken-Only Mode (Default: Enabled)

Detects only pixels that have become darker than their reference, which corresponds to resin wetting the dry fabric. This ignores brightening artifacts from specular reflections, lighting changes, and camera auto-exposure adjustments.

```bash
vrifa --no-darken-only ...
```

### Peak Brightness Reference (Default: Enabled)

Instead of comparing to a fixed reference frame, each pixel is compared to its historical maximum brightness. This handles scenarios where pixels start dark, brighten as lighting stabilizes, then darken again as resin fills in.

The algorithm maintains a running maximum brightness map that updates each frame:

```
peak_brightness[pixel] = max(peak_brightness[pixel], current_brightness[pixel])
detection = peak_brightness[pixel] - current_brightness[pixel]
```

```bash
vrifa --no-peak-reference ...
```

### Threshold Offset (Default: -30)

The Otsu threshold is adjusted by this offset value. Negative values increase sensitivity and positive values decrease sensitivity.

```bash
vrifa --threshold-offset -50   # More sensitive
vrifa --threshold-offset 0     # Less sensitive
```

---

## ML Annotation Formats

VRIFA generates training datasets for downstream models. COCO format is exported by default.

### COCO Format (Default)

```bash
vrifa --annotation-formats coco --annotation-mode count --annotation-count 100
```

Categories: `dry` (id `1`) and `wet` (id `2`). Includes segmentation polygons and bounding boxes. Compatible with COCO API, Detectron2, and MMDetection.

### YOLOv5/v8 Format

```bash
vrifa --annotation-formats yolov5 --annotation-mode count --annotation-count 100
```

Label format: `class_id x1 y1 x2 y2 ...` with normalized polygon vertices. Compatible with Ultralytics YOLOv5-seg and YOLOv8-seg.

### Darknet Format

```bash
vrifa --annotation-formats darknet --annotation-mode count --annotation-count 100
```

Label format: `class_id cx cy w h` with normalized bounding boxes. Compatible with Darknet and classic YOLO training flows.

### Multiple Formats

```bash
vrifa --annotation-formats coco,yolov5,darknet
```

### Disable Annotation Output

```bash
vrifa --annotation-formats ""
```

---

## Configuration Reference

### Input / Output

| Flag | Description | Default |
|------|-------------|---------|
| `--video-path` | Input video file | `private_assets/input_video.mp4` |
| `--output-dir` | Output directory | `flow_front_outputs` |
| `--write-videos` | Enable all MP4 outputs | `false` |
| `--write-mask-video` | Save mask video | `false` |
| `--write-overlay-video` | Save overlay video | `true` |
| `--write-heatmap-video` | Save heatmap video | `false` |
| `--write-mask-pngs` | Save mask frames | `false` |
| `--write-overlay-pngs` | Save overlay frames | `false` |
| `--write-heatmap-pngs` | Save heatmap frames | `false` |

### Detection Mode

| Flag | Description | Default |
|------|-------------|---------|
| `--darken-only` | Only detect darkening associated with wetting | `true` |
| `--no-darken-only` | Detect any brightness change | - |
| `--peak-reference` | Compare to peak brightness per pixel | `true` |
| `--no-peak-reference` | Compare to the selected reference frame | - |
| `--threshold-offset` | Offset added to Otsu threshold | `-30` |

### Frame Sampling

| Flag | Description | Default |
|------|-------------|---------|
| `--frame-step` | Process every Nth frame | `1` |

### Region of Interest

| Flag | Description | Default |
|------|-------------|---------|
| `--roi-margin` | Fractional crop on all sides | `0.15` |
| `--roi-margin-top` | Top margin override | - |
| `--roi-margin-bottom` | Bottom margin override | - |
| `--roi-margin-left` | Left margin override | - |
| `--roi-margin-right` | Right margin override | - |

### Image Processing

| Flag | Description | Default |
|------|-------------|---------|
| `--colorspace` | `CIELAB`, `RGB`, `HSV`, `GRAYSCALE` | `CIELAB` |
| `--channel-weights` | Per-channel multipliers | `1,1,1` |
| `--contrast-threshold` | Fixed threshold override | - |
| `--contrast-percentile` | Adaptive percentile threshold | - |
| `--blur-kernel` | Gaussian kernel size (odd) | `9` |
| `--skip-blur` | Skip Gaussian blur | `false` |

### Morphology

| Flag | Description | Default |
|------|-------------|---------|
| `--morph-kernel` | Structuring element size | `13` |
| `--morph-shape` | `ellipse`, `rect`, `cross` | `ellipse` |
| `--morph-close-iterations` | Closing iterations | `1` |
| `--morph-open-iterations` | Opening iterations | `1` |
| `--min-area` | Minimum connected-component area in pixels | `400` |

### Reference Frame Strategy

| Mode | Description |
|------|-------------|
| `--ref-mode first` | Compare to the first frame |
| `--ref-mode running` | Compare to an exponential moving average |
| `--ref-mode prev N` | Compare to `N` frames back |
| `--ref-mode absolute N` | Compare to a fixed frame index |
| `--ref-mode dynamic` | Compare to a modeled lag based on progression |

Dynamic mode options:

| Flag | Description | Default |
|------|-------------|---------|
| `--dynamic-calibration-frames` | Frames used to fit growth behavior | `10` |
| `--dynamic-target-fraction` | Target ROI coverage fraction | `0.2` |
| `--dynamic-lag-linear` | Use a linear lag schedule | `false` |
| `--dynamic-lag-linear-start` | Starting lag in frames | `0` |
| `--dynamic-lag-linear-max` | Maximum lag in frames | `60` |
| `--dynamic-lag-scale` | Scale factor for lag | `1.0` |
| `--dynamic-lag-log` | CSV file for lag logging | - |

### Temporal Filtering

| Flag | Description | Default |
|------|-------------|---------|
| `--lock-frames` | Frames a filled pixel must persist before lock-in | `3` |

### Annotation Output

| Flag | Description | Default |
|------|-------------|---------|
| `--annotation-formats` | `coco`, `yolov5`, `darknet` | `coco` |
| `--annotation-mode` | `all`, `count`, `stride` | `all` |
| `--annotation-count` | Number of frames when using `count` mode | - |
| `--annotation-stride` | Frame interval when using `stride` mode | `1` |
| `--annotation-segmentation-tolerance` | Polygon simplification tolerance in pixels | `0` |
| `--annotation-segmentation-max-edge-length` | Maximum densified edge length in pixels | `0` |

---

## Examples

### Basic Processing

```bash
vrifa \
  --video-path input.mp4 \
  --output-dir outputs \
  --roi-margin 0 \
  --write-videos
```

### High-Sensitivity Detection

```bash
vrifa \
  --video-path input.mp4 \
  --output-dir outputs \
  --threshold-offset -50 \
  --write-videos
```

### Minimal Artifact Set

```bash
vrifa \
  --video-path input.mp4 \
  --output-dir outputs \
  --no-darken-only \
  --no-peak-reference \
  --threshold-offset 0 \
  --annotation-formats "" \
  --write-videos
```

### Generate a Training Dataset

```bash
vrifa \
  --video-path input.mp4 \
  --output-dir dataset \
  --annotation-formats coco,yolov5,darknet \
  --annotation-mode count \
  --annotation-count 200
```

### Dynamic Reference with Linear Lag

```bash
vrifa \
  --video-path input.mp4 \
  --ref-mode dynamic \
  --dynamic-lag-linear \
  --dynamic-lag-linear-start 0 \
  --dynamic-lag-linear-max 45 \
  --write-videos
```

### Custom ROI

```bash
vrifa \
  --video-path input.mp4 \
  --roi-margin 0 \
  --roi-margin-left 0.1 \
  --roi-margin-right 0.1 \
  --write-videos
```

### Process Every Fifth Frame

```bash
vrifa \
  --video-path input.mp4 \
  --frame-step 5 \
  --write-videos
```

---

## Algorithm Overview

1. Extract or sample frames from the input video.
2. Apply the configured region of interest.
3. Convert the frame into the requested working colorspace.
4. Update the peak-brightness map if that mode is enabled.
5. Select the reference frame according to the chosen strategy.
6. Compute the response image in darken-only or full-distance mode.
7. Normalize and threshold the response map.
8. Apply morphology and minimum-area filtering.
9. Lock persistent detections through the temporal guard.
10. Extract contours and annotation boxes.
11. Write the requested videos, frame images, and annotations.

### Why Darken-Only Plus Peak Reference?

In VARTM monitoring, dry fabric often appears lighter and wet fabric darkens as resin fills the weave. Camera auto-exposure, reflections, and glare can also brighten parts of the frame temporarily. Tracking darkening against the highest previously observed brightness makes the detector robust to those brightening events while staying sensitive to true wetting.

---

## Reproducibility

Each run writes `run_summary.yaml` with the resolved configuration, timing information, ROI coverage, and output settings. Keep that file with exported artifacts if you need to reproduce or compare runs later.

---

## Troubleshooting

### Detection is too sensitive

```bash
vrifa --threshold-offset -10
vrifa --contrast-threshold 30
```

### Detection is not sensitive enough

```bash
vrifa --threshold-offset -50
```

### Detections are noisy or speckled

```bash
vrifa --morph-close-iterations 2 --morph-open-iterations 2
vrifa --min-area 1000
```

### Detections flicker

```bash
vrifa --lock-frames 5
```

### Edge fixtures leak into the mask

```bash
vrifa --roi-margin 0.2
```

---

## Verifying Your Changes

For fast regression coverage, run:

```bash
cd vrifa-rs
cargo test --workspace --release
```

For artifact-level validation against the frozen sample runs, use the internal harness in `_dev/`:

```bash
python3 _dev/validation/compare_runs.py /tmp/py_run /tmp/rs_run
```

The fixture generator and stage-dump tools that refresh validation assets also live under `_dev/validation/`.
