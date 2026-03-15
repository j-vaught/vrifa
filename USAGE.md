# VRIFA Usage Reference

## Detection Modes

VRIFA provides specialized detection modes optimized for VARTM process monitoring.

### Darken-Only Mode (Default: Enabled)

Detects only pixels that have become darker than their reference, which corresponds to resin wetting the dry fabric. This ignores brightening artifacts from specular reflections, lighting changes, and camera auto-exposure adjustments.

```bash
python vrifa.py --no-darken-only ...
```

### Peak Brightness Reference (Default: Enabled)

Instead of comparing to a fixed reference frame, each pixel is compared to its historical maximum brightness. This handles scenarios where pixels start dark, brighten as lighting stabilizes, then darken again as resin fills in.

The algorithm maintains a running maximum brightness map that updates each frame:
```
peak_brightness[pixel] = max(peak_brightness[pixel], current_brightness[pixel])
detection = peak_brightness[pixel] - current_brightness[pixel]
```

```bash
python vrifa.py --no-peak-reference ...
```

### Threshold Offset (Default: -30)

The Otsu threshold is adjusted by this offset value. Negative values increase sensitivity (detect more subtle changes), positive values decrease sensitivity.

```bash
python vrifa.py --threshold-offset -50   # More sensitive
python vrifa.py --threshold-offset 0     # Less sensitive
```

---

## ML Annotation Export Formats

VRIFA generates training datasets for deep learning models. COCO format is exported by default.

### COCO Format (Default)

```bash
python vrifa.py --annotation-formats coco --annotation-mode count --annotation-count 100
```

Categories: `dry` (id: 1), `wet` (id: 2). Includes segmentation polygons and bounding boxes. Compatible with COCO API, Detectron2, MMDetection.

### YOLOv5/v8 Format

```bash
python vrifa.py --annotation-formats yolov5 --annotation-mode count --annotation-count 100
```

Label format: `class_id x1 y1 x2 y2 ...` (normalized polygon). Compatible with Ultralytics YOLOv5-seg, YOLOv8-seg.

### Darknet Format

```bash
python vrifa.py --annotation-formats darknet --annotation-mode count --annotation-count 100
```

Label format: `class_id cx cy w h` (normalized bounding box). Compatible with Darknet, YOLOv3/v4.

### Multiple Formats

```bash
python vrifa.py --annotation-formats coco,yolov5,darknet
```

### Disable Annotation Export

```bash
python vrifa.py --annotation-formats ""
```

---

## Configuration Reference

### Input/Output

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
| `--darken-only` | Only detect darkening (resin wetting) | `true` |
| `--no-darken-only` | Detect any brightness change | - |
| `--peak-reference` | Compare to peak brightness per pixel | `true` |
| `--no-peak-reference` | Compare to fixed reference frame | - |
| `--threshold-offset` | Offset added to Otsu threshold | `-30` |

### Frame Sampling

| Flag | Description | Default |
|------|-------------|---------|
| `--frame-step` | Process every Nth frame | `1` |

### Region of Interest

| Flag | Description | Default |
|------|-------------|---------|
| `--roi-margin` | Fractional crop on all sides (0-0.49) | `0.15` |
| `--roi-margin-top` | Top margin override | - |
| `--roi-margin-bottom` | Bottom margin override | - |
| `--roi-margin-left` | Left margin override | - |
| `--roi-margin-right` | Right margin override | - |

### Image Processing

| Flag | Description | Default |
|------|-------------|---------|
| `--colorspace` | `CIELAB`, `RGB`, `HSV`, `GRAYSCALE` | `CIELAB` |
| `--channel-weights` | Per-channel multipliers (comma-separated) | `1,1,1` |
| `--contrast-threshold` | Fixed threshold override (0-255) | - |
| `--contrast-percentile` | Adaptive percentile threshold (0-100) | - |
| `--blur-kernel` | Gaussian kernel size (odd) | `9` |
| `--skip-blur` | Skip Gaussian blur step | `false` |

### Morphology

| Flag | Description | Default |
|------|-------------|---------|
| `--morph-kernel` | Structuring element size | `13` |
| `--morph-shape` | `ellipse`, `rect`, `cross` | `ellipse` |
| `--morph-close-iterations` | Closing iterations | `1` |
| `--morph-open-iterations` | Opening iterations | `1` |
| `--min-area` | Minimum component area (px) | `400` |

### Reference Frame Strategy

| Mode | Description |
|------|-------------|
| `--ref-mode first` | Compare to first frame (default) |
| `--ref-mode running` | Exponential moving average |
| `--ref-mode prev N` | Compare to N frames back |
| `--ref-mode absolute N` | Compare to fixed frame index |
| `--ref-mode dynamic` | Modeled lag based on flow progression |

**Dynamic mode options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--dynamic-calibration-frames` | Frames for growth modeling | `10` |
| `--dynamic-target-fraction` | Target ROI coverage (0-1) | `0.2` |
| `--dynamic-lag-linear` | Use linear lag schedule | `false` |
| `--dynamic-lag-linear-start` | Starting lag (frames) | `0` |
| `--dynamic-lag-linear-max` | Maximum lag (frames) | `60` |
| `--dynamic-lag-scale` | Scale factor for lag | `1.0` |
| `--dynamic-lag-log` | CSV file for lag logging | - |

### Temporal Filtering

| Flag | Description | Default |
|------|-------------|---------|
| `--lock-frames` | Frames pixel must stay filled to persist | `3` |

### Annotation Export

| Flag | Description | Default |
|------|-------------|---------|
| `--annotation-formats` | `coco`, `yolov5`, `darknet` (comma-separated) | `coco` |
| `--annotation-mode` | `all`, `count`, `stride` | `all` |
| `--annotation-count` | Number of frames (with `count` mode) | - |
| `--annotation-stride` | Frame interval (with `stride` mode) | `1` |
| `--annotation-segmentation-tolerance` | Polygon simplification (px) | `0` |
| `--annotation-segmentation-max-edge-length` | Max edge length (px) | `0` |

---

## Examples

### Basic Processing (Recommended)

```bash
python vrifa.py \
  --video-path input.mp4 \
  --output-dir outputs \
  --roi-margin 0 \
  --write-videos
```

### High Sensitivity Detection

```bash
python vrifa.py \
  --video-path input.mp4 \
  --output-dir outputs \
  --threshold-offset -50 \
  --write-videos
```

### Classic Mode (No New Features)

```bash
python vrifa.py \
  --video-path input.mp4 \
  --output-dir outputs \
  --no-darken-only \
  --no-peak-reference \
  --threshold-offset 0 \
  --annotation-formats "" \
  --write-videos
```

### Generate ML Training Dataset

```bash
python vrifa.py \
  --video-path input.mp4 \
  --output-dir dataset \
  --annotation-formats coco,yolov5,darknet \
  --annotation-mode count \
  --annotation-count 200
```

### Dynamic Reference with Linear Lag

```bash
python vrifa.py \
  --video-path input.mp4 \
  --ref-mode dynamic \
  --dynamic-lag-linear \
  --dynamic-lag-linear-start 0 \
  --dynamic-lag-linear-max 45 \
  --write-videos
```

### Custom ROI

```bash
python vrifa.py \
  --video-path input.mp4 \
  --roi-margin 0 \
  --roi-margin-left 0.1 \
  --roi-margin-right 0.1 \
  --write-videos
```

### Process Every 5th Frame

```bash
python vrifa.py \
  --video-path input.mp4 \
  --frame-step 5 \
  --write-videos
```

---

## Algorithm Overview

1. Frame extraction from input video at specified intervals
2. ROI cropping to remove fixtures and edges
3. Colorspace conversion (default: CIELAB for perceptual uniformity)
4. Peak brightness tracking (if enabled): maintain per-pixel maximum brightness
5. Reference frame computation based on selected strategy
6. Difference calculation (darken-only, peak-reference, or absolute)
7. Adaptive thresholding using Otsu's method with configurable offset
8. Morphological operations (closing, opening) to clean mask
9. Temporal consistency filtering to reduce transient noise
10. Contour extraction for annotation polygon generation
11. Output generation (masks, overlays, heatmaps, annotations)

### Why Darken-Only + Peak Reference?

In VARTM monitoring, dry fabric appears lighter (reflects more light) and wet fabric (resin-infused) appears darker (absorbs more light). Lighting variations and camera auto-exposure can cause temporary brightening. By tracking darkening relative to peak brightness, VRIFA robustly detects resin wetting while ignoring specular highlights, auto-exposure events, and shadow movements.

---

## Reproducibility

Each run generates a `run_summary.yaml` file containing all parameters and timing information. Use this file to reproduce exact configurations or document experimental setups.

---

## Troubleshooting

### Detection is too sensitive (false positives)
```bash
python vrifa.py --threshold-offset -10
python vrifa.py --contrast-threshold 30
```

### Detection is not sensitive enough (missing flow front)
```bash
python vrifa.py --threshold-offset -50
```

### Noisy/speckled detections
```bash
python vrifa.py --morph-close-iterations 2 --morph-open-iterations 2
python vrifa.py --min-area 1000
```

### Flickering detections
```bash
python vrifa.py --lock-frames 5
```

### Edge artifacts from fixtures
```bash
python vrifa.py --roi-margin 0.2
```
