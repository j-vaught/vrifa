# VRIFA: VARTM Resin Infusion Flow-Front Assessment

A computer vision tool for automated detection and tracking of resin flow fronts in Vacuum-Assisted Resin Transfer Molding (VARTM) processes. VRIFA processes video recordings of composite manufacturing to generate binary masks, visual overlays, temporal heatmaps, and machine learning training datasets.

## Citation

If you use VRIFA in your research, please cite:

```bibtex
@article{vrifa2025,
  title     = {VRIFA: Automated Flow-Front Detection for VARTM Process Monitoring},
  author    = {[Author Names]},
  journal   = {[Journal Name]},
  year      = {2025},
  volume    = {[Volume]},
  pages     = {[Pages]},
  doi       = {[DOI]}
}
```

> **Note:** Citation details will be updated upon publication.

## Features

- **Darken-only detection** to track resin wetting (ignores brightening artifacts)
- **Peak brightness reference** for handling variable lighting conditions
- **Flow-front detection** using adaptive Otsu thresholding with configurable offset
- **Multiple reference frame strategies** (first frame, running average, dynamic lag)
- **Temporal consistency filtering** to reduce noise and glare artifacts
- **ML annotation export**: COCO format (default), YOLOv5, and Darknet
- **Configurable ROI cropping** to focus on relevant regions
- **Multi-format outputs**: PNG frames, MP4 videos, YAML summaries

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/j-vaught/vrifa.git
cd vrifa
pip install -r requirements.txt
```

### Dependencies

```
opencv-python>=4.5
numpy>=1.20
pyyaml>=5.4
tqdm>=4.60
```

## Quick Start

Basic usage with default settings (darken-only detection, peak reference, COCO export):

```bash
python vrifa.py --video-path input.mp4 --output-dir outputs --write-videos
```

This will:
- Detect darkening pixels (resin wetting) compared to their peak brightness
- Apply -30 Otsu threshold offset for increased sensitivity
- Export COCO annotations for ML training
- Generate overlay, mask, and heatmap videos

## Output Structure

```
outputs/
├── masks/              # Binary mask PNGs (if enabled)
├── overlays/           # Visualization PNGs (if enabled)
├── heatmap/            # Temporal accumulation PNGs (if enabled)
├── videos/             # MP4 outputs (if enabled)
│   ├── overlay.mp4     # Original frames with red flow-front edge
│   ├── mask.mp4        # Binary detection masks
│   └── heatmap.mp4     # Contrast intensity visualization
├── formatCOCO/         # COCO annotations (default)
├── formatYOLO/         # YOLOv5 annotations (if enabled)
├── formatYOLO_v2/      # Darknet annotations (if enabled)
└── run_summary.yaml    # Configuration and timing log
```

---

## Detection Modes

VRIFA provides specialized detection modes optimized for VARTM process monitoring.

### Darken-Only Mode (Default: Enabled)

Detects only pixels that have become darker than their reference, which corresponds to resin wetting the dry fabric. This ignores brightening artifacts from:
- Specular reflections
- Lighting changes
- Camera auto-exposure adjustments

```bash
# Enabled by default, explicitly disable with:
python vrifa.py --no-darken-only ...
```

### Peak Brightness Reference (Default: Enabled)

Instead of comparing to a fixed reference frame, each pixel is compared to its **historical maximum brightness**. This handles scenarios where:
- Pixels start dark (e.g., shadows)
- Brighten as lighting stabilizes
- Then darken again as resin fills in

The algorithm maintains a running maximum brightness map that updates each frame:
```
peak_brightness[pixel] = max(peak_brightness[pixel], current_brightness[pixel])
detection = peak_brightness[pixel] - current_brightness[pixel]
```

```bash
# Enabled by default, explicitly disable with:
python vrifa.py --no-peak-reference ...
```

### Threshold Offset (Default: -30)

The Otsu threshold is adjusted by this offset value. Negative values increase sensitivity (detect more subtle changes), positive values decrease sensitivity.

```bash
# More sensitive (detect subtle darkening)
python vrifa.py --threshold-offset -50

# Less sensitive (only detect strong darkening)
python vrifa.py --threshold-offset 0

# Default is -30
```

---

## ML Annotation Export Formats

VRIFA generates training datasets for deep learning models. COCO format is exported by default.

### COCO Format (Default)

```bash
# Enabled by default, or explicitly:
python vrifa.py --annotation-formats coco --annotation-mode count --annotation-count 100
```

**Structure:**
```
formatCOCO/
├── annotations/instances_default.json
└── images/default/*.png
```

- Categories: `dry` (id: 1), `wet` (id: 2)
- Includes segmentation polygons and bounding boxes
- Compatible with: COCO API, Detectron2, MMDetection

### YOLOv5/v8 Format

```bash
python vrifa.py --annotation-formats yolov5 --annotation-mode count --annotation-count 100
```

**Structure:**
```
formatYOLO/
├── data.yaml
├── train.txt
├── images/train/*.png
└── labels/train/*.txt
```

- Label format: `class_id x1 y1 x2 y2 ...` (normalized polygon)
- Compatible with: Ultralytics YOLOv5-seg, YOLOv8-seg

### Darknet Format

```bash
python vrifa.py --annotation-formats darknet --annotation-mode count --annotation-count 100
```

**Structure:**
```
formatYOLO_v2/
├── obj.data
├── obj.names
├── train.txt
└── obj_train_data/*.png, *.txt
```

- Label format: `class_id cx cy w h` (normalized bounding box)
- Compatible with: Darknet, YOLOv3/v4

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

Process video with all default settings optimized for VARTM:

```bash
python vrifa.py \
  --video-path input.mp4 \
  --output-dir outputs \
  --roi-margin 0 \
  --write-videos
```

### High Sensitivity Detection

For detecting subtle resin penetration:

```bash
python vrifa.py \
  --video-path input.mp4 \
  --output-dir outputs \
  --threshold-offset -50 \
  --write-videos
```

### Classic Mode (No New Features)

Revert to traditional absolute difference detection:

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

Create datasets in multiple formats with sampled frames:

```bash
python vrifa.py \
  --video-path input.mp4 \
  --output-dir dataset \
  --annotation-formats coco,yolov5,darknet \
  --annotation-mode count \
  --annotation-count 200
```

### Dynamic Reference with Linear Lag

For long infusions where a sliding reference window helps:

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

Focus on specific region of the mold:

```bash
python vrifa.py \
  --video-path input.mp4 \
  --roi-margin 0 \
  --roi-margin-left 0.1 \
  --roi-margin-right 0.1 \
  --write-videos
```

### Process Every 5th Frame

Speed up processing for long videos:

```bash
python vrifa.py \
  --video-path input.mp4 \
  --frame-step 5 \
  --write-videos
```

---

## Algorithm Overview

1. **Frame extraction** from input video at specified intervals
2. **ROI cropping** to remove fixtures and edges
3. **Colorspace conversion** (default: CIELAB for perceptual uniformity)
4. **Peak brightness tracking** (if enabled): maintain per-pixel maximum brightness
5. **Reference frame computation** based on selected strategy
6. **Difference calculation**:
   - If `--darken-only`: compute `reference - current` (positive = darkening)
   - If `--peak-reference`: compare to historical peak brightness
   - Otherwise: compute absolute difference
7. **Adaptive thresholding** using Otsu's method with configurable offset
8. **Morphological operations** (closing, opening) to clean mask
9. **Temporal consistency filtering** to reduce transient noise
10. **Contour extraction** for annotation polygon generation
11. **Output generation** (masks, overlays, heatmaps, annotations)

### Why Darken-Only + Peak Reference?

In VARTM monitoring:
- **Dry fabric** appears lighter (reflects more light)
- **Wet fabric** (resin-infused) appears darker (absorbs more light)
- **Lighting variations** can cause temporary brightening
- **Camera auto-exposure** can shift overall brightness

By tracking darkening relative to peak brightness, VRIFA robustly detects resin wetting while ignoring:
- Specular highlights from resin surface
- Auto-exposure compensation events
- Shadow movements from personnel/equipment

---

## Troubleshooting

### Detection is too sensitive (false positives)
```bash
# Increase threshold offset (less negative or positive)
python vrifa.py --threshold-offset -10
# Or use a fixed threshold
python vrifa.py --contrast-threshold 30
```

### Detection is not sensitive enough (missing flow front)
```bash
# Decrease threshold offset (more negative)
python vrifa.py --threshold-offset -50
```

### Noisy/speckled detections
```bash
# Increase morphological operations
python vrifa.py --morph-close-iterations 2 --morph-open-iterations 2
# Or increase minimum area filter
python vrifa.py --min-area 1000
```

### Flickering detections
```bash
# Increase temporal lock frames
python vrifa.py --lock-frames 5
```

### Edge artifacts from fixtures
```bash
# Increase ROI margins
python vrifa.py --roi-margin 0.2
```

---

## Reproducibility

Each run generates a `run_summary.yaml` file containing all parameters and timing information:

```yaml
run_timestamp: '2025-01-17T22:43:00+00:00'
video_path: input.mp4
output_dir: outputs
darken_only: true
peak_reference: true
threshold_offset: -30.0
annotation_formats: ['coco']
# ... all other parameters
```

Use this file to reproduce exact configurations or document experimental setups.

---

## License

[Add license information]

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.
