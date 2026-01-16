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

- **Flow-front detection** using adaptive thresholding and morphological operations
- **Multiple reference frame strategies** (first frame, running average, dynamic lag)
- **Temporal consistency filtering** to reduce noise and glare artifacts
- **Three ML annotation export formats**: COCO, YOLOv5, and Darknet
- **Configurable ROI cropping** to focus on relevant regions
- **Multi-format outputs**: PNG frames, MP4 videos, YAML summaries

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/[username]/vrifa.git
cd vrifa
pip install -r requirements.txt
```

## Quick Start

```bash
python vrifa.py \
  --video-path input.mp4 \
  --output-dir outputs \
  --write-videos
```

## Output Structure

```
outputs/
├── masks/              # Binary mask PNGs (if enabled)
├── overlays/           # Visualization PNGs (if enabled)
├── heatmap/            # Temporal accumulation PNGs (if enabled)
├── videos/             # MP4 outputs (if enabled)
│   ├── overlay.mp4
│   ├── mask.mp4
│   └── heatmap.mp4
├── formatCOCO/         # COCO annotations (if enabled)
├── formatYOLO/         # YOLOv5 annotations (if enabled)
├── formatYOLO_v2/      # Darknet annotations (if enabled)
└── run_summary.yaml    # Configuration and timing log
```

---

## ML Annotation Export Formats

VRIFA generates training datasets for deep learning models in three standard formats.

### COCO Format

```bash
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

---

## Configuration Reference

### Input/Output

| Flag | Description | Default |
|------|-------------|---------|
| `--video-path` | Input video file | `private_assets/input_video.mp4` |
| `--output-dir` | Output directory | `flow_front_outputs` |
| `--write-videos` | Enable all MP4 outputs | `false` |
| `--write-mask-pngs` | Save mask frames | `false` |
| `--write-overlay-pngs` | Save overlay frames | `false` |
| `--write-heatmap-pngs` | Save heatmap frames | `false` |

### Frame Sampling

| Flag | Description | Default |
|------|-------------|---------|
| `--frame-step` | Process every Nth frame | `1` |

### Region of Interest

| Flag | Description | Default |
|------|-------------|---------|
| `--roi-margin` | Fractional crop on all sides (0-0.49) | `0.15` |
| `--roi-margin-top/bottom/left/right` | Per-side overrides | - |

### Image Processing

| Flag | Description | Default |
|------|-------------|---------|
| `--colorspace` | `CIELAB`, `RGB`, `HSV`, `GRAYSCALE` | `CIELAB` |
| `--channel-weights` | Per-channel multipliers | `1,1,1` |
| `--contrast-threshold` | Fixed threshold (0-255) | - |
| `--contrast-percentile` | Adaptive percentile (0-100) | `50` |
| `--blur-kernel` | Gaussian kernel size (odd) | `9` |

### Morphology

| Flag | Description | Default |
|------|-------------|---------|
| `--morph-kernel` | Structuring element size | `5` |
| `--morph-shape` | `ellipse`, `rect`, `cross` | `ellipse` |
| `--morph-close-iterations` | Closing iterations | `2` |
| `--morph-open-iterations` | Opening iterations | `1` |
| `--min-area` | Minimum component area (px) | `500` |

### Reference Frame Strategy

| Mode | Description |
|------|-------------|
| `--ref-mode first` | Compare to first frame |
| `--ref-mode running` | Exponential moving average |
| `--ref-mode prev N` | Compare to N frames back |
| `--ref-mode absolute N` | Compare to fixed frame index |
| `--ref-mode dynamic` | Modeled lag based on flow progression |

**Dynamic mode options:**
- `--dynamic-calibration-frames N`
- `--dynamic-target-fraction F`
- `--dynamic-lag-linear` with `--dynamic-lag-linear-start/max`

### Temporal Filtering

| Flag | Description | Default |
|------|-------------|---------|
| `--lock-frames` | Frames pixel must stay filled to persist | `3` |

### Annotation Export

| Flag | Description |
|------|-------------|
| `--annotation-formats` | `coco`, `yolov5`, `darknet` (comma-separated) |
| `--annotation-mode` | `all`, `count`, `stride` |
| `--annotation-count` | Number of frames (with `count` mode) |
| `--annotation-stride` | Frame interval (with `stride` mode) |

---

## Examples

**Basic processing with video output:**
```bash
python vrifa.py \
  --video-path input.mp4 \
  --output-dir outputs \
  --write-videos
```

**Generate ML training dataset (all formats):**
```bash
python vrifa.py \
  --video-path input.mp4 \
  --output-dir dataset \
  --annotation-formats coco,yolov5,darknet \
  --annotation-mode count \
  --annotation-count 200
```

**Dynamic reference with linear lag model:**
```bash
python vrifa.py \
  --video-path input.mp4 \
  --ref-mode dynamic \
  --dynamic-lag-linear \
  --dynamic-lag-linear-start 0 \
  --dynamic-lag-linear-max 45
```

**Custom ROI and thresholding:**
```bash
python vrifa.py \
  --video-path input.mp4 \
  --roi-margin-left 0.05 \
  --roi-margin-right 0.05 \
  --contrast-percentile 40 \
  --threshold-offset 5
```

---

## Algorithm Overview

1. **Frame extraction** from input video at specified intervals
2. **ROI cropping** to remove fixtures and edges
3. **Colorspace conversion** (default: CIELAB for perceptual uniformity)
4. **Reference frame computation** based on selected strategy
5. **Difference calculation** between current and reference frames
6. **Adaptive thresholding** using percentile-based cutoff
7. **Morphological operations** (closing, opening) to clean mask
8. **Temporal consistency filtering** to reduce transient noise
9. **Contour extraction** for annotation polygon generation
10. **Output generation** (masks, overlays, heatmaps, annotations)

---

## Reproducibility

Each run generates a `run_summary.yaml` file containing all parameters and timing information. Use this file to reproduce exact configurations.

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.
