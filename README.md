# VRIFA – Resin Infusion Flow-Front Assessment

Lightweight CLI script (`vrifa.py`) that reads an infusion video, detects the advancing resin front, and writes masks, overlays, heatmaps, optional MP4s, and machine learning annotation exports.

## Annotation Export Formats

VRIFA supports three annotation export formats for training machine learning models. Each format creates a self-contained directory structure with images and labels.

### COCO Format (`--annotation-formats coco`)

Standard COCO format for object detection and instance segmentation.

**Output Structure:**
```
output_dir/
└── formatCOCO/
    ├── annotations/
    │   └── instances_default.json    # COCO annotations with segmentation polygons
    └── images/
        └── default/
            ├── frame_000000.png
            ├── frame_000500.png
            └── ...
```

**Features:**
- Two categories: `dry` (id: 1) and `wet` (id: 2)
- Full segmentation polygon coordinates
- Bounding boxes with area calculations
- Compatible with: COCO API, Detectron2, MMDetection, CVAT import

### YOLOv5/v8 Format (`--annotation-formats yolov5`)

Ultralytics YOLO format with segmentation polygons for instance segmentation training.

**Output Structure:**
```
output_dir/
└── formatYOLO/
    ├── data.yaml          # Dataset configuration
    ├── train.txt          # List of training image paths
    ├── images/
    │   └── train/
    │       ├── frame_000000.png
    │       └── ...
    └── labels/
        └── train/
            ├── frame_000000.txt    # Segmentation labels (normalized polygons)
            └── ...
```

**Label Format:** `class_id x1 y1 x2 y2 x3 y3 ...` (normalized 0-1 polygon coordinates)

**Features:**
- Two classes: `dry` (0) and `wet` (1)
- Normalized polygon segmentation coordinates
- Compatible with: YOLOv5-seg, YOLOv8-seg, Ultralytics training

### Darknet Format (`--annotation-formats darknet`)

Classic Darknet YOLO format with bounding box annotations.

**Output Structure:**
```
output_dir/
└── formatYOLO_v2/
    ├── obj.data           # Dataset configuration
    ├── obj.names          # Class names (dry, wet)
    ├── train.txt          # List of training image paths
    └── obj_train_data/
        ├── frame_000000.png
        ├── frame_000000.txt    # Bounding box labels
        └── ...
```

**Label Format:** `class_id center_x center_y width height` (normalized 0-1 bbox coordinates)

**Features:**
- Two classes: `dry` (0) and `wet` (1)
- Normalized bounding box format
- Compatible with: Darknet, YOLOv3/v4, AlexeyAB/darknet

### Export Examples

Export all three formats with 50 evenly sampled frames:
```bash
python vrifa.py \
  --video-path input.mp4 \
  --output-dir outputs \
  --annotation-formats coco,yolov5,darknet \
  --annotation-mode count \
  --annotation-count 50
```

Export COCO format only, every 10th processed frame:
```bash
python vrifa.py \
  --video-path input.mp4 \
  --annotation-formats coco \
  --annotation-mode stride \
  --annotation-stride 10
```

---

## Prerequisites
- Python 3.10+ recommended
- FFmpeg libraries that ship with `opencv-python`

Install Python deps:
```bash
python -m pip install -r requirements.txt
# if yaml is missing on your system:
python -m pip install pyyaml
```

## Quick Start
```bash
python vrifa.py \
  --video-path input_1.mp4 \
  --output-dir flow_front_outputs \
  --write-videos \
  --write-overlay-pngs true
```
Defaults: processes every frame, crops a 15% margin on all sides, uses CIELAB colorspace, writes overlay MP4, skips mask/heatmap outputs unless enabled.

## Inputs & Outputs
- Input: RGB video file (`--video-path`, default `private_assets/input_video.mp4`).
- Outputs land under `--output-dir` (default `flow_front_outputs`):
  - `masks/`, `overlays/`, `heatmap/` for per-frame PNGs (if enabled)
  - `videos/overlay.mp4`, `mask.mp4`, `heatmap.mp4` (if enabled)
  - `formatCOCO/`, `formatYOLO/`, `formatYOLO_v2/` for ML annotation exports (if enabled)
  - `run_summary.yaml` with all run settings and timing

## Boolean Flag Syntax
Most on/off flags expect the literal strings `true/false` (case-insensitive). `--write-videos` is a pure flag that turns on all MP4 writers.

## What You Can Tune Each Run

**I/O and Export**
- `--video-path PATH` : input video.
- `--output-dir PATH` : destination root.
- `--write-videos` : enable all MP4 outputs.
- `--write-mask-video/overlay-video/heatmap-video {true,false}` : MP4 streams individually.
- `--write-mask-pngs/overlay-pngs/heatmap-pngs {true,false}` : per-frame PNGs.

**Frame Sampling**
- `--frame-step N` : process every Nth frame (reduces work and outputs).

**Region of Interest**
- `--roi-margin F` : fractional crop on all sides (0–0.49).
- Side overrides: `--roi-margin-top/bottom/left/right F`.

**Colors & Weights**
- `--colorspace {CIELAB,RGB,HSV,GRAYSCALE}`
- `--channel-weights a,b,c` : per-channel multipliers (single value repeats).

**Thresholding & Preprocessing**
- `--contrast-threshold V` : fixed 0–255 threshold.
- `--contrast-percentile P` : percentile (0–100) within ROI; overrides manual threshold.
- `--threshold-offset V` : shift after thresholding.
- `--skip-blur` : disable Gaussian smoothing.
- `--blur-kernel K` : odd kernel size (default 9).

**Morphology & Cleanup**
- `--morph-kernel K` : odd size for close/open.
- `--morph-shape {ellipse,rect,cross}`
- `--morph-close-iterations N`, `--morph-open-iterations N`
- `--min-area PX` : drop small connected components.

**Reference Frame Strategy**
- `--ref-mode first` : always compare to frame 1.
- `--ref-mode running` + `--ref-running-alpha A` : exponential moving reference.
- `--ref-mode prev N` : compare to the frame N steps back.
- `--ref-mode absolute N` : compare to a fixed frame index N.
- `--ref-mode dynamic` : modeled lag reference; knobs:
  - `--dynamic-calibration-frames N`
  - `--dynamic-target-fraction F` (ROI fraction to maintain)
  - `--dynamic-ref-cache-size N`
  - `--dynamic-lag-scale S`
  - `--dynamic-lag-linear` with `--dynamic-lag-linear-start N`, `--dynamic-lag-linear-max N`
  - `--dynamic-lag-log PATH` : CSV of frame,lag

**Glare/Noise Guard**
- `--lock-frames N` : pixel must stay filled N consecutive processed frames to “stick” (0 disables).

**Annotations (optional)**
- `--annotation-formats coco,yolov5,darknet` (see Export Formats section above)
- `--annotation-mode {all,count,stride}`
- `--annotation-count N` (used with `count`)
- `--annotation-stride N` (used with `stride`)
- `--annotation-segmentation-tolerance PX`
- `--annotation-segmentation-max-edge-length PX`

## Example Runs
Process every 5th frame, ROI tighter on left/right, overlay PNGs plus MP4:
```bash
python vrifa.py \
  --video-path input_2.mp4 \
  --output-dir runs/run02 \
  --frame-step 5 \
  --roi-margin-left 0.05 --roi-margin-right 0.05 \
  --write-overlay-pngs true \
  --write-videos
```

Dynamic reference with linear lag and all annotation formats on 50 evenly spaced frames:
```bash
python vrifa.py \
  --video-path input_1.mp4 \
  --output-dir runs/dynamic01 \
  --ref-mode dynamic \
  --dynamic-lag-linear \
  --dynamic-lag-linear-start 0 \
  --dynamic-lag-linear-max 45 \
  --annotation-formats coco,yolov5,darknet \
  --annotation-mode count \
  --annotation-count 50
```

Locking off and manual thresholding:
```bash
python vrifa.py \
  --lock-frames 0 \
  --contrast-threshold 40 \
  --threshold-offset 5 \
  --write-heatmap-video true
```

Run summaries are written to `run_summary.yaml` under your output directory for reproducibility. Use that file to copy the exact settings into future runs. 
