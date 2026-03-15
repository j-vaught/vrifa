# VRIFA: VARTM Resin Infusion Flow-Front Assessment

A computer vision tool for automated detection and tracking of resin flow fronts in Vacuum-Assisted Resin Transfer Molding (VARTM) processes. VRIFA processes video recordings of composite manufacturing to generate binary masks, visual overlays, temporal heatmaps, and machine learning training datasets.

## Demo

YOLO object detection overlay on a VARTM infusion, trained on VRIFA-generated annotations:

<video src="https://github.com/j-vaught/vrifa/releases/download/v0.1.0/yolo_overlay_input4.mp4" controls width="100%"></video>

## Features

- Darken-only detection to track resin wetting (ignores brightening artifacts)
- Peak brightness reference for handling variable lighting conditions
- Flow-front detection using adaptive Otsu thresholding with configurable offset
- Multiple reference frame strategies (first frame, running average, dynamic lag)
- Temporal consistency filtering to reduce noise and glare artifacts
- ML annotation export in COCO, YOLOv5, and Darknet formats
- Configurable ROI cropping to focus on relevant regions

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/j-vaught/vrifa.git
cd vrifa
pip install -r requirements.txt
```

## Quick Start

```bash
python vrifa.py --video-path data/input_1.mp4 --output-dir outputs --write-videos
```

This detects darkening pixels (resin wetting) compared to their peak brightness, applies adaptive thresholding, exports COCO annotations, and generates overlay, mask, and heatmap videos.

See [USAGE.md](USAGE.md) for the full configuration reference, detection modes, annotation formats, examples, and troubleshooting.

## Output Structure

```
outputs/
├── videos/             # MP4 outputs (overlay, mask, heatmap)
├── formatCOCO/         # COCO annotations (default)
│   ├── annotations/    # JSON annotation files
│   └── images/         # Extracted frames
├── formatYOLO/         # YOLOv5 annotations (if enabled)
├── formatYOLO_v2/      # Darknet annotations (if enabled)
└── run_summary.yaml    # Configuration and timing log
```

## Sample Data

The `data/` folder contains input videos used for the example runs. The `outputs_run*/` folders contain COCO annotations and run summaries, but not the extracted image frames (to keep the repository size manageable).

To regenerate frames from an input video:

```bash
ffmpeg -i data/input_1.mp4 frames/frame_%06d.png
```

Or run VRIFA directly to produce both frames and annotations:

```bash
python vrifa.py --video-path data/input_1.mp4 --output-dir outputs_run --annotation-formats coco
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.
