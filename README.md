# VRIFA

VRIFA detects and tracks resin flow fronts in Vacuum-Assisted Resin Transfer Molding (VARTM) video. It produces binary masks, red-edge overlays, normalized heatmaps, and machine-learning annotation sets in COCO, YOLOv5, and Darknet layouts.

## What It Is

The pipeline is designed for repeatable, offline analysis of infusion video. A run converts each frame into a working colorspace, compares it against a configurable reference, thresholds the response, cleans the mask with morphology, applies temporal locking, and writes the requested artifacts to disk.

## Install / Build

Build the workspace from `vrifa-rs/`:

```bash
cd vrifa-rs
cargo build --release
```

If Cargo cannot find OpenCV on macOS with Homebrew, set:

```bash
export PKG_CONFIG_PATH="$(brew --prefix opencv)/lib/pkgconfig:$PKG_CONFIG_PATH"
```

## Quick Start

From the repository root:

```bash
./vrifa-rs/target/release/vrifa \
  --video-path data/input_1.mp4 \
  --output-dir outputs \
  --write-videos
```

## Output Structure

```
outputs/
├── videos/                  # MP4 outputs when --write-videos is enabled
├── masks/                   # PNG mask frames when requested
├── overlays/                # PNG overlay frames when requested
├── heatmap/                 # PNG heatmap frames when requested
├── formatCOCO/              # COCO annotations and extracted frames
│   ├── annotations/
│   └── images/
├── formatYOLO/              # YOLOv5-style segmentation labels
├── formatYOLO_v2/           # Darknet bounding-box labels
└── run_summary.yaml         # Serialized configuration and timing log
```

## Demo

YOLO segmentation overlay generated from VRIFA annotation output:

<video src="https://github.com/j-vaught/vrifa/releases/download/v0.1.0/yolo_overlay_input4.mp4" controls width="100%"></video>

## License

MIT. See [LICENSE](LICENSE).

## See Also

- [USAGE.md](USAGE.md) for the full CLI reference and algorithm options.
- [vrifa-rs/README.md](vrifa-rs/README.md) for workspace layout, fixtures, and benchmarks.
- [vrifa-rs/HACKING.md](vrifa-rs/HACKING.md) for change workflow and verification guidance.
