# VARTM Resin Infusion Flow‑Front Assessment Algorithm (VRIFA)

The VARTM Resin Infusion Flow‑Front Assessment Algorithm detects and visualizes the advancing resin flow‑front in VARTM videos. It compares each frame to the dry reference, builds a contrast map, and outputs per‑frame masks, red‑edge overlays, heatmaps, and videos(optional).

## 1) Description
- Purpose: map the moving flow-front from a single RGB video captured during infusion.
- Approach: contrast-to-reference in CIELAB + smoothing + morphology + visualization.
- Footprint: single Python script (CPU), writes PNGs and optional MP4s.

## 2) Usage & Setup
### Setup
1. Install Python 3.9+.
2. (Recommended) create a virtual environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Basic usage
```bash
python vrifa.py \
  --video-path docs/assets/demo_input.mp4 \
  --output-dir demo_run \
  --frame-step 1 \
  --roi-margin 0.15 \
  --write-videos
```

### Variations
- Faster pass (sample every 3rd frame):
  ```bash
  python vrifa.py --video-path <your.mp4> --output-dir fast --frame-step 3
  ```
- Tighter ROI and stronger smoothing:
  ```bash
  python vrifa.py --video-path <your.mp4> --output-dir tuned \
    --roi-margin 0.10 --blur-kernel 11 --morph-kernel 15 --min-area 600
  ```
- No videos (PNGs only): omit `--write-videos`.

## 3) Outputs & Samples
On each run the tool creates:
- `<out>/masks/` – binary mask of the detected region.
- `<out>/overlays/` – input frame with the front edge in red.
- `<out>/heatmap/` – Turbo colormap of the normalized contrast.
- `<out>/videos/` – optional MP4s when `--write-videos` is set.

### Sample input frames
Dry to near-complete infusion snapshots:

![Dry 0%](docs/assets/input_frames/000_dry.png)
![15%](docs/assets/input_frames/015_early.png)
![30%](docs/assets/input_frames/030_earlymid.png)
![50%](docs/assets/input_frames/050_mid.png)
![70%](docs/assets/input_frames/070_late.png)
![85%](docs/assets/input_frames/085_nearcomplete.png)

### Sample outputs
Overlay mid-infusion and near-complete:

![Overlay mid](docs/assets/sample_outputs/overlay_mid.png)
![Overlay near-complete](docs/assets/sample_outputs/overlay_nearcomplete.png)

## 4) Algorithm Explanation (10 Phases)
The 10‑phase process below uses a representative frame (≈30% early‑mid progression):

1. Original RGB frame
   ![01 original](docs/assets/algorithm_10phase/01_original.png)
2. Raw contrast to dry reference (CIELAB delta)
   ![02 raw](docs/assets/algorithm_10phase/02_raw_contrast.png)
3. Gaussian blur on contrast map (`--blur-kernel`)
   ![03 blur](docs/assets/algorithm_10phase/03_blurred_contrast.png)
4. Otsu threshold → binary segmentation
   ![04 binary](docs/assets/algorithm_10phase/04_otsu_binary.png)
5. Morphological close (fill small gaps)
   ![05 close](docs/assets/algorithm_10phase/05_morph_close.png)
6. Morphological open (remove speckle)
   ![06 open](docs/assets/algorithm_10phase/06_morph_open.png)
7. Small-component filtering (`--min-area`)
   ![07 filtered](docs/assets/algorithm_10phase/07_small_components_removed.png)
8. Edge gradient of the mask (front border)
   ![08 edge](docs/assets/algorithm_10phase/08_edge_gradient.png)
9. Red-edge overlay on original RGB
   ![09 overlay](docs/assets/algorithm_10phase/09_overlay.png)
10. Heatmap of normalized contrast (context)
   ![10 heatmap](docs/assets/algorithm_10phase/10_heatmap.png)

Notes:
- The dry frame (first video frame) is the reference for all later frames.
- ROI cropping is applied internally based on `--roi-margin`.
- Parameters allow trading sensitivity vs. stability across capture conditions.
