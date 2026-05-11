"""Render the four colorspace-projection panels for fig:colorspace_projection
and compute the wet-vs-dry histogram per projection inside the ROI.

Canonical frame: input_1.mp4 frame 352 (50% fill, hand-labeled).
Projections: raw BGR, CIELAB L*, HSV V, grayscale.
Single-channel projections are rendered with the Viridis perceptually
uniform colormap so wet/dry separability is visible.

Outputs:
  paper/typst/figures/colorspace_panels/col0_bgr.png         # raw BGR
  paper/typst/figures/colorspace_panels/col1_lab_l.png       # L* via viridis
  paper/typst/figures/colorspace_panels/col2_hsv_v.png       # V via viridis
  paper/typst/figures/colorspace_panels/col3_gray.png        # gray via viridis
  paper/typst/figures/colorspace_panels/hist_bgr.csv         # luminance hist, wet/dry
  paper/typst/figures/colorspace_panels/hist_lab_l.csv       # L* hist, wet/dry
  paper/typst/figures/colorspace_panels/hist_hsv_v.csv       # V hist, wet/dry
  paper/typst/figures/colorspace_panels/hist_gray.csv        # gray hist, wet/dry

CSV columns: bin_center, wet_count, dry_count (one row per histogram bin).
"""

from __future__ import annotations

import json
import csv
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/user/Downloads/vrifa")
VIDEO = REPO / "data" / "input_1.mp4"
LABELS = REPO / "data" / "labels.json"
ROI_MASK = REPO / "data" / "roi_masks" / "input_1.png"
OUT_DIR = REPO / "paper" / "typst" / "figures" / "colorspace_panels"
OUT_DIR.mkdir(exist_ok=True, parents=True)

FRAME_IDX = 352
PANEL_W, PANEL_H = 960, 540   # half of native 1920x1080
N_BINS = 48                   # histogram resolution (kept modest for CeTZ render)


def read_frame(path, idx):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {idx} from {path}")
    return frame


def gt_wet_mask(labels_path, target_file, h, w):
    """Rasterize GT polygons for the target frame into a binary wet mask."""
    labels = json.loads(labels_path.read_text())
    img_id = None
    for im in labels["images"]:
        if im["file_name"] == target_file:
            img_id = im["id"]
            break
    if img_id is None:
        raise RuntimeError(f"no labels for {target_file}")
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in labels["annotations"]:
        if ann["image_id"] != img_id:
            continue
        for seg in ann.get("segmentation", []) or []:
            pts = np.asarray(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


def apply_viridis(single_channel_u8):
    """Apply Viridis (perceptually uniform) colormap to a uint8 single channel."""
    return cv2.applyColorMap(single_channel_u8, cv2.COLORMAP_VIRIDIS)


def histogram_wet_dry(channel_u8, wet, dry, n_bins=N_BINS):
    """Return histograms restricted to wet and dry pixel sets."""
    wet_vals = channel_u8[wet > 0]
    dry_vals = channel_u8[dry > 0]
    edges = np.linspace(0, 256, n_bins + 1)
    h_wet, _ = np.histogram(wet_vals, bins=edges)
    h_dry, _ = np.histogram(dry_vals, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers, h_wet, h_dry


def save_hist_csv(path, centers, wet, dry):
    """Emit normalized densities (each distribution integrates to 1) so the
    wet and dry shapes are directly comparable regardless of pixel counts.
    Bin width is uniform, so density = count / total / bin_width with
    bin_width factored out (constant across rows) since the renderer
    rescales to a shared global max anyway.
    """
    wet_total = float(wet.sum())
    dry_total = float(dry.sum())
    wet_dens = wet / wet_total if wet_total > 0 else wet
    dry_dens = dry / dry_total if dry_total > 0 else dry
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_center", "wet_density", "dry_density",
                    "wet_count", "dry_count"])
        for c, dw, dd, cw, cd in zip(centers, wet_dens, dry_dens, wet, dry):
            w.writerow([f"{c:.2f}", f"{dw:.6f}", f"{dd:.6f}",
                        int(cw), int(cd)])


def main():
    frame = read_frame(VIDEO, FRAME_IDX)
    h, w = frame.shape[:2]

    roi = cv2.imread(str(ROI_MASK), cv2.IMREAD_GRAYSCALE)
    if roi.shape != (h, w):
        roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_NEAREST)
    roi_bin = (roi > 127).astype(np.uint8) * 255

    wet_mask = gt_wet_mask(LABELS, f"input_1__frame_000352.png", h, w)
    # Restrict wet/dry sets to inside the ROI.
    wet_in_roi = cv2.bitwise_and(wet_mask, roi_bin)
    dry_in_roi = cv2.bitwise_and(cv2.bitwise_not(wet_mask), roi_bin)

    # --- Projections ---
    bgr = frame  # column 1 stays as-is
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_star = lab[..., 0]                                # uint8 0..255
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v_chan = hsv[..., 2]                                # uint8 0..255
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # uint8 0..255

    # --- Resize for panels ---
    def fit(img):
        return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(OUT_DIR / "col0_bgr.png"),    fit(bgr))
    cv2.imwrite(str(OUT_DIR / "col1_lab_l.png"),  fit(apply_viridis(l_star)))
    cv2.imwrite(str(OUT_DIR / "col2_hsv_v.png"),  fit(apply_viridis(v_chan)))
    cv2.imwrite(str(OUT_DIR / "col3_gray.png"),   fit(apply_viridis(gray)))

    # --- Histograms (wet vs dry within ROI) ---
    # For BGR we use perceptual luminance via cvtColor BGR->GRAY (matches the
    # gray column conceptually, but the BGR column shows the original pixels).
    bgr_lum = gray
    for channel, name in (
        (bgr_lum, "hist_bgr"),
        (l_star, "hist_lab_l"),
        (v_chan, "hist_hsv_v"),
        (gray,   "hist_gray"),
    ):
        centers, hw, hd = histogram_wet_dry(channel, wet_in_roi, dry_in_roi)
        save_hist_csv(OUT_DIR / f"{name}.csv", centers, hw, hd)

    print(f"Wrote panels and histograms to {OUT_DIR}")


if __name__ == "__main__":
    main()
