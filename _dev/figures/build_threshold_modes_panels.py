"""Render the threshold-mode panels for fig:threshold_modes.

Computes the normalized response field D-tilde for input_1 frame 352
(using the integrated reference: first-frame + peak map, darken-only,
ROI applied, post-blur k_b=9 then min-max normalize), then applies six
threshold modes and saves each resulting binary mask as a PNG. Also
emits a histogram CSV of D-tilde inside the ROI plus the threshold
values for Otsu and Triangle so the Typst figure can plot the
histogram with threshold markers.

Outputs:
  paper/typst/figures/threshold_modes_panels/mask_otsu.png
  paper/typst/figures/threshold_modes_panels/mask_triangle.png
  paper/typst/figures/threshold_modes_panels/mask_manual.png
  paper/typst/figures/threshold_modes_panels/mask_percentile.png
  paper/typst/figures/threshold_modes_panels/mask_adaptive_mean.png
  paper/typst/figures/threshold_modes_panels/mask_adaptive_gaussian.png
  paper/typst/figures/threshold_modes_panels/histogram.csv
  paper/typst/figures/threshold_modes_panels/thresholds.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/user/Downloads/vrifa")
VIDEO = REPO / "data" / "input_1.mp4"
ROI_PATH = REPO / "data" / "roi_masks" / "input_1.png"
OUT_DIR = REPO / "paper" / "typst" / "figures" / "threshold_modes_panels"
OUT_DIR.mkdir(exist_ok=True, parents=True)

CURRENT_IDX = 352
KB_POST = 9                  # post-delta blur kernel size
DELTA_TAU_OFFSET = -30       # threshold offset for Otsu / Triangle / manual / percentile

# Threshold-mode parameters.
TAU_MANUAL = 64              # manual mode threshold
P_PERCENTILE = 70            # percentile mode quantile
ADAPTIVE_B = 21              # adaptive neighborhood size
ADAPTIVE_C_MEAN = 6          # subtracted constant for adaptive-mean (tuned to match Otsu coverage)
ADAPTIVE_C_GAUSSIAN = 4      # subtracted constant for adaptive-gaussian (tuned to match Otsu coverage)

PANEL_W, PANEL_H = 960, 540
N_HIST_BINS = 64

# Mask colors: white wet, dark gray dry (inside ROI), black outside ROI.
MASK_WET_BGR = (255, 255, 255)
MASK_DRY_BGR = (60, 60, 60)
MASK_OUT_BGR = (0, 0, 0)


def read_lstar_up_to(idx):
    """Read input_1 L* frames [0, idx], return as (idx+1, h, w) uint8."""
    cap = cv2.VideoCapture(str(VIDEO))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = np.zeros((idx + 1, h, w), dtype=np.uint8)
    for t in range(idx + 1):
        ok, f = cap.read()
        if not ok:
            out = out[:t]
            break
        out[t] = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[..., 0]
    cap.release()
    return out


def load_roi(h, w):
    raw = cv2.imread(str(ROI_PATH), cv2.IMREAD_GRAYSCALE)
    if raw.shape != (h, w):
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    return (raw > 127).astype(np.uint8) * 255


def normalize_minmax(field, roi):
    """Min-max normalize to [0, 255] uint8 over ROI pixels."""
    inside = field[roi > 0]
    lo, hi = float(inside.min()), float(inside.max())
    if hi <= lo:
        return np.zeros_like(field, dtype=np.uint8)
    n = (field.astype(np.float32) - lo) / (hi - lo) * 255.0
    n = np.clip(n, 0, 255).astype(np.uint8)
    n[roi == 0] = 0
    return n


def colorize_mask(binary, roi):
    """Render a binary mask as a 3-channel BGR image: wet pixels white,
    dry pixels inside ROI dark gray, outside ROI black."""
    h, w = binary.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[..., :] = MASK_DRY_BGR
    out[binary > 0] = MASK_WET_BGR
    out[roi == 0] = MASK_OUT_BGR
    return out


def fit(img):
    return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


def main():
    L = read_lstar_up_to(CURRENT_IDX)
    print(f"loaded {L.shape[0]} L* frames, {L.shape[1]}x{L.shape[2]}")
    h, w = L.shape[1:]
    roi = load_roi(h, w)

    # Reference G = running peak of L* up to current frame (integrated).
    G = np.maximum.accumulate(L, axis=0)[-1]
    F = L[CURRENT_IDX]

    # Darken-only delta, post-blur, normalize.
    delta = np.maximum(G.astype(np.int16) - F.astype(np.int16), 0)
    delta = np.clip(delta, 0, 255).astype(np.uint8)
    delta[roi == 0] = 0
    delta_blurred = cv2.GaussianBlur(delta, (KB_POST, KB_POST), 0)
    delta_blurred[roi == 0] = 0
    d_tilde = normalize_minmax(delta_blurred, roi)
    print(f"D-tilde inside ROI: min={d_tilde[roi > 0].min()}, "
          f"max={d_tilde[roi > 0].max()}, mean={d_tilde[roi > 0].mean():.2f}")

    # --- Threshold modes ---
    # Otsu + offset.
    otsu_thr, _ = cv2.threshold(d_tilde[roi > 0], 0, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    otsu_thr_with_offset = max(otsu_thr + DELTA_TAU_OFFSET, 0)
    mask_otsu = ((d_tilde > otsu_thr_with_offset) & (roi > 0)).astype(np.uint8) * 255

    # Triangle + offset.
    tri_thr, _ = cv2.threshold(d_tilde[roi > 0], 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    tri_thr_with_offset = max(tri_thr + DELTA_TAU_OFFSET, 0)
    mask_triangle = ((d_tilde > tri_thr_with_offset) & (roi > 0)).astype(np.uint8) * 255

    # Manual at TAU_MANUAL + offset.
    manual_thr_with_offset = max(TAU_MANUAL + DELTA_TAU_OFFSET, 0)
    mask_manual = ((d_tilde > manual_thr_with_offset) & (roi > 0)).astype(np.uint8) * 255

    # Percentile.
    perc_thr = float(np.percentile(d_tilde[roi > 0], P_PERCENTILE))
    mask_percentile = ((d_tilde > perc_thr) & (roi > 0)).astype(np.uint8) * 255

    # Adaptive-mean. cv2.adaptiveThreshold needs the full image; we mask
    # the result with the ROI afterwards.
    am = cv2.adaptiveThreshold(d_tilde, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY,
                               ADAPTIVE_B, ADAPTIVE_C_MEAN)
    mask_adaptive_mean = (am > 0).astype(np.uint8) * 255
    mask_adaptive_mean[roi == 0] = 0

    # Adaptive-gaussian.
    ag = cv2.adaptiveThreshold(d_tilde, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY,
                               ADAPTIVE_B, ADAPTIVE_C_GAUSSIAN)
    mask_adaptive_gaussian = (ag > 0).astype(np.uint8) * 255
    mask_adaptive_gaussian[roi == 0] = 0

    print(f"thresholds: otsu={otsu_thr:.0f}, triangle={tri_thr:.0f}, "
          f"manual_offset={manual_thr_with_offset}, percentile={perc_thr:.1f}")
    print(f"wet-pixel fractions inside ROI:")
    for name, m in (("otsu", mask_otsu),
                    ("triangle", mask_triangle),
                    ("manual", mask_manual),
                    ("percentile", mask_percentile),
                    ("adaptive-mean", mask_adaptive_mean),
                    ("adaptive-gaussian", mask_adaptive_gaussian)):
        frac = (m > 0).sum() / max((roi > 0).sum(), 1)
        print(f"  {name:18s}: {frac*100:.1f}%")

    masks = [
        ("mask_otsu.png",              mask_otsu),
        ("mask_triangle.png",          mask_triangle),
        ("mask_manual.png",            mask_manual),
        ("mask_percentile.png",        mask_percentile),
        ("mask_adaptive_mean.png",     mask_adaptive_mean),
        ("mask_adaptive_gaussian.png", mask_adaptive_gaussian),
    ]
    for name, m in masks:
        cv2.imwrite(str(OUT_DIR / name), fit(colorize_mask(m, roi)))

    # Histogram of D-tilde inside ROI (for the top strip in the figure).
    vals = d_tilde[roi > 0]
    edges = np.linspace(0, 255, N_HIST_BINS + 1)
    counts, _ = np.histogram(vals, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    with open(OUT_DIR / "histogram.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["bin_center", "count"])
        for c, n in zip(centers, counts):
            wr.writerow([f"{c:.2f}", int(n)])

    # Threshold values for annotation on the histogram strip.
    with open(OUT_DIR / "thresholds.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "value"])
        wr.writerow(["otsu_raw", f"{otsu_thr:.2f}"])
        wr.writerow(["otsu_with_offset", f"{otsu_thr_with_offset:.2f}"])
        wr.writerow(["triangle_raw", f"{tri_thr:.2f}"])
        wr.writerow(["triangle_with_offset", f"{tri_thr_with_offset:.2f}"])
        wr.writerow(["manual_with_offset", f"{manual_thr_with_offset:.2f}"])
        wr.writerow(["percentile", f"{perc_thr:.2f}"])

    print(f"wrote 6 masks + histogram.csv + thresholds.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
