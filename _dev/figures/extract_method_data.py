#!/usr/bin/env python3
"""Generate data and image assets for the Method-section diagrams.

Produces:
  paper/assets/method/peak_reference.csv      -- L* time series for one tracked pixel
  paper/assets/method/darken_naive.png        -- naive Euclidean delta heatmap on frame 80
  paper/assets/method/darken_only.png         -- darken-only delta heatmap on frame 80
  paper/assets/method/darken_raw.png          -- raw input frame 80
  paper/assets/method/cleanup_<stage>.png     -- five mask-cleanup intermediates on frame 200

Run from the repo root.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO = REPO_ROOT / "data" / "input_1.mp4"
DUMPS = Path("/tmp/vrifa_method_run2/dumps")
OUT = REPO_ROOT / "paper" / "assets" / "method"

OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Peak-reference time series. Sample the L* channel at one pixel that begins
# bright (dry fabric) and ends dark (resin-saturated). Scan the whole video
# frame-by-frame and record raw L* alongside the running maximum.
# ---------------------------------------------------------------------------

# Pixel chosen by inspection: roughly central, slightly off-axis toward inlet,
# in coordinates valid for input_1's 1920x1080 frame.
TRACKED_PX = (760, 500)  # (x, y)

cap = cv2.VideoCapture(str(VIDEO))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
peak = None
rows: list[dict[str, float]] = []
for idx in range(total):
    ok, frame_bgr = cap.read()
    if not ok:
        break
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = float(lab[TRACKED_PX[1], TRACKED_PX[0], 0])
    if peak is None:
        peak = L
    else:
        peak = max(peak, L)
    rows.append({
        "frame": idx,
        "time_s": idx / max(fps, 1.0),
        "L_raw": L,
        "L_peak": peak,
    })
cap.release()
peak_csv = OUT / "peak_reference.csv"
with peak_csv.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["frame", "time_s", "L_raw", "L_peak"])
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {peak_csv} ({len(rows)} rows; pixel {TRACKED_PX})")

# ---------------------------------------------------------------------------
# Darken-only vs naive Euclidean delta on frame 80. Both deltas use the
# committed first-frame reference. Naive Euclidean is computed across all
# three CIELAB channels; darken-only is computed on the L* channel with
# negative differences clipped to zero.
# ---------------------------------------------------------------------------

ref_lab = np.load(DUMPS / "frame_000001" / "frame_converted.npy").astype(np.float32)
cur_lab = np.load(DUMPS / "frame_000080" / "frame_converted.npy").astype(np.float32)

naive = np.sqrt(np.sum((cur_lab - ref_lab) ** 2, axis=2))
darken = np.maximum(0.0, ref_lab[:, :, 0] - cur_lab[:, :, 0])


def normalize_to_u8(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo = float(a.min())
    hi = float(a.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


# Re-load raw BGR for frame 80 (the dump is in CIELAB, not BGR).
cap = cv2.VideoCapture(str(VIDEO))
cap.set(cv2.CAP_PROP_POS_FRAMES, 79)  # 0-indexed; frame_000080 is the 80th
ok, raw_bgr = cap.read()
cap.release()
assert ok
cv2.imwrite(str(OUT / "darken_raw.png"), raw_bgr)

naive_u8 = normalize_to_u8(naive)
darken_u8 = normalize_to_u8(darken)
naive_color = cv2.applyColorMap(naive_u8, cv2.COLORMAP_TURBO)
darken_color = cv2.applyColorMap(darken_u8, cv2.COLORMAP_TURBO)
cv2.imwrite(str(OUT / "darken_naive.png"), naive_color)
cv2.imwrite(str(OUT / "darken_only.png"), darken_color)
print(f"wrote {OUT}/darken_{{raw,naive,only}}.png")

# ---------------------------------------------------------------------------
# Mask-cleanup sequence on frame 200. Five thumbnails:
#   delta_norm  (post min-max normalization, before threshold)
#   binary      (post Otsu threshold + offset, pre morphology)
#   close       (synthesized: morph close kernel applied to binary)
#   open        (synthesized: morph open kernel applied after close)
#   final       (final mask post min-area filter, equals the dumped 'mask')
# We compute close/open ourselves because the Rust binary dumps only the
# inputs and outputs of the morphology stage, not the intermediates.
# ---------------------------------------------------------------------------

dn = np.load(DUMPS / "frame_000200" / "delta_norm.npy")
binary = np.load(DUMPS / "frame_000200" / "binary.npy")
mask = np.load(DUMPS / "frame_000200" / "mask.npy")

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

# Save the raw input as a heatmap for the first cell, then four binary masks.
cv2.imwrite(str(OUT / "cleanup_1_delta_norm.png"),
            cv2.applyColorMap(dn.astype(np.uint8), cv2.COLORMAP_TURBO))
cv2.imwrite(str(OUT / "cleanup_2_binary.png"), binary.astype(np.uint8))
cv2.imwrite(str(OUT / "cleanup_3_close.png"), closed.astype(np.uint8))
cv2.imwrite(str(OUT / "cleanup_4_open.png"), opened.astype(np.uint8))
cv2.imwrite(str(OUT / "cleanup_5_final.png"), mask.astype(np.uint8))
print(f"wrote {OUT}/cleanup_*.png")

# Also save a couple of 'real' frame thumbs for the montage figure since
# they will be useful regardless of the labeling pass timing.
for fidx in (50, 200, 500):
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ok, frame = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(REPO_ROOT / "paper" / "assets" / f"input_1_frame_{fidx:06d}.png"), frame)

print("done")
