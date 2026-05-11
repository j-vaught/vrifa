"""Render panels for fig:cleanup (mask cleanup montage) and
fig:morph_kernels (structuring-element comparison), both on input_1
frame 200.

fig:cleanup (5 panels):
  panel0_response.png     normalized response field D-tilde (Turbo)
  panel1_threshold.png    binary mask from Otsu + integrated offset
  panel2_closed.png       after morphological closing
  panel3_opened.png       after morphological opening
  panel4_area_filter.png  after connected-components area filter

fig:morph_kernels (3 panels):
  morph_ellipse.png   cleaned mask using elliptical SE (integrated)
  morph_rect.png      cleaned mask using rectangular SE
  morph_cross.png     cleaned mask using cross-shaped SE
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/user/Downloads/vrifa")
VIDEO = REPO / "data" / "input_1.mp4"
ROI_PATH = REPO / "data" / "roi_masks" / "input_1.png"
OUT_CLEAN = REPO / "paper" / "typst" / "figures" / "mask_cleanup_panels"
OUT_MORPH = REPO / "paper" / "typst" / "figures" / "morph_kernels_panels"
OUT_CLEAN.mkdir(exist_ok=True, parents=True)
OUT_MORPH.mkdir(exist_ok=True, parents=True)

CURRENT_IDX = 200
KB_POST = 9
DELTA_TAU_OFFSET = -30
KM = 13                       # morphology structuring-element size
A_MIN = 400                   # connected-component area floor
MORPH_ITERS = 1

PANEL_W, PANEL_H = 960, 540
BG_INSIDE_ROI = (40, 40, 40)
BG_OUTSIDE_ROI = (0, 0, 0)
MASK_WET_BGR = (255, 255, 255)
MASK_DRY_BGR = (60, 60, 60)
MASK_OUT_BGR = (0, 0, 0)


def read_lstar_up_to(idx):
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
    inside = field[roi > 0]
    lo, hi = float(inside.min()), float(inside.max())
    if hi <= lo:
        return np.zeros_like(field, dtype=np.uint8)
    n = (field.astype(np.float32) - lo) / (hi - lo) * 255.0
    n = np.clip(n, 0, 255).astype(np.uint8)
    n[roi == 0] = 0
    return n


def heatmap(delta, roi):
    hm = cv2.applyColorMap(delta, cv2.COLORMAP_TURBO)
    hm[delta == 0] = BG_INSIDE_ROI
    hm[roi == 0] = BG_OUTSIDE_ROI
    return hm


def colorize_mask(binary, roi):
    h, w = binary.shape
    out = np.full((h, w, 3), MASK_DRY_BGR, dtype=np.uint8)
    out[binary > 0] = MASK_WET_BGR
    out[roi == 0] = MASK_OUT_BGR
    return out


def fit(img):
    return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


def kernel(shape, k):
    return cv2.getStructuringElement(shape, (k, k))


def area_filter(mask_u8, min_area):
    """Drop connected components below min_area pixels."""
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8,
    )
    out = np.zeros_like(mask_u8)
    for i in range(1, n_lbl):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def main():
    L = read_lstar_up_to(CURRENT_IDX)
    h, w = L.shape[1:]
    roi = load_roi(h, w)
    G = np.maximum.accumulate(L, axis=0)[-1]
    F = L[CURRENT_IDX]

    # Darken-only delta on L*, ROI, post-blur, normalize.
    d = np.maximum(G.astype(np.int16) - F.astype(np.int16), 0).clip(0, 255).astype(np.uint8)
    d[roi == 0] = 0
    db = cv2.GaussianBlur(d, (KB_POST, KB_POST), 0)
    db[roi == 0] = 0
    d_tilde = normalize_minmax(db, roi)

    # --- fig:cleanup pipeline ---
    # Panel 0: normalized response (D-tilde as Turbo heatmap).
    cv2.imwrite(str(OUT_CLEAN / "panel0_response.png"),
                fit(heatmap(d_tilde, roi)))

    # Panel 1: thresholded binary mask via Otsu + integrated offset.
    otsu_thr, _ = cv2.threshold(d_tilde[roi > 0], 0, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thr_value = max(otsu_thr + DELTA_TAU_OFFSET, 0)
    mask_thr = ((d_tilde > thr_value) & (roi > 0)).astype(np.uint8) * 255
    cv2.imwrite(str(OUT_CLEAN / "panel1_threshold.png"),
                fit(colorize_mask(mask_thr, roi)))

    # Panel 2: after morphological closing (elliptical, KM x KM).
    ellipse_k = kernel(cv2.MORPH_ELLIPSE, KM)
    mask_closed = cv2.morphologyEx(mask_thr, cv2.MORPH_CLOSE, ellipse_k,
                                    iterations=MORPH_ITERS)
    mask_closed[roi == 0] = 0
    cv2.imwrite(str(OUT_CLEAN / "panel2_closed.png"),
                fit(colorize_mask(mask_closed, roi)))

    # Panel 3: after morphological opening.
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, ellipse_k,
                                    iterations=MORPH_ITERS)
    mask_opened[roi == 0] = 0
    cv2.imwrite(str(OUT_CLEAN / "panel3_opened.png"),
                fit(colorize_mask(mask_opened, roi)))

    # Panel 4: after connected-components area filter.
    mask_filtered = area_filter(mask_opened, A_MIN)
    mask_filtered[roi == 0] = 0
    cv2.imwrite(str(OUT_CLEAN / "panel4_area_filter.png"),
                fit(colorize_mask(mask_filtered, roi)))

    print("fig:cleanup wet-pixel fractions (vs ROI):")
    for name, m in (("threshold", mask_thr),
                    ("closed",    mask_closed),
                    ("opened",    mask_opened),
                    ("area-filt", mask_filtered)):
        frac = (m > 0).sum() / max((roi > 0).sum(), 1)
        print(f"  {name:9s}: {frac * 100:.1f}%")

    # --- fig:morph_kernels: cleaned mask under three SE shapes ---
    shapes = (
        ("morph_ellipse.png", cv2.MORPH_ELLIPSE),
        ("morph_rect.png",    cv2.MORPH_RECT),
        ("morph_cross.png",   cv2.MORPH_CROSS),
    )
    print("\nfig:morph_kernels wet-pixel fractions:")
    for name, shape in shapes:
        k = kernel(shape, KM)
        closed = cv2.morphologyEx(mask_thr, cv2.MORPH_CLOSE, k,
                                   iterations=MORPH_ITERS)
        closed[roi == 0] = 0
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k,
                                   iterations=MORPH_ITERS)
        opened[roi == 0] = 0
        filtered = area_filter(opened, A_MIN)
        filtered[roi == 0] = 0
        cv2.imwrite(str(OUT_MORPH / name),
                    fit(colorize_mask(filtered, roi)))
        frac = (filtered > 0).sum() / max((roi > 0).sum(), 1)
        print(f"  {name:18s}: {frac * 100:.1f}%")

    print(f"\nwrote 5 cleanup panels to {OUT_CLEAN}")
    print(f"wrote 3 morph-kernel panels to {OUT_MORPH}")


if __name__ == "__main__":
    main()
