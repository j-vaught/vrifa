"""Render the four render-output panels for fig:heatmap_overlay_contour.

Demonstrates what the pipeline emits for one frame of input_1 (frame
352, 50 percent fill, integrated configuration).

Four panels:
  panel0_input.png      raw BGR current frame
  panel1_heatmap.png    Turbo heatmap of normalized D-tilde
  panel2_overlay.png    raw BGR frame with the locked-mask boundary
                        painted in red
  panel3_contour.png    raw BGR frame with the Douglas-Peucker-simplified
                        contour polygon drawn and vertices marked
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/user/Downloads/vrifa")
VIDEO = REPO / "data" / "input_1.mp4"
ROI_PATH = REPO / "data" / "roi_masks" / "input_1.png"
OUT_DIR = REPO / "paper" / "typst" / "figures" / "heatmap_overlay_contour_panels"
OUT_DIR.mkdir(exist_ok=True, parents=True)

CURRENT_IDX = 352
KB_POST = 9
DELTA_TAU_OFFSET = -30
KM = 13
A_MIN = 400
MORPH_ITERS = 1

# Overlay-boundary thickness and color (red in BGR).
BOUNDARY_BGR = (0, 0, 255)
BOUNDARY_THICK = 5

# Contour-render appearance.
CONTOUR_BGR = (0, 255, 255)        # yellow polygon edges
CONTOUR_THICK = 3
VERTEX_BGR = (0, 0, 255)           # red vertex markers
VERTEX_R = 8                       # vertex circle radius
DOUGLAS_PEUCKER_EPS = 4.0          # contour simplification tolerance

PANEL_W, PANEL_H = 960, 540
BG_INSIDE_ROI = (40, 40, 40)
BG_OUTSIDE_ROI = (0, 0, 0)


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


def read_bgr(idx):
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ok, f = cap.read()
    cap.release()
    return f


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


def area_filter(mask_u8, min_area):
    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8,
    )
    out = np.zeros_like(mask_u8)
    for i in range(1, n_lbl):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def fit(img):
    return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


def main():
    L = read_lstar_up_to(CURRENT_IDX)
    h, w = L.shape[1:]
    roi = load_roi(h, w)
    bgr = read_bgr(CURRENT_IDX)

    G = np.maximum.accumulate(L, axis=0)[-1]
    F = L[CURRENT_IDX]

    # Compute the cleaned mask via the integrated pipeline.
    d = np.maximum(G.astype(np.int16) - F.astype(np.int16), 0).clip(0, 255).astype(np.uint8)
    d[roi == 0] = 0
    db = cv2.GaussianBlur(d, (KB_POST, KB_POST), 0)
    db[roi == 0] = 0
    d_tilde = normalize_minmax(db, roi)

    otsu_thr, _ = cv2.threshold(d_tilde[roi > 0], 0, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thr_value = max(otsu_thr + DELTA_TAU_OFFSET, 0)
    mask = ((d_tilde > thr_value) & (roi > 0)).astype(np.uint8) * 255

    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KM, KM))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ek, iterations=MORPH_ITERS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ek, iterations=MORPH_ITERS)
    mask[roi == 0] = 0
    mask = area_filter(mask, A_MIN)
    mask[roi == 0] = 0

    # --- Panel 0: raw BGR.
    cv2.imwrite(str(OUT_DIR / "panel0_input.png"), fit(bgr))

    # --- Panel 1: heatmap of D-tilde.
    cv2.imwrite(str(OUT_DIR / "panel1_heatmap.png"),
                fit(heatmap(d_tilde, roi)))

    # --- Panel 2: overlay of mask boundary on raw frame, in red.
    boundary = cv2.morphologyEx(
        mask, cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    if BOUNDARY_THICK > 1:
        boundary = cv2.dilate(
            boundary,
            cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (BOUNDARY_THICK, BOUNDARY_THICK)),
        )
    overlay = bgr.copy()
    overlay[boundary > 0] = BOUNDARY_BGR
    cv2.imwrite(str(OUT_DIR / "panel2_overlay.png"), fit(overlay))

    # --- Panel 3: COCO-format polygon export with vertices marked.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    contour_render = bgr.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) < A_MIN:
            continue
        # Douglas-Peucker simplification — emits sparser vertex set,
        # matching the polygon export the pipeline actually writes.
        simplified = cv2.approxPolyDP(cnt, DOUGLAS_PEUCKER_EPS, True)
        cv2.polylines(contour_render, [simplified], True,
                      CONTOUR_BGR, CONTOUR_THICK, cv2.LINE_AA)
        for pt in simplified.reshape(-1, 2):
            cv2.circle(contour_render, tuple(pt), VERTEX_R,
                       VERTEX_BGR, -1, cv2.LINE_AA)
    cv2.imwrite(str(OUT_DIR / "panel3_contour.png"), fit(contour_render))

    print(f"wrote 4 panels to {OUT_DIR}")
    print(f"  mask wet fraction (vs ROI): "
          f"{(mask > 0).sum() / (roi > 0).sum() * 100:.1f}%")
    total_vertices = sum(
        cv2.approxPolyDP(c, DOUGLAS_PEUCKER_EPS, True).shape[0]
        for c in contours if cv2.contourArea(c) >= A_MIN
    )
    n_polys = sum(1 for c in contours if cv2.contourArea(c) >= A_MIN)
    print(f"  contour polygons: {n_polys}, total vertices after DP: {total_vertices}")


if __name__ == "__main__":
    main()
