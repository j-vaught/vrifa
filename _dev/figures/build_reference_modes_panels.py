"""Render the six reference-mode panels for fig:reference_modes.

Each panel shows the same current frame (input_1 frame 352, 50% fill)
compared against a different choice of reference image $G_t$ via the
pipeline's darken-only delta $D_t = max(0, G_t - F_t)$ on the CIELAB
$L^*$ channel, restricted to the ROI, rendered as a Turbo heatmap on a
shared intensity scale.

Panel order matches fig:reference_modes:
  Top row    first-frame + peak (integrated),  running EMA,         previous fixed-offset
  Bottom row absolute pinned frame,            dynamic sqrt-area,   dynamic linear-lag

Outputs:
  paper/typst/figures/reference_modes_panels/panel0_integrated.png
  paper/typst/figures/reference_modes_panels/panel1_running.png
  paper/typst/figures/reference_modes_panels/panel2_previous.png
  paper/typst/figures/reference_modes_panels/panel3_absolute.png
  paper/typst/figures/reference_modes_panels/panel4_dynamic_sqrt.png
  paper/typst/figures/reference_modes_panels/panel5_dynamic_linear.png
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/user/Downloads/vrifa")
VIDEO = REPO / "data" / "input_1.mp4"
ROI_PATH = REPO / "data" / "roi_masks" / "input_1.png"
OUT_DIR = REPO / "paper" / "typst" / "figures" / "reference_modes_panels"
OUT_DIR.mkdir(exist_ok=True, parents=True)

CURRENT_IDX = 352

# Per-mode reference frame choices (or special handling).
EMA_ALPHA = 0.05
PREV_K = 30                # previous-frame fixed offset
ABSOLUTE_REF = 100         # absolute-mode pinned frame
DYN_SQRT_REF = 120         # approximates dynamic sqrt-area lag at t=352, rho=0.2
DYN_LINEAR_REF = 252       # approximates dynamic linear-lag at t=352, lag=100

PANEL_W, PANEL_H = 960, 540
BG_INSIDE_ROI = (40, 40, 40)
BG_OUTSIDE_ROI = (0, 0, 0)


def read_all_lstar():
    """Read every frame's CIELAB L* channel into a (N, H, W) uint8 array."""
    cap = cv2.VideoCapture(str(VIDEO))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = np.zeros((n, h, w), dtype=np.uint8)
    for t in range(n):
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


def delta_field(reference_l, current_l, roi):
    """max(0, reference - current) on L*, restricted to ROI."""
    d = np.maximum(
        reference_l.astype(np.int16) - current_l.astype(np.int16), 0,
    )
    d = np.clip(d, 0, 255).astype(np.uint8)
    d[roi == 0] = 0
    return d


def heatmap(delta, roi, vmax):
    """Render the delta field as a Turbo heatmap on a shared vmax."""
    scaled = np.clip(
        delta.astype(np.float32) * 255.0 / max(vmax, 1), 0, 255,
    ).astype(np.uint8)
    hm = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    hm[delta == 0] = BG_INSIDE_ROI
    hm[roi == 0] = BG_OUTSIDE_ROI
    return hm


def fit(img):
    return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


def main():
    L = read_all_lstar()
    n, h, w = L.shape
    print(f"loaded {n} L* frames, {h}x{w}")
    roi = load_roi(h, w)
    F = L[CURRENT_IDX]

    # Reference 1: first-frame mode + peak map (integrated).
    G1 = np.maximum.accumulate(L[: CURRENT_IDX + 1], axis=0)[-1]

    # Reference 2: running EMA up through CURRENT_IDX.
    G2 = L[0].astype(np.float32)
    for t in range(1, CURRENT_IDX + 1):
        G2 = (1.0 - EMA_ALPHA) * G2 + EMA_ALPHA * L[t].astype(np.float32)
    G2 = np.clip(G2, 0, 255).astype(np.uint8)

    # Reference 3: previous fixed-offset.
    G3 = L[CURRENT_IDX - PREV_K]

    # Reference 4: absolute pinned frame.
    G4 = L[ABSOLUTE_REF]

    # Reference 5: dynamic sqrt-area approximation.
    G5 = L[DYN_SQRT_REF]

    # Reference 6: dynamic linear-lag approximation.
    G6 = L[DYN_LINEAR_REF]

    refs = [
        ("panel0_integrated.png",       G1),
        ("panel1_running.png",          G2),
        ("panel2_previous.png",         G3),
        ("panel3_absolute.png",         G4),
        ("panel4_dynamic_sqrt.png",     G5),
        ("panel5_dynamic_linear.png",   G6),
    ]

    deltas = [delta_field(G, F, roi) for _, G in refs]
    vmax = int(max(d.max() for d in deltas))
    print(f"shared vmax: {vmax}")
    for (name, _), d in zip(refs, deltas):
        print(f"  {name}: max={d.max()}, mean (in ROI)={d[roi > 0].mean():.2f}")

    for (name, _), d in zip(refs, deltas):
        cv2.imwrite(str(OUT_DIR / name), fit(heatmap(d, roi, vmax)))

    print(f"wrote 6 panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
