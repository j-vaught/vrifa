"""Render the pre/post-blur panels for fig:pre_post_blur.

Uses the input_1 bump event (frame 10 reference, frame 75 current) with
camera-shift stabilization deliberately TURNED OFF. The ~3.7 px residual
shift produces high-frequency edge artifacts on every laminate edge,
which is exactly the regime where pre- and post-delta blurring matter.

Six panels in a 2-row by 3-column grid:

   row 1 (working channel after pre-delta blur):
     k_p = 0  (integrated, no pre-blur)        col 0
     k_p = 5                                    col 1
     k_p = 9                                    col 2

   row 2 (resulting delta D_t after the same pre-blur, then a Gaussian
          post-blur with k_b = 9):
     k_p = 0  (integrated)                      col 0
     k_p = 5                                    col 1
     k_p = 9                                    col 2

The two integrated-configuration panels (col 0 on both rows) get a
garnet outline in the Typst figure.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/user/Downloads/vrifa")
VIDEO = REPO / "data" / "input_1.mp4"
OUT_DIR = REPO / "paper" / "typst" / "figures" / "pre_post_blur_panels"
OUT_DIR.mkdir(exist_ok=True, parents=True)

REF_IDX = 10
CUR_IDX = 75

KP_LEVELS = (0, 5, 9)       # pre-delta blur kernel sizes
KB = 9                       # post-delta blur kernel size

PANEL_W, PANEL_H = 960, 540

BG_INSIDE_ROI = (40, 40, 40)
BG_OUTSIDE_ROI = (0, 0, 0)


def read_frame(idx):
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ok, f = cap.read()
    cap.release()
    return f


def gaussian_blur(channel_u8, k):
    if k <= 0:
        return channel_u8.copy()
    k = k if k % 2 == 1 else k + 1
    return cv2.GaussianBlur(channel_u8, (k, k), 0)


def fit(img):
    return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


def heatmap(delta, vmax):
    scaled = np.clip(
        delta.astype(np.float32) * 255.0 / max(vmax, 1), 0, 255,
    ).astype(np.uint8)
    hm = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    hm[delta == 0] = BG_INSIDE_ROI
    return hm


def main():
    ref = read_frame(REF_IDX)
    cur = read_frame(CUR_IDX)
    L_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)[..., 0]
    L_cur = cv2.cvtColor(cur, cv2.COLOR_BGR2LAB)[..., 0]

    # For each pre-blur level, compute L* after pre-blur (current frame
    # only — that's the "working channel after pre-delta blur" panel),
    # the delta, and the post-blurred delta.
    L_curs = []
    deltas = []
    for kp in KP_LEVELS:
        L_ref_b = gaussian_blur(L_ref, kp)
        L_cur_b = gaussian_blur(L_cur, kp)
        L_curs.append(L_cur_b)

        d = np.maximum(L_ref_b.astype(np.int16) - L_cur_b.astype(np.int16), 0)
        d = np.clip(d, 0, 255).astype(np.uint8)
        d_post = gaussian_blur(d, KB)
        deltas.append(d_post)

    vmax = int(max(d.max() for d in deltas))
    print(f"shared vmax: {vmax}")
    for kp, d in zip(KP_LEVELS, deltas):
        print(f"  k_p={kp}: max={d.max()}, mean={d.mean():.2f}")

    # Row 1: working channel after pre-blur (grayscale).
    for kp, l in zip(KP_LEVELS, L_curs):
        cv2.imwrite(str(OUT_DIR / f"row0_kp{kp}.png"), fit(l))

    # Row 2: resulting delta after pre-blur + post-blur (Turbo heatmap).
    for kp, d in zip(KP_LEVELS, deltas):
        cv2.imwrite(str(OUT_DIR / f"row1_kp{kp}.png"), fit(heatmap(d, vmax)))

    print(f"wrote 6 panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
