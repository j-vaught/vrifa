"""Render the darken-only delta-comparison panels for fig:darken_only.

Three panels for the same frame of input_2.mp4:
  panel0_input.png       the raw current frame, with vacuum-bag
                         specular highlights visible.
  panel1_euclidean.png   the pipeline's full-color delta: the
                         channel-weighted Euclidean distance in CIELAB
                         between the reference and the current frame.
                         The specular highlights light up here because
                         Euclidean distance is sign-insensitive.
  panel2_darken.png      the integrated configuration's darken-only
                         delta: max(0, L_ref - L_cur). Specular
                         brightening is clipped to zero; only true
                         wetting (darkening) survives.

Reference frame: input_2 frame 0 (dry preform, no specular reflections).
Current frame:   input_2 frame 74 (mid-late fill, visible specular
                 highlights on the vacuum bag).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/user/Downloads/vrifa")
VIDEO = REPO / "data" / "input_2.mp4"
OUT_DIR = REPO / "paper" / "typst" / "figures" / "darken_only_panels"
OUT_DIR.mkdir(exist_ok=True, parents=True)

REF_IDX = 0
CUR_IDX = 15

# Rectangular ROI: 0% margin (input_2 and other inputs except input_1
# are processed without ROI restriction).
ROI_MARGIN = 0.0

# Output dimensions for each rendered panel.
PANEL_W, PANEL_H = 960, 540

BG_INSIDE_ROI = (40, 40, 40)
BG_OUTSIDE_ROI = (0, 0, 0)


def read_frame(idx):
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {idx} from {VIDEO}")
    return frame


def rectangular_roi(h, w, margin=ROI_MARGIN):
    mx = int(round(w * margin))
    my = int(round(h * margin))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[my:h - my, mx:w - mx] = 255
    return mask


def euclidean_delta(lab_ref, lab_cur, roi):
    """Channel-weighted Euclidean distance across all three CIELAB
    channels, sign-insensitive. Returned as uint8 in [0, 255]."""
    diff = lab_ref.astype(np.float32) - lab_cur.astype(np.float32)
    d = np.sqrt((diff ** 2).sum(axis=2))
    d = np.clip(d, 0, 255).astype(np.uint8)
    d[roi == 0] = 0
    return d


def darken_only_delta(lab_ref, lab_cur, roi):
    """max(0, L_ref - L_cur). Only catches darkening."""
    d = np.maximum(
        lab_ref[..., 0].astype(np.int16) - lab_cur[..., 0].astype(np.int16),
        0,
    )
    d = np.clip(d, 0, 255).astype(np.uint8)
    d[roi == 0] = 0
    return d


def heatmap(delta, roi, vmax):
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
    ref = read_frame(REF_IDX)
    cur = read_frame(CUR_IDX)
    h, w = cur.shape[:2]

    roi = rectangular_roi(h, w)
    lab_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
    lab_cur = cv2.cvtColor(cur, cv2.COLOR_BGR2LAB)

    delta_eu = euclidean_delta(lab_ref, lab_cur, roi)
    delta_do = darken_only_delta(lab_ref, lab_cur, roi)

    vmax = int(max(delta_eu.max(), delta_do.max(), 1))
    print(f"shared vmax: {vmax}")
    print(f"  euclidean: max={delta_eu.max()}, mean (in ROI)={delta_eu[roi > 0].mean():.2f}")
    print(f"  darken-only: max={delta_do.max()}, mean (in ROI)={delta_do[roi > 0].mean():.2f}")

    cv2.imwrite(str(OUT_DIR / "panel0_input.png"), fit(cur))
    cv2.imwrite(str(OUT_DIR / "panel1_euclidean.png"),
                fit(heatmap(delta_eu, roi, vmax)))
    cv2.imwrite(str(OUT_DIR / "panel2_darken.png"),
                fit(heatmap(delta_do, roi, vmax)))

    print(f"wrote 3 panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
