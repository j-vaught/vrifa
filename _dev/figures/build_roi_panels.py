"""Render the three ROI panels for fig:roi_crop.

Canonical frame: input_1.mp4 frame 352 (50% fill, hand-labeled).

Three panels:
  panel0_input.png   raw input frame, no overlay
  panel1_rect.png    same frame, with the rectangular ROI (15% margin)
                     rendered by cross-hatching the OUTSIDE in garnet
                     and leaving the INSIDE clear
  panel2_mask.png    same frame, with the imported PNG mask
                     (data/roi_masks/input_1.png) rendered the same way

The cross-hatch is two diagonal line families (slope +1 and slope -1)
drawn at uniform spacing on top of the frame, restricted to pixels where
the ROI mask is zero. The image inside the ROI stays untouched so the
reader can compare what the pipeline does and does not operate on.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/user/Downloads/vrifa")
VIDEO = REPO / "data" / "input_1.mp4"
IMPORTED_MASK = REPO / "data" / "roi_masks" / "input_1.png"
OUT_DIR = REPO / "paper" / "typst" / "figures" / "roi_panels"
OUT_DIR.mkdir(exist_ok=True, parents=True)

FRAME_IDX = 352
PANEL_W, PANEL_H = 960, 540
ROI_MARGIN = 0.15

# Garnet in OpenCV's BGR order. Hex #73000A = (R=115, G=0, B=10).
GARNET_BGR = (10, 0, 115)

# Hatch parameters at native (1920x1080) resolution.
HATCH_SPACING = 28      # pixels between adjacent diagonals
HATCH_THICKNESS = 3     # line thickness in pixels


def read_frame(path, idx):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {idx} from {path}")
    return frame


def rectangular_mask(h, w, margin):
    """Binary mask: 255 inside the rectangle bounded by `margin` fractional
    margins on every edge, 0 outside.
    """
    mx = int(round(w * margin))
    my = int(round(h * margin))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[my:h - my, mx:w - mx] = 255
    return mask


def imported_mask(path, h, w):
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"could not read mask {path}")
    if raw.shape != (h, w):
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    return (raw > 127).astype(np.uint8) * 255


def build_hatch(h, w, spacing=HATCH_SPACING, thickness=HATCH_THICKNESS):
    """Render a binary cross-hatch over an HxW canvas. Pixels under either
    of the two diagonal line families are 255.
    """
    hatch = np.zeros((h, w), dtype=np.uint8)
    # Slope +1 (forward slashes): y = x - off, swept over off.
    for off in range(-h, w + 1, spacing):
        cv2.line(hatch, (off, 0), (off + h, h), 255, thickness, cv2.LINE_AA)
    # Slope -1 (back slashes): y = -x + off.
    for off in range(0, w + h + 1, spacing):
        cv2.line(hatch, (off, 0), (off - h, h), 255, thickness, cv2.LINE_AA)
    return hatch


def overlay_outside_hatch(frame, roi_mask, hatch=None):
    """Return a copy of `frame` with the cross-hatch painted in garnet over
    every pixel where roi_mask == 0. Pixels inside the ROI are untouched.
    """
    h, w = frame.shape[:2]
    if hatch is None:
        hatch = build_hatch(h, w)
    outside = roi_mask == 0
    paint = (hatch > 0) & outside
    out = frame.copy()
    out[paint] = GARNET_BGR
    return out


def fit(img):
    return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


def main():
    frame = read_frame(VIDEO, FRAME_IDX)
    h, w = frame.shape[:2]

    # Panel 0: raw input.
    cv2.imwrite(str(OUT_DIR / "panel0_input.png"), fit(frame))

    # Shared hatch at native resolution.
    hatch = build_hatch(h, w)

    # Panel 1: rectangular ROI with 15% margins.
    rect = rectangular_mask(h, w, ROI_MARGIN)
    cv2.imwrite(
        str(OUT_DIR / "panel1_rect.png"),
        fit(overlay_outside_hatch(frame, rect, hatch=hatch)),
    )

    # Panel 2: imported PNG mask for input_1.
    imp = imported_mask(IMPORTED_MASK, h, w)
    cv2.imwrite(
        str(OUT_DIR / "panel2_mask.png"),
        fit(overlay_outside_hatch(frame, imp, hatch=hatch)),
    )

    print(f"Wrote 3 ROI panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
