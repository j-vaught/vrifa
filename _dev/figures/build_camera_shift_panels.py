"""Render the camera-shift panels for fig:camera_shift_pair.

Demonstrates the camera-shift registration stage on the real bump event
in input_1.mp4 between frames ~65 and ~70 (gradual ~3.7 px drift that
settles by frame 75).

Three panels:
  panel0_context.png   Full frame at pre-event (frame 63) with a red
                       rectangle marking the zoom region examined below.
  panel1_uncorrected.png  Zoomed red/green overlay of pre vs post-event
                       (post = frame 75). Red shows the pre-event frame,
                       green shows the post-event frame. Where they
                       align: yellow. Where they don't: red and green
                       fringes show the shift directly.
  panel2_corrected.png Same zoomed overlay after applying the
                       registration warp to the post-event frame. The
                       red/green fringes collapse back into yellow,
                       showing that the pipeline brings the live frame
                       back into the reference coordinate system.

The registration warp uses cv2.phaseCorrelate (the same primitive the
Rust pipeline invokes) to estimate the integer-plus-subpixel translation
between the two frames, then cv2.warpAffine to apply the inverse.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/user/Downloads/vrifa")
VIDEO = REPO / "data" / "input_1.mp4"
OUT_DIR = REPO / "paper" / "typst" / "figures" / "camera_shift_panels"
OUT_DIR.mkdir(exist_ok=True, parents=True)

PRE_IDX = 63        # well before the bump (frames 65-70)
POST_IDX = 75       # after the bump has settled

# Crop region for the zoom panels — a small, tightly framed window over
# a high-contrast static edge (laminate corner) so the 3-4 pixel shift
# is large relative to the crop and the red/green edge fringes are
# obvious. 16:9 aspect to match the context-panel display.
CROP_X0, CROP_Y0 = 410, 720
CROP_W,  CROP_H  = 320, 180

# Output dimensions for each rendered panel.
CONTEXT_W, CONTEXT_H = 960, 540
ZOOM_W,    ZOOM_H    = 960, 540

# Zoom-region marker drawn on the context panel.
MARKER_BGR  = (0, 0, 255)        # bright red
MARKER_THICK = 6


def read_frame(idx):
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {idx}")
    return frame


def phase_translation(pre_gray, post_gray):
    """Return the (dx, dy) translation that brings post into pre, via
    cv2.phaseCorrelate. dx is x-shift, dy is y-shift.
    """
    return cv2.phaseCorrelate(
        pre_gray.astype(np.float32),
        post_gray.astype(np.float32),
    )[0]


def warp_translation(img, dx, dy):
    """Translate img by (dx, dy) via affine warp. Positive dx moves the
    image content rightward. To correct a post-event frame whose
    content has shifted by (dx, dy) relative to pre, we apply the
    inverse translation (-dx, -dy).
    """
    h, w = img.shape[:2]
    M = np.array([[1.0, 0.0, dx],
                  [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def edge_magnitude(bgr):
    """Sobel gradient magnitude on a BGR frame, returned as uint8."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag * 0.8, 0, 255).astype(np.uint8)
    return mag


def red_green_overlay(pre_bgr, post_bgr, dark_bg_value=24):
    """Build a single image where the pre-frame Sobel edges fill the red
    channel and the post-frame Sobel edges fill the green channel,
    layered over a dark gray background. Aligned edges show as bright
    yellow; misaligned edges break out into pure red and pure green
    parallel lines, making a 3-4 pixel translation visible at a glance.
    """
    pre_edges  = edge_magnitude(pre_bgr)
    post_edges = edge_magnitude(post_bgr)
    h, w = pre_edges.shape
    overlay = np.full((h, w, 3), dark_bg_value, dtype=np.uint8)
    overlay[..., 2] = np.maximum(overlay[..., 2], pre_edges)   # R
    overlay[..., 1] = np.maximum(overlay[..., 1], post_edges)  # G
    return overlay


def fit(img, target_w, target_h):
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def main():
    pre  = read_frame(PRE_IDX)
    post = read_frame(POST_IDX)

    # Estimate the shift on the full frame, in grayscale.
    pre_gray  = cv2.cvtColor(pre,  cv2.COLOR_BGR2GRAY)
    post_gray = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)
    dx, dy = phase_translation(pre_gray, post_gray)
    print(f"detected shift post relative to pre: dx={dx:.2f} px, dy={dy:.2f} px")

    # Apply the inverse shift to post to bring it back into pre coords.
    post_corrected = warp_translation(post, -dx, -dy)

    # --- Panel 0: full pre frame with the zoom rectangle marked. ---
    context = pre.copy()
    cv2.rectangle(
        context,
        (CROP_X0, CROP_Y0),
        (CROP_X0 + CROP_W, CROP_Y0 + CROP_H),
        MARKER_BGR, MARKER_THICK, cv2.LINE_AA,
    )
    cv2.imwrite(str(OUT_DIR / "panel0_context.png"),
                fit(context, CONTEXT_W, CONTEXT_H))

    # --- Panel 1: zoomed uncorrected overlay (pre red, post green). ---
    crop_pre  = pre [CROP_Y0:CROP_Y0+CROP_H, CROP_X0:CROP_X0+CROP_W]
    crop_post = post[CROP_Y0:CROP_Y0+CROP_H, CROP_X0:CROP_X0+CROP_W]
    overlay_uncorr = red_green_overlay(crop_pre, crop_post)
    cv2.imwrite(str(OUT_DIR / "panel1_uncorrected.png"),
                fit(overlay_uncorr, ZOOM_W, ZOOM_H))

    # --- Panel 2: zoomed corrected overlay (same crop, post is warped). ---
    crop_post_c = post_corrected[CROP_Y0:CROP_Y0+CROP_H,
                                 CROP_X0:CROP_X0+CROP_W]
    overlay_corr = red_green_overlay(crop_pre, crop_post_c)
    cv2.imwrite(str(OUT_DIR / "panel2_corrected.png"),
                fit(overlay_corr, ZOOM_W, ZOOM_H))

    print(f"Wrote 3 camera-shift panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
