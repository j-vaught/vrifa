"""Render the camera-shift panels for fig:camera_shift_pair.

Demonstrates the camera-shift registration stage on the real bump event
in input_1.mp4 between frames ~65 and ~70 (gradual ~3.7 px drift that
settles by frame 75). The figure tells the story through the pipeline's
own delta field, the per-pixel scalar that drives the threshold. Without
the registration warp every laminate edge produces a false-positive
wetting signal; with the warp, the field collapses to near-empty
residuals.

Three panels:
  panel0_context.png   Reference frame (input_1, frame 63), for context.
  panel1_uncorrected.png  Pipeline delta field, max(0, ref - cur) on the
                          working channel ($L^*$), with the post-event
                          frame (frame 75) as cur. ROI mask applied.
                          Rendered as Turbo heatmap with a shared
                          intensity scale.
  panel2_corrected.png Same delta field after the registration warp is
                          applied to the post-event frame. Same Turbo
                          scale so the magnitudes are directly
                          comparable.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/user/Downloads/vrifa")
VIDEO = REPO / "data" / "input_1.mp4"
ROI_MASK_PATH = REPO / "data" / "roi_masks" / "input_1.png"
OUT_DIR = REPO / "paper" / "typst" / "figures" / "camera_shift_panels"
OUT_DIR.mkdir(exist_ok=True, parents=True)

PRE_IDX = 63        # well before the bump (frames 65-70)
POST_IDX = 75       # after the bump has settled

# Output dimensions for each rendered panel (16:9 to match other figures).
PANEL_W, PANEL_H = 960, 540

# Heatmap background colors: dim gray for "inside ROI but no signal",
# pure black for "outside ROI". Matches the visual language of the
# rendered overlay/heatmap stages in the Method figures.
BG_INSIDE_ROI = (40, 40, 40)
BG_OUTSIDE_ROI = (0, 0, 0)


def read_frame(idx):
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {idx}")
    return frame


def lstar(bgr):
    """Extract CIELAB L* channel, 8-bit."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0]


def load_roi(h, w):
    raw = cv2.imread(str(ROI_MASK_PATH), cv2.IMREAD_GRAYSCALE)
    if raw.shape != (h, w):
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    return (raw > 127).astype(np.uint8) * 255


def phase_translation(pre_l, post_l):
    """cv2.phaseCorrelate on L* gives (dx, dy) translation pre to post."""
    return cv2.phaseCorrelate(pre_l.astype(np.float32),
                              post_l.astype(np.float32))[0]


def warp_translation(img, dx, dy):
    h, w = img.shape[:2]
    M = np.array([[1.0, 0.0, dx],
                  [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def delta_field(reference_l, current_l, roi):
    """Pipeline delta on the L* working channel: max(0, ref - cur),
    restricted to the ROI. Output uint8 in [0, 255]."""
    d = np.maximum(
        reference_l.astype(np.int16) - current_l.astype(np.int16),
        0,
    )
    d = np.clip(d, 0, 255).astype(np.uint8)
    d[roi == 0] = 0
    return d


def heatmap(delta, roi, vmax):
    """Render a delta field as a Turbo heatmap with a shared vmax scale.
    Zero-delta pixels show as a dim gray "no signal" background; outside
    the ROI is painted black.
    """
    scaled = np.clip(
        delta.astype(np.float32) * 255.0 / max(vmax, 1),
        0, 255,
    ).astype(np.uint8)
    hm = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    hm[delta == 0] = BG_INSIDE_ROI
    hm[roi == 0] = BG_OUTSIDE_ROI
    return hm


def fit(img):
    return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


def main():
    pre = read_frame(PRE_IDX)
    post = read_frame(POST_IDX)
    h, w = pre.shape[:2]

    roi = load_roi(h, w)
    L_pre = lstar(pre)
    L_post = lstar(post)

    # Detect shift and apply correction.
    dx, dy = phase_translation(L_pre, L_post)
    print(f"detected shift post relative to pre: dx={dx:.2f} px, dy={dy:.2f} px")

    post_corrected = warp_translation(post, -dx, -dy)
    L_post_c = lstar(post_corrected)

    # Compute both delta fields.
    delta_uncorr = delta_field(L_pre, L_post, roi)
    delta_corr   = delta_field(L_pre, L_post_c, roi)

    # Shared display scale so the two heatmaps are honestly comparable.
    vmax = int(max(delta_uncorr.max(), 1))
    print(f"shared vmax: {vmax}")
    print(f"  uncorrected: max={delta_uncorr.max()}, "
          f"mean (in ROI)={delta_uncorr[roi > 0].mean():.2f}")
    print(f"  corrected:   max={delta_corr.max()}, "
          f"mean (in ROI)={delta_corr[roi > 0].mean():.2f}")

    # Panel 0: reference frame (raw BGR), full extent, for context.
    cv2.imwrite(str(OUT_DIR / "panel0_context.png"), fit(pre))

    # Panels 1 and 2: zoom in on the central 50% (crop outer 25% on
    # every edge) of the delta heatmap so the laminate-edge structure
    # reads clearly at print size.
    x0, y0 = w // 4, h // 4
    x1, y1 = x0 + w // 2, y0 + h // 2
    def crop_center(img):
        return img[y0:y1, x0:x1]

    cv2.imwrite(
        str(OUT_DIR / "panel1_uncorrected.png"),
        fit(crop_center(heatmap(delta_uncorr, roi, vmax))),
    )
    cv2.imwrite(
        str(OUT_DIR / "panel2_corrected.png"),
        fit(crop_center(heatmap(delta_corr, roi, vmax))),
    )

    print(f"Wrote 3 camera-shift panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
