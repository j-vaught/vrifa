"""Render the colorspace-projection panels for fig:colorspace_projection.

Canonical frame: input_1.mp4 frame 352 (50% fill, hand-labeled).

Layout in the figure (handled by colorspace_projection.typ):

                                      R       G       B
    [grayscale reference]
                                      L*      a*      b*
    [raw color reference]
                                      H       S       V

Single-channel panels are rendered as 8-bit grayscale (the channel value
mapped directly to luminance). The two reference panels on the left are
the cv2 BGR2GRAY projection and the raw BGR frame.

Outputs:
  colorspace_panels/ref_gray.png        # grayscale reference
  colorspace_panels/ref_color.png       # raw color reference
  colorspace_panels/rgb_r.png           # R channel
  colorspace_panels/rgb_g.png           # G channel
  colorspace_panels/rgb_b.png           # B channel
  colorspace_panels/lab_l.png           # CIELAB L* channel
  colorspace_panels/lab_a.png           # CIELAB a* channel
  colorspace_panels/lab_b.png           # CIELAB b* channel
  colorspace_panels/hsv_h.png           # HSV H channel (rescaled to 0-255)
  colorspace_panels/hsv_s.png           # HSV S channel
  colorspace_panels/hsv_v.png           # HSV V channel
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/user/Downloads/vrifa")
VIDEO = REPO / "data" / "input_1.mp4"
OUT_DIR = REPO / "paper" / "typst" / "figures" / "colorspace_panels"
OUT_DIR.mkdir(exist_ok=True, parents=True)

FRAME_IDX = 352
# Panel size for the small channel tiles. Reference tiles use the same
# aspect ratio at a larger size; the figure caps width.
PANEL_W, PANEL_H = 960, 540


def read_frame(path, idx):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {idx} from {path}")
    return frame


def write_gray(path, channel_u8):
    """Save a single-channel 8-bit image as a grayscale PNG."""
    fit = cv2.resize(channel_u8, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), fit)


def write_color(path, bgr):
    fit = cv2.resize(bgr, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), fit)


def main():
    frame = read_frame(VIDEO, FRAME_IDX)

    # Split raw BGR into B, G, R. OpenCV's channel order is BGR.
    b_chan, g_chan, r_chan = cv2.split(frame)

    # CIELAB. OpenCV maps L to [0,255], a/b to [0,255] with 128 = neutral.
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_lab = cv2.split(lab)

    # HSV. OpenCV puts H in [0,179] for 8-bit. Rescale H to [0,255] for
    # consistent grayscale rendering; the channel still represents hue.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    h_chan_scaled = (h_chan.astype(np.uint16) * 255 // 179).astype(np.uint8)

    # Grayscale reference (cv2 BGR2GRAY = standard luminance).
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Save reference panels.
    write_gray(OUT_DIR / "ref_gray.png", gray)
    write_color(OUT_DIR / "ref_color.png", frame)

    # Save RGB row.
    write_gray(OUT_DIR / "rgb_r.png", r_chan)
    write_gray(OUT_DIR / "rgb_g.png", g_chan)
    write_gray(OUT_DIR / "rgb_b.png", b_chan)

    # Save CIELAB row.
    write_gray(OUT_DIR / "lab_l.png", l_chan)
    write_gray(OUT_DIR / "lab_a.png", a_chan)
    write_gray(OUT_DIR / "lab_b.png", b_lab)

    # Save HSV row.
    write_gray(OUT_DIR / "hsv_h.png", h_chan_scaled)
    write_gray(OUT_DIR / "hsv_s.png", s_chan)
    write_gray(OUT_DIR / "hsv_v.png", v_chan)

    print(f"Wrote 11 panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
