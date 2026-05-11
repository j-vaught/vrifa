"""Render the sample-montage panels for fig:sample_montage.

One thumbnail per sample at the labeled 50%-fill frame (the third of
the five hand-labeled frames per sample). Eleven thumbnails total.
"""

from __future__ import annotations

from pathlib import Path

import cv2

REPO = Path("/Users/user/Downloads/vrifa")
DATA_DIR = REPO / "data"
OUT_DIR = REPO / "paper" / "typst" / "figures" / "sample_montage_panels"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# 50%-fill labeled frame for each sample.
SAMPLES = (
    ("input_1",  352),
    ("input_2",   50),
    ("input_3",  100),
    ("input_4", 4050),
    ("input_5", 4050),
    ("input_6", 4050),
    ("input_7", 4050),
    ("input_8", 6300),
    ("input_9", 7734),
    ("input_10",5712),
    ("input_11",7438),
)

PANEL_W, PANEL_H = 800, 450


def read_frame(slug, idx):
    cap = cv2.VideoCapture(str(DATA_DIR / f"{slug}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {idx} from {slug}.mp4")
    return f


def fit(img):
    h, w = img.shape[:2]
    # Maintain aspect ratio: pad to 16:9 if needed, then resize.
    target_ar = PANEL_W / PANEL_H
    src_ar = w / h
    if abs(src_ar - target_ar) < 1e-3:
        return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    # Letterbox / pillarbox.
    if src_ar > target_ar:
        new_w = int(round(h * target_ar))
        x0 = max((w - new_w) // 2, 0)
        img = img[:, x0:x0 + new_w]
    else:
        new_h = int(round(w / target_ar))
        y0 = max((h - new_h) // 2, 0)
        img = img[y0:y0 + new_h]
    return cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)


def main():
    for slug, idx in SAMPLES:
        f = read_frame(slug, idx)
        h, w = f.shape[:2]
        cv2.imwrite(str(OUT_DIR / f"{slug}.png"), fit(f))
        print(f"{slug}: frame {idx} from {w}x{h} -> {PANEL_W}x{PANEL_H}")

    print(f"wrote {len(SAMPLES)} thumbnails to {OUT_DIR}")


if __name__ == "__main__":
    main()
