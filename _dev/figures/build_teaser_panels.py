"""Render the twelve teaser panels as separate PNGs at print resolution
(960x540 each), with no labels baked in. Typst + CeTZ adds the labels
and legend on top so the figure stays vector-text where possible.

Output: /Users/user/Downloads/vrifa/paper/typst/figures/teaser_panels/
  row{0,1,2}_col{0,1,2,3}.png
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/user/Downloads/vrifa")
LABELS = REPO / "data" / "labels.json"
VIDEO = REPO / "data" / "input_1.mp4"
OUT_DIR = REPO / "paper" / "typst" / "figures" / "teaser_panels"
OUT_DIR.mkdir(exist_ok=True, parents=True)

FRAMES = [(176, "25"), (352, "50"), (670, "95")]
TILE_W, TILE_H = 960, 540

LEK_COLOR = (0, 255, 0)            # bright green BGR
ALM_COLOR = (255, 0, 255)          # bright magenta BGR
OURS_COLOR = (255, 255, 0)         # bright cyan BGR
GT_COLOR = (255, 255, 255)         # white BGR
GT_THICKNESS = 3
OURS_THICKNESS = 3

FRAME_NAME_RE = re.compile(r"^(?P<slug>input_\d+)__frame_(?P<idx>\d+)\.png$")


def gt_polygons(coco):
    img_to_idx = {im["id"]: int(FRAME_NAME_RE.match(im["file_name"]).group("idx"))
                  for im in coco["images"]
                  if im["file_name"].startswith("input_1__")}
    gt = {}
    for ann in coco["annotations"]:
        idx = img_to_idx.get(ann["image_id"])
        if idx is None:
            continue
        for seg in ann.get("segmentation", []) or []:
            gt.setdefault(idx, []).append(seg)
    return gt


def read_frame(path, idx):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    ok, frame = cap.read()
    cap.release()
    return frame


def draw_gt(canvas, polys, color, thickness):
    for poly in polys:
        if len(poly) < 6:
            continue
        pts = np.asarray(poly, np.float32).reshape(-1, 2).astype(np.int32)
        cv2.polylines(canvas, [pts], True, color, thickness, cv2.LINE_AA)


def overlay_mask(canvas, mask, color):
    if mask.shape != canvas.shape[:2]:
        mask = cv2.resize(mask, (canvas.shape[1], canvas.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    canvas[mask > 127] = color


def boundary_of_region(mask, thickness=2):
    g = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                         np.ones((3, 3), np.uint8))
    if thickness > 1:
        g = cv2.dilate(g, np.ones((thickness, thickness), np.uint8))
    return (g > 0).astype(np.uint8) * 255


def panel_boundary(shape, polys, mask, color):
    h, w = shape[:2]
    canvas = np.zeros((h, w, 3), np.uint8)
    overlay_mask(canvas, mask, color)
    draw_gt(canvas, polys, GT_COLOR, GT_THICKNESS)
    return cv2.resize(canvas, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)


def main():
    coco = json.loads(LABELS.read_text())
    gt = gt_polygons(coco)
    for r, (idx, _label) in enumerate(FRAMES):
        raw = read_frame(VIDEO, idx)
        polys = gt.get(idx, [])
        lek = (cv2.imread(f"/tmp/baseline_lekanidis/input_1/masks/frame_{idx:06d}.png", 0) > 127).astype(np.uint8)
        alm = (cv2.imread(f"/tmp/baseline_almazan/input_1/masks/frame_{idx:06d}.png", 0) > 127).astype(np.uint8)
        ours_region = (cv2.imread(f"/tmp/vrifa_best/input_1/masks/frame_{idx + 1:06d}.png", 0) > 127).astype(np.uint8)
        ours_b = boundary_of_region(ours_region, OURS_THICKNESS)

        # Column 1: input frame, scaled down
        cv2.imwrite(str(OUT_DIR / f"row{r}_col0.png"),
                    cv2.resize(raw, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA))
        # Column 2: Lekanidis + GT on black
        cv2.imwrite(str(OUT_DIR / f"row{r}_col1.png"),
                    panel_boundary(raw.shape, polys, lek * 255, LEK_COLOR))
        # Column 3: Almazán + GT on black
        cv2.imwrite(str(OUT_DIR / f"row{r}_col2.png"),
                    panel_boundary(raw.shape, polys, alm * 255, ALM_COLOR))
        # Column 4: Ours + GT on black
        cv2.imwrite(str(OUT_DIR / f"row{r}_col3.png"),
                    panel_boundary(raw.shape, polys, ours_b, OURS_COLOR))
    print(f"wrote 12 panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
