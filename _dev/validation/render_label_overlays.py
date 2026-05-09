#!/usr/bin/env python3
"""Render side-by-side overlays so the annotator can verify AND-clipping.

For each labeled frame in `paper/data/labels_55.json`, produces a PNG at
`paper/data/label_overlays/<sample>__frame_<idx>__overlay.png` showing:

  - The source frame as background.
  - The boundary polygon for that sample, drawn as a thin garnet outline.
  - The frame's cleaned polygon (post-AND), drawn as a filled rose shape
    at 35% alpha so the underlying frame stays readable.
  - The frame's pre-AND polygon (loaded from the matching raw export),
    drawn as a dashed atlantic outline so any clipped area is visible.

Run after `clean_and_merge_labels.py` has populated labels_55.json:

  python3 _dev/validation/render_label_overlays.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_FILE = REPO_ROOT / "paper" / "data" / "labels_55.json"
RAW_DIR = REPO_ROOT / "paper" / "data" / "raw_label_exports"
FRAMES_DIR = REPO_ROOT / "paper" / "data" / "label_frames"
OUT_DIR = REPO_ROOT / "paper" / "data" / "label_overlays"

GARNET = (10, 0, 115)         # BGR for #73000A — boundary outline
ROSE_FILL = (64, 46, 204)     # BGR for #CC2E40 — cleaned-polygon fill
ATLANTIC = (159, 106, 70)     # BGR for #466A9F — pre-AND outline

FRAME_NAME_RE = re.compile(r"^(?P<slug>.+)__frame_(?P<idx>\d+)\.png$")


def parse_frame(name: str) -> tuple[str, int]:
    m = FRAME_NAME_RE.match(name)
    assert m, f"unexpected filename: {name}"
    return m.group("slug"), int(m.group("idx"))


def polygons_from_anns(anns: list[dict]) -> list[np.ndarray]:
    polys: list[np.ndarray] = []
    for ann in anns:
        for seg in ann.get("segmentation", []):
            if seg and len(seg) >= 6:
                pts = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
                polys.append(pts.astype(np.int32))
    return polys


def draw_dashed_polyline(canvas: np.ndarray, pts: np.ndarray, color: tuple[int, int, int],
                        thickness: int = 2, dash: int = 18, gap: int = 12) -> None:
    """Draw a dashed closed polygon outline on canvas (in place)."""
    n = len(pts)
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        seg_len = float(np.linalg.norm(b - a))
        if seg_len < 1e-6:
            continue
        direction = (b - a).astype(np.float32) / seg_len
        traveled = 0.0
        drawing = True
        while traveled < seg_len:
            step = dash if drawing else gap
            p1 = a + direction * traveled
            p2 = a + direction * min(traveled + step, seg_len)
            if drawing:
                cv2.line(canvas, tuple(p1.astype(int)), tuple(p2.astype(int)),
                         color, thickness, cv2.LINE_AA)
            traveled += step
            drawing = not drawing


def boundary_polygons_for_sample(coco: dict, sample: str) -> list[np.ndarray]:
    """Return the polygons of the last anchor frame for a sample (the boundary)."""
    sample_imgs = sorted(
        [img for img in coco["images"] if parse_frame(img["file_name"])[0] == sample],
        key=lambda img: parse_frame(img["file_name"])[1],
    )
    if not sample_imgs:
        return []
    last_img = sample_imgs[-1]
    anns = [a for a in coco["annotations"] if a["image_id"] == last_img["id"]]
    return polygons_from_anns(anns)


def raw_polygons_for_frame(sample: str, frame_idx: int) -> list[np.ndarray]:
    """Look up the pre-AND polygon for a frame from raw_label_exports/."""
    if not RAW_DIR.exists():
        return []
    target_name = f"{sample}__frame_{frame_idx:06d}.png"
    for raw_path in RAW_DIR.glob(f"{sample}__*.json"):
        try:
            data = json.loads(raw_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for img in data.get("images", []):
            if img["file_name"] != target_name:
                continue
            anns = [a for a in data.get("annotations", []) if a["image_id"] == img["id"]]
            return polygons_from_anns(anns)
    return []


def render_one(coco: dict, image_entry: dict, out_dir: Path, frames_dir: Path) -> Path | None:
    sample, frame_idx = parse_frame(image_entry["file_name"])
    frame_path = frames_dir / image_entry["file_name"]
    if not frame_path.exists():
        print(f"  SKIP {image_entry['file_name']} (source PNG missing)")
        return None

    canvas = cv2.imread(str(frame_path))
    if canvas is None:
        print(f"  SKIP {image_entry['file_name']} (failed to read)")
        return None

    boundary_polys = boundary_polygons_for_sample(coco, sample)
    cleaned_polys = polygons_from_anns(
        [a for a in coco["annotations"] if a["image_id"] == image_entry["id"]]
    )
    raw_polys = raw_polygons_for_frame(sample, frame_idx)

    # Filled cleaned polygons at 35% alpha.
    overlay_layer = canvas.copy()
    if cleaned_polys:
        cv2.fillPoly(overlay_layer, cleaned_polys, ROSE_FILL)
    blended = cv2.addWeighted(overlay_layer, 0.35, canvas, 0.65, 0)

    # Boundary outline (always solid, garnet).
    for poly in boundary_polys:
        cv2.polylines(blended, [poly], isClosed=True,
                      color=GARNET, thickness=2, lineType=cv2.LINE_AA)

    # Cleaned outline on top of fill, slightly darker rose.
    for poly in cleaned_polys:
        cv2.polylines(blended, [poly], isClosed=True,
                      color=(54, 32, 158), thickness=2, lineType=cv2.LINE_AA)

    # Pre-AND outline as dashed atlantic, drawn last so clipped area is obvious.
    sample_imgs = [img for img in coco["images"] if parse_frame(img["file_name"])[0] == sample]
    is_boundary_frame = (
        sample_imgs
        and image_entry["id"] == max(sample_imgs, key=lambda img: parse_frame(img["file_name"])[1])["id"]
    )
    if not is_boundary_frame:
        for poly in raw_polys:
            draw_dashed_polyline(blended, poly, ATLANTIC, thickness=2, dash=18, gap=12)

    # Header label so the screenshot is self-describing.
    label_text = f"{image_entry['file_name']}"
    sub_text = "BOUNDARY (untouched)" if is_boundary_frame else "rose = post-AND, dashed blue = pre-AND"
    cv2.rectangle(blended, (0, 0), (canvas.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(blended, label_text, (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blended, sub_text, (16, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample}__frame_{frame_idx:06d}__overlay.png"
    cv2.imwrite(str(out_path), blended)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=LABELS_FILE)
    parser.add_argument("--frames", type=Path, default=FRAMES_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--sample", type=str, default=None,
                        help="restrict to a single sample slug")
    args = parser.parse_args()

    if not args.labels.exists():
        print(f"labels file not found: {args.labels}")
        return 1
    coco = json.loads(args.labels.read_text())

    targets = coco["images"]
    if args.sample:
        targets = [img for img in targets if parse_frame(img["file_name"])[0] == args.sample]
        if not targets:
            print(f"no images for sample {args.sample!r}")
            return 1

    print(f"rendering {len(targets)} overlays into {args.out}")
    written = 0
    for img in sorted(targets, key=lambda i: parse_frame(i["file_name"])):
        path = render_one(coco, img, args.out, args.frames)
        if path:
            print(f"  {path.name}")
            written += 1
    print(f"wrote {written} overlays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
