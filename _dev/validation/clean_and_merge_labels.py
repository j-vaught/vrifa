#!/usr/bin/env python3
"""Clean a makesense.ai COCO export and merge into the running labels_55.json.

Per-sample workflow:

  - The labels for the LAST anchor frame in each sample (the 95%-fill
    position) define the laminate boundary. That polygon is treated as
    the in-bounds region for the entire sample.
  - Every other annotation in the same sample is intersected with the
    boundary polygon so any over-extension into bag/wrinkle/background
    pixels is clipped.
  - The cleaned annotations replace any previous entries for the same
    sample in `paper/data/labels_55.json`.

Use:

  python3 _dev/validation/clean_and_merge_labels.py \
    --input <path/to/raw_makesense_export.json> \
    --sample input_1
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_OUT = REPO_ROOT / "paper" / "data" / "labels_55.json"
RAW_DIR = REPO_ROOT / "paper" / "data" / "raw_label_exports"
FRAME_NAME_RE = re.compile(r"^(?P<slug>.+)__frame_(?P<idx>\d+)\.png$")


def parse_frame(name: str) -> tuple[str, int]:
    m = FRAME_NAME_RE.match(name)
    if not m:
        raise ValueError(f"unexpected filename: {name}")
    return m.group("slug"), int(m.group("idx"))


def rasterize(seg_list: list[list[float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for seg in seg_list:
        if not seg or len(seg) < 6:
            continue
        pts = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def mask_to_segmentation(mask: np.ndarray, min_pixels: int = 16) -> list[list[float]]:
    """Convert a binary mask back to COCO-style segmentation polygons."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[list[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_pixels:
            continue
        flat = contour.flatten().astype(float).tolist()
        if len(flat) >= 6:
            out.append(flat)
    return out


def mask_bbox_and_area(mask: np.ndarray) -> tuple[list[int], int]:
    if mask.sum() == 0:
        return [0, 0, 0, 0], 0
    ys, xs = np.where(mask > 0)
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    return [x, y, w, h], int(mask.sum())


def clean_sample(coco: dict[str, Any], sample: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (cleaned_coco, summary)."""
    sample_images = [img for img in coco["images"] if parse_frame(img["file_name"])[0] == sample]
    if not sample_images:
        raise ValueError(f"no images in export match sample {sample!r}")

    sample_images.sort(key=lambda img: parse_frame(img["file_name"])[1])
    last_img = sample_images[-1]
    height, width = last_img["height"], last_img["width"]

    last_annotations = [a for a in coco["annotations"] if a["image_id"] == last_img["id"]]
    if not last_annotations:
        raise ValueError(
            f"the last frame for {sample} ({last_img['file_name']}) has no polygon; "
            "this is the saturated frame and is required to define the boundary"
        )

    # Build the boundary mask from EVERY polygon on the last frame, in case the
    # annotator drew it as multiple shapes.
    boundary = np.zeros((height, width), dtype=np.uint8)
    for ann in last_annotations:
        boundary |= rasterize(ann["segmentation"], height, width)
    if boundary.sum() == 0:
        raise ValueError(f"boundary mask for {sample} is empty after rasterization")

    summary = {
        "sample": sample,
        "boundary_image": last_img["file_name"],
        "boundary_pixel_count": int(boundary.sum()),
        "frames": [],
    }

    cleaned_annotations: list[dict[str, Any]] = []
    next_ann_id = max((a["id"] for a in coco["annotations"]), default=0) + 1

    for img in sample_images:
        frame_anns = [a for a in coco["annotations"] if a["image_id"] == img["id"]]
        if img["id"] == last_img["id"]:
            # Boundary frame stays untouched.
            for ann in frame_anns:
                cleaned_annotations.append(ann)
            summary["frames"].append({
                "frame": img["file_name"],
                "is_boundary": True,
                "polygons_in": len(frame_anns),
                "polygons_out": len(frame_anns),
                "pixels_in": sum(int(rasterize(a["segmentation"], height, width).sum()) for a in frame_anns),
                "pixels_out": sum(int(rasterize(a["segmentation"], height, width).sum()) for a in frame_anns),
            })
            continue

        if not frame_anns:
            summary["frames"].append({
                "frame": img["file_name"],
                "is_boundary": False,
                "polygons_in": 0,
                "polygons_out": 0,
                "pixels_in": 0,
                "pixels_out": 0,
            })
            continue

        # Combine all polygons on this frame, intersect with boundary.
        combined = np.zeros((height, width), dtype=np.uint8)
        for ann in frame_anns:
            combined |= rasterize(ann["segmentation"], height, width)
        in_pixels = int(combined.sum())
        clipped = combined & boundary
        out_pixels = int(clipped.sum())

        new_seg = mask_to_segmentation(clipped)
        bbox, area = mask_bbox_and_area(clipped)

        if new_seg:
            new_ann = {
                "id": next_ann_id,
                "image_id": img["id"],
                "category_id": frame_anns[0].get("category_id", 1),
                "segmentation": new_seg,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            }
            cleaned_annotations.append(new_ann)
            next_ann_id += 1

        summary["frames"].append({
            "frame": img["file_name"],
            "is_boundary": False,
            "polygons_in": len(frame_anns),
            "polygons_out": len(new_seg),
            "pixels_in": in_pixels,
            "pixels_out": out_pixels,
            "pixels_dropped": in_pixels - out_pixels,
        })

    # Other-sample annotations pass through untouched.
    other_image_ids = {img["id"] for img in coco["images"] if parse_frame(img["file_name"])[0] != sample}
    cleaned_annotations.extend(a for a in coco["annotations"] if a["image_id"] in other_image_ids)

    cleaned = {
        "info": coco.get("info", {}),
        "categories": coco.get("categories", []),
        "images": coco["images"],
        "annotations": cleaned_annotations,
    }
    return cleaned, summary


def merge_into_running(running: dict[str, Any], new_coco: dict[str, Any], sample: str) -> dict[str, Any]:
    """Merge new sample annotations into the running labels_55.json."""
    if not running.get("images"):
        return new_coco

    # Drop any prior images and annotations for this sample.
    keep_image_ids = {img["id"] for img in running["images"] if parse_frame(img["file_name"])[0] != sample}
    kept_images = [img for img in running["images"] if img["id"] in keep_image_ids]
    kept_anns = [a for a in running["annotations"] if a["image_id"] in keep_image_ids]

    # Renumber incoming images and annotations to avoid id collisions.
    next_img_id = max((img["id"] for img in kept_images), default=0) + 1
    next_ann_id = max((a["id"] for a in kept_anns), default=0) + 1
    id_remap: dict[int, int] = {}

    new_images = [img for img in new_coco["images"] if parse_frame(img["file_name"])[0] == sample]
    new_anns = [a for a in new_coco["annotations"] if any(img["id"] == a["image_id"] and parse_frame(img["file_name"])[0] == sample for img in new_coco["images"])]

    merged_images = list(kept_images)
    for img in new_images:
        old_id = img["id"]
        new_img = dict(img)
        new_img["id"] = next_img_id
        id_remap[old_id] = next_img_id
        merged_images.append(new_img)
        next_img_id += 1

    merged_anns = list(kept_anns)
    for ann in new_anns:
        new_ann = dict(ann)
        new_ann["id"] = next_ann_id
        new_ann["image_id"] = id_remap[ann["image_id"]]
        merged_anns.append(new_ann)
        next_ann_id += 1

    merged = {
        "info": running.get("info", new_coco.get("info", {})),
        "categories": running.get("categories") or new_coco.get("categories", []),
        "images": merged_images,
        "annotations": merged_anns,
    }
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="raw makesense.ai COCO export to ingest")
    parser.add_argument("--sample", type=str, required=True,
                        help="sample slug, e.g. input_1")
    parser.add_argument("--running", type=Path, default=LABELS_OUT,
                        help="running labels_55.json to merge into")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                        help="directory to stash a copy of the raw export")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1

    coco = json.loads(args.input.read_text())
    cleaned, summary = clean_sample(coco, args.sample)

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    raw_dest = args.raw_dir / f"{args.sample}__{args.input.stem}.json"
    shutil.copy2(args.input, raw_dest)
    print(f"stashed raw export at {raw_dest}")

    if args.running.exists():
        running = json.loads(args.running.read_text())
    else:
        running = {"info": {}, "categories": [], "images": [], "annotations": []}

    merged = merge_into_running(running, cleaned, args.sample)
    args.running.parent.mkdir(parents=True, exist_ok=True)
    args.running.write_text(json.dumps(merged, indent=2))
    print(f"wrote {args.running}")

    print()
    print(f"=== cleanup summary for {summary['sample']} ===")
    print(f"boundary frame: {summary['boundary_image']} ({summary['boundary_pixel_count']:,} px)")
    for f in summary["frames"]:
        marker = " <-- boundary" if f["is_boundary"] else ""
        if f["is_boundary"]:
            print(f"  {f['frame']}: {f['pixels_out']:,} px{marker}")
        else:
            dropped = f.get("pixels_dropped", 0)
            pct = (100 * dropped / f["pixels_in"]) if f["pixels_in"] else 0
            print(f"  {f['frame']}: {f['pixels_in']:,} -> {f['pixels_out']:,} px "
                  f"(clipped {dropped:,} / {pct:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
