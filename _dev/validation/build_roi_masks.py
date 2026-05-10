#!/usr/bin/env python3
"""Build per-sample ROI mask PNGs from the labeled boundary polygon.

For each sample in data/labels.json, find the labeled frame with the
highest frame index (the saturated-fill boundary frame the user labels
as the operative laminate region) and rasterize its polygon to a binary
PNG: 255 inside the operative ROI, 0 outside.

The PNGs are at the source video's full spatial resolution (1920x1080
for the labeled subset) so they can be loaded directly by vrifa's
forthcoming --roi-mask flag without any rescale.

Output:
  data/roi_masks/input_<N>.png   one per sample
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = REPO_ROOT / "data" / "labels.json"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "roi_masks"

FRAME_NAME_RE = re.compile(r"^(?P<slug>input_\d+)__frame_(?P<idx>\d+)\.png$")


def parse_image(name: str) -> tuple[str, int]:
    m = FRAME_NAME_RE.match(name)
    if not m:
        raise ValueError(f"unexpected label image filename: {name!r}")
    return m.group("slug"), int(m.group("idx"))


def boundary_frame_per_sample(coco: dict) -> dict[str, dict]:
    """Per sample: pick the image with the highest frame index."""
    by_sample: dict[str, dict] = {}
    for img in coco["images"]:
        slug, idx = parse_image(img["file_name"])
        existing = by_sample.get(slug)
        if existing is None or idx > existing["idx"]:
            by_sample[slug] = {
                "idx": idx,
                "image_id": img["id"],
                "height": img["height"],
                "width": img["width"],
                "file_name": img["file_name"],
            }
    return by_sample


def rasterize_boundary_mask(
    coco: dict,
    image_id: int,
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in coco["annotations"]:
        if ann["image_id"] != image_id:
            continue
        for seg in ann.get("segmentation", []):
            if not seg or len(seg) < 6:
                continue
            pts = np.asarray(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
    return mask


DEFAULT_SAMPLES = ("input_1",)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples", nargs="+", default=list(DEFAULT_SAMPLES),
                        help="sample slugs to materialize ROI masks for. "
                             "Default: input_1 only (the others use --roi-margin 0). "
                             "Pass `all` to generate every sample in labels.json.")
    args = parser.parse_args(argv)

    coco = json.loads(args.labels.read_text())
    by_sample = boundary_frame_per_sample(coco)
    if not by_sample:
        print("no images in labels file", file=sys.stderr)
        return 1

    if len(args.samples) == 1 and args.samples[0].lower() == "all":
        wanted = sorted(by_sample, key=lambda x: int(x.split("_")[1]))
    else:
        wanted = [s for s in args.samples if s in by_sample]
        missing = set(args.samples) - set(by_sample)
        if missing:
            print(f"WARN: requested samples not in labels: {sorted(missing)}",
                  file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for slug in wanted:
        rec = by_sample[slug]
        mask = rasterize_boundary_mask(coco, rec["image_id"], rec["height"], rec["width"])
        if mask.sum() == 0:
            print(f"  WARN {slug}: boundary polygon rasterized to empty mask")
        out_path = args.out_dir / f"{slug}.png"
        cv2.imwrite(str(out_path), mask)
        white_px = int((mask > 0).sum())
        white_pct = white_px / mask.size * 100
        print(f"  {slug}: boundary frame {rec['idx']:>6}  "
              f"-> {out_path.name}  "
              f"({rec['width']}x{rec['height']}, {white_px:,} white px, {white_pct:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
