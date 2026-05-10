#!/usr/bin/env python3
"""Trim large input videos to a stride-N subset that preserves labeled frames.

For each video listed in data/labels.json, decide whether to trim:
  * input_1, input_2, input_3 are already short enough -- copy as-is.
  * input_4 .. input_11 get cv::VideoCapture-iterated and re-emitted as a
    new mp4 containing every Nth frame (default N=15) UNION the labeled
    frame indices for that sample. The resulting video plays in compressed
    time at the same fps, so VRIFA's lock counter and dynamic-reference
    calibration behave as if the recording were intrinsically faster.

Output:
  data/ablation_data/input_<k>.mp4   for k in 1..11
  data/ablation_data/labels.json     same schema as data/labels.json, with each
                               trimmed image entry gaining `original_frame_index`
                               and `original_file_name`, and `file_name` /
                               `frame_index` reflecting the new position.

Run from the repo root:
  python3 _dev/validation/trim_videos.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
LABELS_PATH = DATA_DIR / "labels.json"
ABLATION_DATA_DIR = DATA_DIR / "ablation_data"
DEFAULT_STRIDE = 15
KEEP_AS_IS = {"input_1", "input_2", "input_3"}

FRAME_NAME_RE = re.compile(r"^(?P<slug>input_\d+)__frame_(?P<idx>\d+)\.png$")


def parse_image(name: str) -> tuple[str, int]:
    m = FRAME_NAME_RE.match(name)
    if not m:
        raise ValueError(f"unexpected label image filename: {name!r}")
    return m.group("slug"), int(m.group("idx"))


def trim_one(
    video_path: Path,
    out_path: Path,
    labeled_indices: set[int],
    stride: int,
) -> dict[int, int]:
    """Write a stride+labels-preserving copy of `video_path` to `out_path`.

    Returns a dict mapping original frame index -> new frame index for every
    labeled frame the caller cares about. Missing labels (frames absent from
    the source) raise.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"unable to open writer for {out_path}")

    new_index_for_orig: dict[int, int] = {}
    new_index = 0
    seen_labels: set[int] = set()
    for orig_idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        keep = (orig_idx % stride == 0) or (orig_idx in labeled_indices)
        if not keep:
            continue
        writer.write(frame)
        if orig_idx in labeled_indices:
            new_index_for_orig[orig_idx] = new_index
            seen_labels.add(orig_idx)
        new_index += 1

    cap.release()
    writer.release()

    missing = labeled_indices - seen_labels
    if missing:
        raise RuntimeError(
            f"{video_path.name}: labels at original indices {sorted(missing)} "
            f"were never seen (video ended at orig_idx={orig_idx}, total={total})"
        )

    return new_index_for_orig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help=f"keep every Nth frame (default {DEFAULT_STRIDE})")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=ABLATION_DATA_DIR)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    args = parser.parse_args()

    coco = json.loads(args.labels.read_text())

    # Group labeled frames by sample slug.
    labels_by_sample: dict[str, set[int]] = {}
    for img in coco["images"]:
        slug, idx = parse_image(img["file_name"])
        labels_by_sample.setdefault(slug, set()).add(idx)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Build new image entries with remapped frame indices.
    new_images = []
    new_index_maps: dict[str, dict[int, int]] = {}

    for img in coco["images"]:
        slug, orig_idx = parse_image(img["file_name"])
        src_video = args.data_dir / f"{slug}.mp4"
        dst_video = args.out_dir / f"{slug}.mp4"

        if slug in KEEP_AS_IS:
            # Copy original verbatim, no remap. Done once per slug.
            if not dst_video.exists() or dst_video.stat().st_size != src_video.stat().st_size:
                shutil.copyfile(src_video, dst_video)
                print(f"  {slug}: copied as-is ({dst_video.stat().st_size:,} bytes)")
            new_images.append({
                **img,
                "original_frame_index": orig_idx,
                "original_file_name": img["file_name"],
            })
            continue

        # Trimmed sample. Materialize once per slug; remap label indices.
        if slug not in new_index_maps:
            print(f"  {slug}: trimming with stride={args.stride}, "
                  f"labels={sorted(labels_by_sample[slug])}")
            new_index_maps[slug] = trim_one(
                src_video,
                dst_video,
                labels_by_sample[slug],
                args.stride,
            )
            print(f"    -> wrote {dst_video} ({dst_video.stat().st_size:,} bytes)")

        new_idx = new_index_maps[slug][orig_idx]
        new_file = f"{slug}__frame_{new_idx:06d}.png"
        new_images.append({
            **img,
            "file_name": new_file,
            "original_frame_index": orig_idx,
            "original_file_name": img["file_name"],
        })

    out_labels = {
        **coco,
        "images": new_images,
    }
    out_labels_path = args.out_dir / "labels.json"
    out_labels_path.write_text(json.dumps(out_labels, indent=2) + "\n")
    print(f"\nwrote {out_labels_path} with {len(new_images)} image entries")

    # Print a per-sample summary.
    print("\nper-sample frame count and label remapping:")
    for slug in sorted(labels_by_sample, key=lambda x: int(x.split("_")[1])):
        if slug in KEEP_AS_IS:
            print(f"  {slug}: kept as-is (no remap)")
        else:
            entries = sorted(new_index_maps[slug].items())
            mapping = ", ".join(f"{o}->{n}" for o, n in entries)
            print(f"  {slug}: {mapping}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
