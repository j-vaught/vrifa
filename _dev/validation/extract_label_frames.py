#!/usr/bin/env python3
"""Extract anchor frames for the labeling pass.

For every video in `data/`, samples five frames at the 5 / 25 / 50 /
75 / 95 percent positions of the total frame count and writes them to
`paper/data/label_frames/<sample_slug>/<frame_index>.png`.

The slug is the video filename without extension, with spaces and
colons replaced by underscores so it survives makesense.ai's import
form.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "paper" / "data" / "label_frames"
AIDS_DIR = REPO_ROOT / "paper" / "data" / "label_aids"

PERCENTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def slugify(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    return cleaned.strip("_")


def extract_one(video_path: Path, output_root: Path) -> list[Path]:
    """Extract anchor frames into a flat directory.

    Filenames are prefixed with the sample slug so makesense.ai can ingest
    every frame in a single drop without filename collisions. The
    agreement script later parses the slug back out of the filename.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"no frames reported for {video_path}")

    slug = slugify(video_path.stem)
    output_root.mkdir(parents=True, exist_ok=True)

    indices = [max(0, min(total - 1, int(round(p * (total - 1))))) for p in PERCENTILES]
    indices = sorted(set(indices))
    written: list[Path] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  WARN failed to read {video_path.name} frame {idx}")
            continue
        out_path = output_root / f"{slug}__frame_{idx:06d}.png"
        cv2.imwrite(str(out_path), frame)
        written.append(out_path)
    cap.release()
    return written


def extract_dry_reference(video_path: Path, aids_root: Path) -> Path | None:
    """Save frame 0 of the video as a dry-fabric reference for the annotator.

    The dry reference is what the laminate looks like before any resin
    arrives. Keeping it open in a second tab lets the labeler distinguish
    pre-existing dark features (wrinkles, weave, sealant) from new wetting.
    Loading the algorithm's own delta or mask as a labeling aid would create
    circular feedback in the agreement metric and is intentionally not done.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    aids_root.mkdir(parents=True, exist_ok=True)
    slug = slugify(video_path.stem)
    out_path = aids_root / f"{slug}__dry_reference.png"
    cv2.imwrite(str(out_path), frame)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--aids-dir", type=Path, default=AIDS_DIR,
                        help="Directory for per-sample dry reference frames.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(args.data_dir.glob("*.mp4"))
    if not videos:
        print(f"no .mp4 files under {args.data_dir}")
        return 1

    total_written = 0
    aids_written = 0
    print(f"extracting from {len(videos)} videos into {args.output_dir}")
    for video in videos:
        try:
            written = extract_one(video, args.output_dir)
        except RuntimeError as exc:
            print(f"  SKIP {video.name}: {exc}")
            continue
        total_written += len(written)
        print(f"  {video.name}: {len(written)} frames")
        ref_path = extract_dry_reference(video, args.aids_dir)
        if ref_path is not None:
            aids_written += 1

    print(f"wrote {total_written} anchor frames and {aids_written} dry references")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
