#!/usr/bin/env python3
"""Reimplement Almazan-Lazaro et al. 2022 as a standalone baseline.

Pipeline source:
Applied Computer Vision for Composite Material Manufacturing by
Optimizing the Impregnation Velocity: An Experimental Approach,
J.-A. Almazan-Lazaro, E. Lopez-Alba, F.-A. Diaz-Garrido,
J Manuf Processes 74, 2022.

Assumptions required by gaps in the paper text:

- Scaramuzza-style lens-distortion correction is treated as a no-op for
  these lab videos because the framing shows negligible distortion.
- Histogram equalization is implemented with `cv2.equalizeHist` on the
  grayscale ROI before differencing against the first frame.
- The Sobel magnitude image is min-max normalized to 8-bit and Otsu
  thresholded to obtain the segmentation mask.
- The unspecified erosion and dilation use a 3x3 square structuring
  element for one iteration each.
- The unspecified small-area removal threshold is fixed at 100 pixels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

FRAME_NAME_RE = re.compile(r"^(?P<slug>input_\d+)__frame_(?P<idx>\d+)\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--labels", default=Path("data/labels.json"), type=Path)
    parser.add_argument("--frame-selection", choices=("all", "labeled"), default="all")
    parser.add_argument("--roi-margin", type=float, default=0.15)
    parser.add_argument("--png-compression", type=int, default=3)
    return parser.parse_args()


def parse_frame_name(name: str) -> tuple[str, int]:
    match = FRAME_NAME_RE.match(name)
    if not match:
        raise ValueError(f"unexpected label image filename: {name!r}")
    return match.group("slug"), int(match.group("idx"))


def labeled_frame_indices(labels_path: Path, sample: str) -> list[int]:
    labels = json.loads(labels_path.read_text())
    indices: list[int] = []
    for image in labels["images"]:
        slug, frame_index = parse_frame_name(image["file_name"])
        if slug == sample:
            indices.append(frame_index)
    if not indices:
        raise ValueError(f"no labeled frames found for sample {sample!r}")
    return sorted(indices)


def compute_roi(shape: tuple[int, int, int], margin: float) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x_margin = int(round(width * margin))
    y_margin = int(round(height * margin))
    x0 = min(max(x_margin, 0), width - 1)
    y0 = min(max(y_margin, 0), height - 1)
    x1 = max(x0 + 1, width - x_margin)
    y1 = max(y0 + 1, height - y_margin)
    return x0, y0, x1, y1


def equalized_roi(frame: np.ndarray, roi_margin: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = compute_roi(frame.shape, roi_margin)
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized, (x0, y0, x1, y1)


def area_open(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, dtype=np.uint8)
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == label_index] = 255
    return out


def process_frame(frame: np.ndarray, reference_equalized: np.ndarray, roi_margin: float) -> np.ndarray:
    equalized, (x0, y0, x1, y1) = equalized_roi(frame, roi_margin)
    delta = cv2.absdiff(equalized, reference_equalized)
    smoothed = cv2.blur(delta, (5, 5))

    sobel_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude_u8 = cv2.normalize(magnitude, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(magnitude_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    cleaned = area_open(dilated, min_area=100)

    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = cleaned
    return full_mask


def save_mask(mask: np.ndarray, out_dir: Path, frame_index: int, png_compression: int) -> None:
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    out_path = masks_dir / f"frame_{frame_index:06d}.png"
    ok = cv2.imwrite(str(out_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
    if not ok:
        raise RuntimeError(f"failed to write mask {out_path}")


def process_all_frames(
    cap: cv2.VideoCapture,
    out_dir: Path,
    reference_equalized: np.ndarray,
    roi_margin: float,
    png_compression: int,
) -> int:
    frame_index = 0
    written = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask = process_frame(frame, reference_equalized, roi_margin)
        save_mask(mask, out_dir, frame_index, png_compression)
        written += 1
        if written % 250 == 0:
            print(f"[almazan] wrote {written} frames to {out_dir}", flush=True)
        frame_index += 1
    return written


def process_labeled_frames(
    cap: cv2.VideoCapture,
    out_dir: Path,
    reference_equalized: np.ndarray,
    roi_margin: float,
    png_compression: int,
    frame_indices: list[int],
) -> int:
    written = 0
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {frame_index} from video")
        mask = process_frame(frame, reference_equalized, roi_margin)
        save_mask(mask, out_dir, frame_index, png_compression)
        written += 1
        print(f"[almazan] wrote labeled frame {frame_index} to {out_dir}", flush=True)
    return written


def main() -> None:
    args = parse_args()
    sample = args.video.stem
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video {args.video}")

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"failed to read first frame from {args.video}")
    reference_equalized, _ = equalized_roi(first_frame, args.roi_margin)

    try:
        if args.frame_selection == "all":
            written = process_all_frames(cap, args.out, reference_equalized, args.roi_margin, args.png_compression)
        else:
            indices = labeled_frame_indices(args.labels, sample)
            written = process_labeled_frames(cap, args.out, reference_equalized, args.roi_margin, args.png_compression, indices)
    finally:
        cap.release()

    print(f"[almazan] completed {sample}: wrote {written} mask(s) into {args.out / 'masks'}", flush=True)


if __name__ == "__main__":
    main()
