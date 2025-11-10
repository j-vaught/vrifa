#!/usr/bin/env python3
"""VARTM Resin Infusion Flow-Front Assessment Algorithm (VRIFA).

Detects and visualizes the advancing resin flow-front by comparing each
frame to the initial dry reference, producing masks, red-edge overlays,
and contrast heatmaps. Runs on CPU and writes PNGs and optional MP4s.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "VARTM Resin Infusion Flow-Front Assessment Algorithm (VRIFA): "
            "compare frames to the dry reference and export masks, overlays, "
            "and contrast heatmaps."
        )
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("private_assets/input_video.mp4"),
        help="Input RGB video of the VARTM infusion (typically stored in private_assets/ for the demo).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("flow_front_outputs"),
        help="Directory to store per-frame outputs.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 for all frames).",
    )
    parser.add_argument(
        "--roi-margin",
        type=float,
        default=0.15,
        help="Fractional margin cropped on each border to suppress background clutter.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=9,
        help="Odd Gaussian kernel size for smoothing the contrast map before thresholding.",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=13,
        help="Odd kernel size used for morphological close/open operations.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=400,
        help="Remove connected components below this area (pixels).",
    )
    parser.add_argument(
        "--write-videos",
        action="store_true",
        help="Also encode MP4s for mask, overlay, and heatmap outputs.",
    )
    return parser.parse_args()


def build_roi_mask(shape: Tuple[int, int], margin_fraction: float) -> np.ndarray:
    height, width = shape
    top = int(margin_fraction * height)
    bottom = int((1.0 - margin_fraction) * height)
    left = int(margin_fraction * width)
    right = int((1.0 - margin_fraction) * width)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 1
    return mask


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_front(
    frame_bgr: np.ndarray,
    reference_lab: np.ndarray,
    roi_mask: np.ndarray,
    blur_kernel: int,
    morph_kernel: int,
    min_area: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    diff = frame_lab.astype(np.float32) - reference_lab
    delta = np.sqrt(np.sum(diff * diff, axis=2))
    delta *= roi_mask

    if blur_kernel % 2 == 0:
        blur_kernel += 1
    delta_blur = cv2.GaussianBlur(delta, (blur_kernel, blur_kernel), 0)
    delta_norm = cv2.normalize(delta_blur, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    _, binary = cv2.threshold(delta_norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel_size = morph_kernel + (1 - morph_kernel % 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        filtered = np.zeros_like(binary)
        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == idx] = 255
        binary = filtered

    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, edge_kernel)
    overlay = frame_bgr.copy()
    overlay[edges > 0] = (0, 0, 255)

    heatmap = cv2.applyColorMap(delta_norm, cv2.COLORMAP_TURBO)
    return binary, overlay, heatmap


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video_path}")

    ret, reference_bgr = cap.read()
    if not ret:
        raise RuntimeError("Failed to read reference frame.")

    reference_lab = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    roi_mask = build_roi_mask(reference_lab.shape[:2], args.roi_margin)

    # Rewind so the reference (dry) frame is included in processing.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    mask_dir = args.output_dir / "masks"
    overlay_dir = args.output_dir / "overlays"
    heatmap_dir = args.output_dir / "heatmap"
    for directory in (mask_dir, overlay_dir, heatmap_dir):
        ensure_dir(directory)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mask_writer = overlay_writer = heat_writer = None
    if args.write_videos:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ensure_dir(args.output_dir / "videos")
        mask_writer = cv2.VideoWriter(
            str(args.output_dir / "videos" / "mask.mp4"), fourcc, fps, (width, height), False
        )
        overlay_writer = cv2.VideoWriter(
            str(args.output_dir / "videos" / "overlay.mp4"), fourcc, fps, (width, height)
        )
        heat_writer = cv2.VideoWriter(
            str(args.output_dir / "videos" / "heatmap.mp4"), fourcc, fps, (width, height)
        )
        if not (mask_writer.isOpened() and overlay_writer.isOpened() and heat_writer.isOpened()):
            raise RuntimeError("Failed to open one or more video writers.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    frame_index = 0
    processed = 0

    pbar_total = total_frames // args.frame_step if total_frames else None
    with tqdm(total=pbar_total, desc="Processing frames") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_index += 1
            if frame_index % args.frame_step != 0:
                continue

            mask, overlay, heatmap = detect_front(
                frame_bgr,
                reference_lab,
                roi_mask,
                args.blur_kernel,
                args.morph_kernel,
                args.min_area,
            )

            basename = f"frame_{frame_index:06d}.png"
            cv2.imwrite(str(mask_dir / basename), mask)
            cv2.imwrite(str(overlay_dir / basename), overlay)
            cv2.imwrite(str(heatmap_dir / basename), heatmap)

            if args.write_videos:
                mask_writer.write(mask)
                overlay_writer.write(overlay)
                heat_writer.write(heatmap)

            processed += 1
            pbar.update(1)

    cap.release()
    if args.write_videos:
        mask_writer.release()
        overlay_writer.release()
        heat_writer.release()

    print(f"Processed {processed} frames. Outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
