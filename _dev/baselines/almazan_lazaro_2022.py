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
- When ``--roi-mask`` is supplied, the rectangular crop is replaced by
  the mask's bounding box and the final prediction is clipped by the
  imported ROI mask.
- Histogram equalization is implemented as a masked grayscale
  equalization within the ROI before differencing against the first
  frame.
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
from typing import Any

import cv2
import numpy as np

FRAME_NAME_RE = re.compile(r"^(?P<slug>input_\d+)__frame_(?P<idx>\d+)\.png$")
VIDEO_FILL_COLOR_BGR = np.array((10, 0, 115), dtype=np.uint8)
VIDEO_EDGE_COLOR_BGR = np.array((255, 255, 255), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--labels", default=Path("data/labels.json"), type=Path)
    parser.add_argument("--frame-selection", choices=("all", "labeled"), default="all")
    parser.add_argument("--roi-margin", type=float, default=0.15)
    parser.add_argument("--roi-mask", type=Path, default=None)
    parser.add_argument("--png-compression", type=int, default=3)
    parser.add_argument("--write-overlay-video", action="store_true")
    parser.add_argument("--write-mask-video", action="store_true")
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def rasterize_polygon(segmentation: list[list[float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        if not polygon:
            continue
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def extract_frame_index(name: str) -> int:
    match = FRAME_NAME_RE.match(name)
    if not match:
        return 0
    return int(match.group("idx"))


def load_roi_mask_png(path: Path, shape: tuple[int, int]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to read ROI mask PNG {path}")
    if img.shape != shape:
        raise RuntimeError(
            f"ROI mask is {img.shape[1]}x{img.shape[0]} but video is {shape[1]}x{shape[0]}"
        )
    mask = np.where(img > 127, 255, 0).astype(np.uint8)
    if not np.any(mask):
        raise RuntimeError(f"--roi-mask '{path}' produced an empty mask")
    return mask


def load_roi_mask_coco(path: Path, video_path: Path, shape: tuple[int, int]) -> np.ndarray:
    coco = load_json(path)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    video_stem = video_path.stem

    matches = []
    prefix = f"{video_stem}__"
    for image in images:
        file_name = str(image.get("file_name", ""))
        if len(images) == 1 or file_name.startswith(prefix):
            matches.append(image)
    if not matches:
        raise RuntimeError(
            f"--roi-mask file '{path}' has no images matching '{video_stem}'"
        )
    matches.sort(key=lambda image: extract_frame_index(str(image.get("file_name", ""))))
    chosen = matches[-1]

    height, width = shape
    merged = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        if ann.get("image_id") != chosen.get("id"):
            continue
        segmentation = ann.get("segmentation", [])
        if isinstance(segmentation, list) and segmentation:
            merged |= rasterize_polygon(segmentation, height, width)
    if not np.any(merged):
        raise RuntimeError(
            f"--roi-mask file '{path}' contains no polygon annotations for '{chosen.get('file_name', video_stem)}'"
        )
    return merged


def rectangular_roi_mask(shape: tuple[int, int], margin: float) -> np.ndarray:
    height, width = shape
    x_margin = int(round(width * margin))
    y_margin = int(round(height * margin))
    x0 = min(max(x_margin, 0), width - 1)
    y0 = min(max(y_margin, 0), height - 1)
    x1 = max(x0 + 1, width - x_margin)
    y1 = max(y0 + 1, height - y_margin)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def resolve_roi_mask(
    video_path: Path,
    shape: tuple[int, int],
    margin: float,
    roi_mask_path: Path | None,
) -> np.ndarray:
    if roi_mask_path is None:
        return rectangular_roi_mask(shape, margin)
    suffix = roi_mask_path.suffix.lower()
    if suffix == ".png":
        return load_roi_mask_png(roi_mask_path, shape)
    if suffix == ".json":
        return load_roi_mask_coco(roi_mask_path, video_path, shape)
    raise RuntimeError(f"--roi-mask expects a .png or .json file, got '{suffix}'")


def roi_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        raise RuntimeError("ROI mask is empty")
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    return x0, y0, x1, y1


def masked_minmax_normalize(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(image.shape, dtype=np.uint8)
    pixels = image[mask > 0]
    if pixels.size == 0:
        return out
    min_value = float(pixels.min())
    max_value = float(pixels.max())
    if max_value <= min_value:
        out[mask > 0] = np.clip(pixels, 0, 255).astype(np.uint8)
        return out
    scaled = (pixels.astype(np.float32) - min_value) * (255.0 / (max_value - min_value))
    out[mask > 0] = np.clip(np.round(scaled), 0, 255).astype(np.uint8)
    return out


def masked_equalize_hist(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(gray.shape, dtype=np.uint8)
    pixels = gray[mask > 0]
    if pixels.size == 0:
        return out
    hist = np.bincount(pixels, minlength=256).astype(np.int64)
    nonzero = np.flatnonzero(hist)
    if nonzero.size == 0:
        return out
    cdf = hist.cumsum()
    cdf_min = int(cdf[nonzero[0]])
    total = int(pixels.size)
    if total <= cdf_min:
        out[mask > 0] = pixels
        return out
    lut = np.round((cdf - cdf_min) * 255.0 / (total - cdf_min))
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    out[mask > 0] = lut[pixels]
    return out


def masked_otsu_threshold(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(image.shape, dtype=np.uint8)
    pixels = image[mask > 0]
    if pixels.size == 0:
        return out
    threshold, _ = cv2.threshold(
        pixels.reshape(-1, 1),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    out[mask > 0] = np.where(pixels > threshold, 255, 0).astype(np.uint8)
    return out


def equalized_roi(frame: np.ndarray, roi_mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]:
    x0, y0, x1, y1 = roi_bbox(roi_mask)
    roi = frame[y0:y1, x0:x1]
    local_mask = roi_mask[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    equalized = masked_equalize_hist(gray, local_mask)
    return equalized, (x0, y0, x1, y1), local_mask


def area_open(mask: np.ndarray, min_area: int) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros(mask.shape, dtype=np.uint8)
    for label_index in range(1, num_labels):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == label_index] = 255
    return out


def overlay_frame(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    active = mask > 0
    if np.any(active):
        base = overlay[active].astype(np.float32)
        fill = VIDEO_FILL_COLOR_BGR.astype(np.float32)
        overlay[active] = np.clip(0.65 * base + 0.35 * fill, 0, 255).astype(np.uint8)
        edge = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
        overlay[edge > 0] = VIDEO_EDGE_COLOR_BGR
    return overlay


def open_video_writer(path: Path, fps: float, size: tuple[int, int], is_color: bool) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        size,
        isColor=is_color,
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer {path}")
    return writer


def process_frame(frame: np.ndarray, reference_equalized: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    equalized, (x0, y0, x1, y1), local_mask = equalized_roi(frame, roi_mask)
    delta = cv2.absdiff(equalized, reference_equalized)
    delta[local_mask == 0] = 0
    smoothed = cv2.blur(delta, (5, 5))
    smoothed[local_mask == 0] = 0

    sobel_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude_u8 = masked_minmax_normalize(magnitude, local_mask)
    mask = masked_otsu_threshold(magnitude_u8, local_mask)

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    eroded[local_mask == 0] = 0
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    dilated[local_mask == 0] = 0
    cleaned = area_open(dilated, min_area=100)
    cleaned[local_mask == 0] = 0

    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = cleaned
    full_mask[roi_mask == 0] = 0
    return full_mask


def save_mask(mask: np.ndarray, out_dir: Path, frame_index: int, png_compression: int) -> None:
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    out_path = masks_dir / f"frame_{frame_index:06d}.png"
    ok = cv2.imwrite(str(out_path), mask, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
    if not ok:
        raise RuntimeError(f"failed to write mask {out_path}")


def write_videos(
    frame: np.ndarray,
    mask: np.ndarray,
    overlay_writer: cv2.VideoWriter | None,
    mask_writer: cv2.VideoWriter | None,
) -> None:
    if overlay_writer is not None:
        overlay_writer.write(overlay_frame(frame, mask))
    if mask_writer is not None:
        mask_writer.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))


def process_all_frames(
    cap: cv2.VideoCapture,
    out_dir: Path,
    reference_equalized: np.ndarray,
    roi_mask: np.ndarray,
    png_compression: int,
    overlay_writer: cv2.VideoWriter | None,
    mask_writer: cv2.VideoWriter | None,
) -> int:
    frame_index = 0
    written = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask = process_frame(frame, reference_equalized, roi_mask)
        save_mask(mask, out_dir, frame_index, png_compression)
        write_videos(frame, mask, overlay_writer, mask_writer)
        written += 1
        if written % 250 == 0:
            print(f"[almazan] wrote {written} frames to {out_dir}", flush=True)
        frame_index += 1
    return written


def process_labeled_frames(
    cap: cv2.VideoCapture,
    out_dir: Path,
    reference_equalized: np.ndarray,
    roi_mask: np.ndarray,
    png_compression: int,
    frame_indices: list[int],
    overlay_writer: cv2.VideoWriter | None,
    mask_writer: cv2.VideoWriter | None,
) -> int:
    written = 0
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"failed to read frame {frame_index} from video")
        mask = process_frame(frame, reference_equalized, roi_mask)
        save_mask(mask, out_dir, frame_index, png_compression)
        write_videos(frame, mask, overlay_writer, mask_writer)
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
    roi_mask = resolve_roi_mask(args.video, first_frame.shape[:2], args.roi_margin, args.roi_mask)
    reference_equalized, _, _ = equalized_roi(first_frame, roi_mask)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (first_frame.shape[1], first_frame.shape[0])

    overlay_writer = None
    mask_writer = None
    if args.write_overlay_video:
        overlay_writer = open_video_writer(args.out / "videos" / "overlay.mp4", fps, size, is_color=True)
    if args.write_mask_video:
        mask_writer = open_video_writer(args.out / "videos" / "mask.mp4", fps, size, is_color=True)

    try:
        if args.frame_selection == "all":
            written = process_all_frames(
                cap,
                args.out,
                reference_equalized,
                roi_mask,
                args.png_compression,
                overlay_writer,
                mask_writer,
            )
        else:
            indices = labeled_frame_indices(args.labels, sample)
            written = process_labeled_frames(
                cap,
                args.out,
                reference_equalized,
                roi_mask,
                args.png_compression,
                indices,
                overlay_writer,
                mask_writer,
            )
    finally:
        if overlay_writer is not None:
            overlay_writer.release()
        if mask_writer is not None:
            mask_writer.release()
        cap.release()

    print(f"[almazan] completed {sample}: wrote {written} mask(s) into {args.out / 'masks'}", flush=True)


if __name__ == "__main__":
    main()
