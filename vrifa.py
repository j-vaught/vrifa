#!/usr/bin/env python3
"""VARTM Resin Infusion Flow-Front Assessment Algorithm (VRIFA).

Detects and visualizes the advancing resin flow-front by comparing each
frame to the initial dry reference, producing masks, red-edge overlays,
and contrast heatmaps. Runs on CPU and writes PNGs and optional MP4s.
"""

from __future__ import annotations

import math
import time
from collections import deque, OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple

import cv2
import json
import numpy as np
import yaml
from tqdm import tqdm

COLORSPACE_ALIASES = {
    "CIELAB": "CIELAB",
    "LAB": "CIELAB",
    "RGB": "RGB",
    "HSV": "HSV",
    "GRAYSCALE": "GRAYSCALE",
    "GRAY": "GRAYSCALE",
}

COLORSPACE_CHANNEL_COUNTS = {
    "CIELAB": 3,
    "RGB": 3,
    "HSV": 3,
    "GRAYSCALE": 1,
}

MORPH_SHAPE_MAP = {
    "ellipse": cv2.MORPH_ELLIPSE,
    "rect": cv2.MORPH_RECT,
    "cross": cv2.MORPH_CROSS,
}


class ReferenceMode(NamedTuple):
    name: str
    offset: Optional[int]


@dataclass
class AnnotationBox:
    x: int
    y: int
    w: int
    h: int
    area: int
    segmentation: list[float]


@dataclass
class AnnotationFrame:
    frame_index: int
    frame_bgr: np.ndarray
    boxes: list[AnnotationBox]

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
        help="Default fractional margin cropped equally on every border.",
    )
    parser.add_argument(
        "--roi-margin-top",
        type=float,
        default=None,
        help="Override for the top margin fraction (default: use --roi-margin).",
    )
    parser.add_argument(
        "--roi-margin-bottom",
        type=float,
        default=None,
        help="Override for the bottom margin fraction (default: use --roi-margin).",
    )
    parser.add_argument(
        "--roi-margin-left",
        type=float,
        default=None,
        help="Override for the left margin fraction (default: use --roi-margin).",
    )
    parser.add_argument(
        "--roi-margin-right",
        type=float,
        default=None,
        help="Override for the right margin fraction (default: use --roi-margin).",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=9,
        help="Odd Gaussian kernel size for smoothing the contrast map before thresholding.",
    )
    parser.add_argument(
        "--skip-blur",
        action="store_true",
        help="Skip the Gaussian blur step before thresholding (delta is normalized directly).",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=13,
        help="Odd kernel size used for morphological close/open operations.",
    )
    parser.add_argument(
        "--morph-shape",
        choices=list(MORPH_SHAPE_MAP),
        default="ellipse",
        help="Shape of the structuring element used for morphology.",
    )
    parser.add_argument(
        "--morph-close-iterations",
        type=int,
        default=1,
        help="How many times to apply the closing operation (0 to skip).",
    )
    parser.add_argument(
        "--morph-open-iterations",
        type=int,
        default=1,
        help="How many times to apply the opening operation (0 to skip).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=400,
        help="Remove connected components below this area (pixels).",
    )
    parser.add_argument(
        "--contrast-threshold",
        type=float,
        default=None,
        help="Manual absolute threshold in the normalized contrast map (0-255). Overrides Otsu.",
    )
    parser.add_argument(
        "--contrast-percentile",
        type=float,
        default=None,
        help="Percentile (0-100) within the ROI to set the threshold (overrides --contrast-threshold).",
    )
    parser.add_argument(
        "--threshold-offset",
        type=float,
        default=0.0,
        help="Value added after scaling the threshold (before clamping to 0-255).",
    )
    parser.add_argument(
        "--darken-only",
        action="store_true",
        help="Only detect areas that got darker (lower L* or brightness). Ignores brightening.",
    )
    parser.add_argument(
        "--peak-reference",
        action="store_true",
        help="Compare each pixel to its historical maximum brightness instead of a fixed reference frame. Useful when pixels start dark, brighten, then darken again.",
    )
    parser.add_argument(
        "--write-videos",
        action="store_true",
        help="Quick flag to enable all MP4 outputs (mask, overlay, heatmap).",
    )
    parser.add_argument(
        "--write-mask-pngs",
        type=str,
        default="false",
        help="Write per-frame mask PNGs (true/false, default false).",
    )
    parser.add_argument(
        "--write-overlay-pngs",
        type=str,
        default="false",
        help="Write overlay PNGs with the red front (true/false, default false).",
    )
    parser.add_argument(
        "--write-heatmap-pngs",
        type=str,
        default="false",
        help="Write contrast heatmap PNGs (true/false, default false).",
    )
    parser.add_argument(
        "--write-mask-video",
        type=str,
        default=None,
        help="Write mask MP4 (true/false). Defaults to --write-videos.",
    )
    parser.add_argument(
        "--write-overlay-video",
        type=str,
        default=None,
        help="Write overlay MP4 (true/false). Defaults to overlay-only output unless overridden by --write-videos.",
    )
    parser.add_argument(
        "--write-heatmap-video",
        type=str,
        default=None,
        help="Write heatmap MP4 (true/false). Defaults to --write-videos.",
    )
    parser.add_argument(
        "--lock-frames",
        type=int,
        default=3,
        help=(
            "Frames a pixel must stay filled before it becomes permanent (0 disables the glare guard)."
        ),
    )
    parser.add_argument(
        "--colorspace",
        type=str.upper,
        choices=list(COLORSPACE_ALIASES),
        default="CIELAB",
        help="Colorspace used for contrast comparison (CIELAB, RGB, HSV, or Grayscale).",
    )
    parser.add_argument(
        "--channel-weights",
        type=str,
        default=None,
        help=(
            "Comma-separated per-channel weights applied before computing delta (single value repeats). "
            "Values should follow the selected colorspace (e.g., three weights for RGB/HSV)."
        ),
    )
    parser.add_argument(
        "--ref-mode",
        nargs="+",
        default=["first"],
        help=(
            "Reference selection: `first`, `running`, `prev <N>`, `absolute <N>` (N is frame count)."
        ),
    )
    parser.add_argument(
        "--ref-running-alpha",
        type=float,
        default=0.05,
        help="Exponential smoothing factor when using `--ref-mode running`.",
    )
    parser.add_argument(
        "--dynamic-calibration-frames",
        type=int,
        default=10,
        help=(
            "Number of processed frames used to model sqrt-area growth before dynamic references "
            "become active."
        ),
    )
    parser.add_argument(
        "--dynamic-target-fraction",
        type=float,
        default=0.2,
        help=(
            "Target ROI fraction that defines the difference the dynamic reference should "
            "maintain (0.0-1.0, default 0.2 for 20%% of the ROI)."
        ),
    )
    parser.add_argument(
        "--dynamic-ref-cache-size",
        type=int,
        default=32,
        help="How many randomly accessed reference frames to cache when dynamic reference mode is enabled.",
    )
    parser.add_argument(
        "--annotation-formats",
        type=str,
        default="",
        help="Comma-separated annotation formats to export (coco, yolov5, darknet).",
    )
    parser.add_argument(
        "--annotation-mode",
        choices=["all", "count", "stride"],
        default="all",
        help="How to pick processed frames for annotation exports.",
    )
    parser.add_argument(
        "--annotation-count",
        type=int,
        default=0,
        help="Number of frames to evenly sample when --annotation-mode count is used.",
    )
    parser.add_argument(
        "--annotation-stride",
        type=int,
        default=1,
        help="Stride (every N processed frames) used when --annotation-mode stride is selected.",
    )
    parser.add_argument(
        "--annotation-segmentation-tolerance",
        type=float,
        default=0.0,
        help="Approximation tolerance (px) for the contour polygons (0 = full fidelity).",
    )
    parser.add_argument(
        "--annotation-segmentation-max-edge-length",
        type=float,
        default=0.0,
        help="Maximum length (px) allowed between consecutive segmentation points (0 disables).",
    )
    parser.add_argument(
        "--dynamic-lag-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the modeled lag before converting back to frames (default 1.0).",
    )
    parser.add_argument(
        "--dynamic-lag-linear",
        action="store_true",
        help="Use a linear lag schedule (0 → max) instead of the Δt formula.",
    )
    parser.add_argument(
        "--dynamic-lag-linear-max",
        type=int,
        default=60,
        help="Maximum reference lag (in frames) when --dynamic-lag-linear is enabled.",
    )
    parser.add_argument(
        "--dynamic-lag-linear-start",
        type=int,
        default=0,
        help="Starting reference lag (frames) for linear mode (default 0).",
    )
    parser.add_argument(
        "--dynamic-lag-log",
        type=Path,
        default=None,
        help="Path to dump dynamic reference lag per processed frame (CSV with frame, lag).",
    )
    args = parser.parse_args()
    args.colorspace = COLORSPACE_ALIASES[args.colorspace]
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}
    def parse_bool(text: str, flag: str) -> bool:
        lower = text.strip().lower()
        if lower in truthy:
            return True
        if lower in falsy:
            return False
        raise ValueError(f"{flag} expects true/false but got '{text}'.")
    def parse_bool_optional(value: Optional[str], flag: str) -> Optional[bool]:
        if value is None:
            return None
        return parse_bool(value, flag)
    try:
        args.write_mask_pngs = parse_bool(args.write_mask_pngs, "--write-mask-pngs")
        args.write_overlay_pngs = parse_bool(args.write_overlay_pngs, "--write-overlay-pngs")
        args.write_heatmap_pngs = parse_bool(args.write_heatmap_pngs, "--write-heatmap-pngs")
        args.write_mask_video = parse_bool_optional(args.write_mask_video, "--write-mask-video")
        args.write_overlay_video = parse_bool_optional(
            args.write_overlay_video, "--write-overlay-video"
        )
        args.write_heatmap_video = parse_bool_optional(
            args.write_heatmap_video, "--write-heatmap-video"
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.write_videos:
        if args.write_mask_video is None:
            args.write_mask_video = True
        if args.write_overlay_video is None:
            args.write_overlay_video = True
        if args.write_heatmap_video is None:
            args.write_heatmap_video = True

    if args.write_mask_video is None:
        args.write_mask_video = False
    if args.write_overlay_video is None:
        args.write_overlay_video = True
    if args.write_heatmap_video is None:
        args.write_heatmap_video = False

    args.write_mask_video = bool(args.write_mask_video)
    args.write_overlay_video = bool(args.write_overlay_video)
    args.write_heatmap_video = bool(args.write_heatmap_video)
    channel_count = COLORSPACE_CHANNEL_COUNTS[args.colorspace]
    if args.channel_weights is None:
        args.channel_weights = np.ones(channel_count, dtype=np.float32)
    else:
        try:
            args.channel_weights = parse_channel_weights(
                args.channel_weights, channel_count
            )
        except ValueError as exc:
            parser.error(f"--channel-weights {exc}")
    try:
        args.ref_mode = parse_ref_mode(args.ref_mode)
    except ValueError as exc:
        parser.error(f"--ref-mode {exc}")

    args.ref_running_alpha = float(np.clip(args.ref_running_alpha, 0.0, 1.0))
    args.morph_close_iterations = max(0, args.morph_close_iterations)
    args.morph_open_iterations = max(0, args.morph_open_iterations)
    args.dynamic_calibration_frames = max(1, args.dynamic_calibration_frames)
    args.dynamic_target_fraction = float(
        np.clip(args.dynamic_target_fraction, 0.0, 1.0)
    )
    args.dynamic_ref_cache_size = max(1, args.dynamic_ref_cache_size)
    args.dynamic_lag_scale = max(0.0, args.dynamic_lag_scale)
    args.dynamic_lag_linear_start = max(0, args.dynamic_lag_linear_start)
    args.dynamic_lag_linear_max = max(0, args.dynamic_lag_linear_max)
    if args.dynamic_lag_linear and args.dynamic_lag_linear_start > args.dynamic_lag_linear_max:
        parser.error("--dynamic-lag-linear-start cannot exceed --dynamic-lag-linear-max.")
    args.annotation_segmentation_tolerance = max(0.0, args.annotation_segmentation_tolerance)
    args.annotation_segmentation_max_edge_length = max(
        0.0, args.annotation_segmentation_max_edge_length
    )
    raw_formats = [
        segment.strip().lower()
        for segment in args.annotation_formats.split(",")
        if segment.strip()
    ]
    valid_formats = {"coco", "yolov5", "darknet"}
    for fmt in raw_formats:
        if fmt not in valid_formats:
            parser.error("--annotation-formats expects coco, yolov5, and/or darknet.")
    args.annotation_formats = raw_formats
    if args.annotation_formats:
        if args.annotation_mode == "count" and args.annotation_count <= 0:
            parser.error("--annotation-count must be > 0 when --annotation-mode count is selected.")
        if args.annotation_mode == "stride" and args.annotation_stride < 1:
            parser.error("--annotation-stride must be >= 1 when --annotation-mode stride is selected.")

    return args


def parse_channel_weights(value: str, num_channels: int) -> np.ndarray:
    raw = [segment.strip() for segment in value.split(",") if segment.strip()]
    if not raw:
        raise ValueError("requires at least one numeric weight.")
    try:
        weights = [float(segment) for segment in raw]
    except ValueError as exc:
        raise ValueError("contains non-numeric values.") from exc
    if len(weights) == 1 and num_channels > 1:
        weights *= num_channels
    if len(weights) != num_channels:
        raise ValueError(
            f"expects {num_channels} weight(s), got {len(weights)}."
        )
    return np.array(weights, dtype=np.float32)


VALID_REF_MODES = {"first", "running", "prev", "absolute", "dynamic"}


def parse_ref_mode(raw: list[str]) -> ReferenceMode:
    if not raw:
        raise ValueError("requires a mode name.")
    mode = raw[0].lower()
    if mode not in VALID_REF_MODES:
        raise ValueError(
            f"unsupported option '{raw[0]}'; choose first, running, prev, or absolute."
        )
    if mode in {"first", "running", "dynamic"}:
        if len(raw) != 1:
            raise ValueError(f"'{mode}' does not accept extra arguments.")
        return ReferenceMode(mode, None)
    if len(raw) != 2:
        raise ValueError(f"'{mode}' requires a frame count (e.g., '{mode} 5').")
    try:
        value = int(raw[1])
    except ValueError as exc:
        raise ValueError("frame count must be an integer.") from exc
    if mode == "prev" and value < 1:
        raise ValueError("prev requires a positive count (>= 1).")
    if mode == "absolute" and value < 0:
        raise ValueError("absolute requires a non-negative index.")
    return ReferenceMode(mode, value)


def compute_dynamic_factor(measurements: list[tuple[float, float]]) -> Optional[float]:
    valid = [
        area / (time ** 1.5)
        for time, area in measurements
        if time > 0 and area > 0
    ]
    if not valid:
        return None
    return float(np.median(valid))


def fetch_dynamic_reference_frame(
        dynamic_state: Dict[str, Any],
    cap_reference: cv2.VideoCapture,
    index: int,
    colorspace: str,
    first_frame_converted: np.ndarray,
) -> np.ndarray:
    if index <= 1 or cap_reference is None:
        return first_frame_converted

    cache: OrderedDict[int, np.ndarray] = dynamic_state["ref_cache"]
    if index in cache:
        cache.move_to_end(index)
        return cache[index]

    cap_reference.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
    ret, frame_bgr = cap_reference.read()
    if not ret:
        return first_frame_converted

    converted = convert_frame_to_colorspace(frame_bgr, colorspace).astype(np.float32)
    cache[index] = converted
    if len(cache) > dynamic_state["ref_cache_capacity"]:
        cache.popitem(last=False)
    return converted


def select_dynamic_reference_frame(
    dynamic_state: Dict[str, Any],
    frame_index: int,
    fps: float,
    roi_pixels: int,
    cap_reference: cv2.VideoCapture,
    first_frame_converted: np.ndarray,
    colorspace: str,
) -> tuple[np.ndarray, int]:
    if dynamic_state.get("linear_mode"):
        frame_total = dynamic_state.get("total_frames")
        if frame_total and frame_total > 0:
            progress = (frame_index - 1) / frame_total
        else:
            progress = 0.0
        progress = min(1.0, max(0.0, progress))
        linear_max = dynamic_state["linear_max"]
        linear_start = dynamic_state["linear_start"]
        linear_range = max(0, linear_max - linear_start)
        delta_frames = linear_start + progress * linear_range
        delta_frames *= dynamic_state["lag_scale"]
        delta_frames = int(round(delta_frames))
        delta_frames = min(delta_frames, frame_index - 1)
        ref_index = max(1, frame_index - delta_frames)
        return fetch_dynamic_reference_frame(
            dynamic_state, cap_reference, ref_index, colorspace, first_frame_converted
        ), ref_index
    if (
        dynamic_state["factor"] is None
        or roi_pixels <= 0
        or fps <= 0
        or cap_reference is None
    ):
        return first_frame_converted, 1

    time_current = max(0.0, (frame_index - 1) / fps)
    target_area = dynamic_state["target_fraction"] * roi_pixels
    factor = max(dynamic_state["factor"], 1e-9)
    delta_t = ((target_area / factor) + math.sqrt(time_current)) ** 2 - time_current
    delta_t *= dynamic_state["lag_scale"]
    delta_t = max(0.0, delta_t)
    ref_time = max(0.0, time_current - delta_t)
    ref_index = int(ref_time * fps) + 1
    ref_index = max(1, min(ref_index, max(1, frame_index - 1)))

    reference_frame = fetch_dynamic_reference_frame(
        dynamic_state, cap_reference, ref_index, colorspace, first_frame_converted
    )
    return reference_frame, ref_index


def record_dynamic_measurement(
    dynamic_state: Dict[str, Any],
    frame_index: int,
    mask_area: int,
    fps: float,
) -> None:
    if dynamic_state["factor"] is not None or fps <= 0 or frame_index <= 1:
        return

    time_seconds = (frame_index - 1) / fps
    if time_seconds <= 0 or mask_area <= 0:
        return

    dynamic_state["measurements"].append((time_seconds, mask_area))
    if len(dynamic_state["measurements"]) >= dynamic_state["calibration_frames"]:
        factor = compute_dynamic_factor(dynamic_state["measurements"])
        if factor is not None and factor > 0:
            dynamic_state["factor"] = factor


def densify_polygon(points: list[tuple[float, float]], max_edge_length: float) -> list[tuple[float, float]]:
    if max_edge_length <= 0 or len(points) < 2:
        return points
    result: list[tuple[float, float]] = []
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        segments = max(1, int(math.ceil(dist / max_edge_length))) if dist > 0 else 1
        for s in range(segments):
            t = s / segments
            result.append((x1 + dx * t, y1 + dy * t))
    return result


def extract_bounding_boxes(
    mask: np.ndarray,
    tolerance: float,
    max_edge_length: float,
) -> list[AnnotationBox]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[AnnotationBox] = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area <= 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        approx = cv2.approxPolyDP(contour, tolerance, True) if tolerance > 0 else contour
        if approx is None or approx.size == 0:
            approx = contour
        points = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
        if max_edge_length > 0:
            points = densify_polygon(points, max_edge_length)
        segmentation = [coord for point in points for coord in point]
        if len(segmentation) < 6:
            segmentation = [
                float(x),
                float(y),
                float(x + w),
                float(y),
                float(x + w),
                float(y + h),
                float(x),
                float(y + h),
            ]
        boxes.append(AnnotationBox(x, y, w, h, area, segmentation))
    return boxes


def choose_annotation_indices(
    total: int, mode: str, count: int, stride: int
) -> list[int]:
    if total <= 0:
        return []
    if mode == "all":
        return list(range(total))
    if mode == "count":
        count = min(total, count)
        if count <= 0:
            return []
        if count == 1:
            return [0]
        positions = np.linspace(0, total - 1, count, dtype=int)
        return list(dict.fromkeys(positions.tolist()))
    if mode == "stride":
        if stride <= 0:
            return []
        return list(range(0, total, stride))
    return []


def export_coco_format(
    output_dir: Path,
    records: list[AnnotationFrame],
    selected_indices: list[int],
    width: int,
    height: int,
) -> None:
    """Export annotations in COCO format matching formatCOCO example structure."""
    coco_root = output_dir / "formatCOCO"
    ensure_dir(coco_root)
    annotations_dir = coco_root / "annotations"
    ensure_dir(annotations_dir)
    images_dir = coco_root / "images" / "default"
    ensure_dir(images_dir)

    selected_records = [records[idx] for idx in selected_indices]

    coco_output = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": "",
        },
        "categories": [
            {"id": 1, "name": "dry", "supercategory": ""},
            {"id": 2, "name": "wet", "supercategory": ""},
        ],
        "images": [],
        "annotations": [],
    }
    annotation_id = 1

    for image_id, record in enumerate(selected_records, start=1):
        frame_filename = f"frame_{record.frame_index:06d}.png"
        frame_path = images_dir / frame_filename
        cv2.imwrite(str(frame_path), record.frame_bgr)

        coco_output["images"].append(
            {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": frame_filename,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": 0,
            }
        )

        for bbox in record.boxes:
            x, y, w, h, area, segmentation = (
                bbox.x,
                bbox.y,
                bbox.w,
                bbox.h,
                bbox.area,
                bbox.segmentation,
            )
            coco_output["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [segmentation] if segmentation else [],
                    "area": area,
                    "bbox": [x, y, w, h],
                    "iscrowd": 0,
                    "attributes": {"occluded": False, "rotation": 0.0},
                }
            )
            annotation_id += 1

    coco_path = annotations_dir / "instances_default.json"
    with coco_path.open("w") as handle:
        json.dump(coco_output, handle, separators=(",", ":"))


def export_yolov5_format(
    output_dir: Path,
    records: list[AnnotationFrame],
    selected_indices: list[int],
    width: int,
    height: int,
) -> None:
    """Export annotations in YOLOv5/v8 format with segmentation polygons."""
    yolo_root = output_dir / "formatYOLO"
    ensure_dir(yolo_root)
    images_dir = yolo_root / "images" / "train"
    ensure_dir(images_dir)
    labels_dir = yolo_root / "labels" / "train"
    ensure_dir(labels_dir)

    selected_records = [records[idx] for idx in selected_indices]
    train_list = []

    for record in selected_records:
        frame_filename = f"frame_{record.frame_index:06d}.png"
        frame_path = images_dir / frame_filename
        cv2.imwrite(str(frame_path), record.frame_bgr)
        train_list.append(f"data/images/train/{frame_filename}")

        label_path = labels_dir / f"frame_{record.frame_index:06d}.txt"
        with label_path.open("w") as label_handle:
            for bbox in record.boxes:
                segmentation = bbox.segmentation
                if segmentation and len(segmentation) >= 6:
                    normalized_points = []
                    for i in range(0, len(segmentation), 2):
                        px = segmentation[i] / width
                        py = segmentation[i + 1] / height
                        normalized_points.append(f"{px:.6f}")
                        normalized_points.append(f"{py:.6f}")
                    label_handle.write(f"0 {' '.join(normalized_points)}\n")
                else:
                    x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
                    pts = [
                        f"{x/width:.6f}", f"{y/height:.6f}",
                        f"{(x+w)/width:.6f}", f"{y/height:.6f}",
                        f"{(x+w)/width:.6f}", f"{(y+h)/height:.6f}",
                        f"{x/width:.6f}", f"{(y+h)/height:.6f}",
                    ]
                    label_handle.write(f"0 {' '.join(pts)}\n")

    data_yaml = {
        "names": {0: "dry", 1: "wet"},
        "path": ".",
        "train": "train.txt",
    }
    data_yaml_path = yolo_root / "data.yaml"
    with data_yaml_path.open("w") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False)

    train_txt_path = yolo_root / "train.txt"
    with train_txt_path.open("w") as handle:
        for entry in train_list:
            handle.write(f"{entry}\n")


def export_darknet_format(
    output_dir: Path,
    records: list[AnnotationFrame],
    selected_indices: list[int],
    width: int,
    height: int,
) -> None:
    """Export annotations in Darknet YOLO format with bounding boxes."""
    darknet_root = output_dir / "formatYOLO_v2"
    ensure_dir(darknet_root)
    obj_train_data = darknet_root / "obj_train_data"
    ensure_dir(obj_train_data)

    selected_records = [records[idx] for idx in selected_indices]
    train_list = []

    for record in selected_records:
        frame_filename = f"frame_{record.frame_index:06d}.png"
        frame_path = obj_train_data / frame_filename
        cv2.imwrite(str(frame_path), record.frame_bgr)
        train_list.append(f"data/obj_train_data/{frame_filename}")

        label_path = obj_train_data / f"frame_{record.frame_index:06d}.txt"
        with label_path.open("w") as label_handle:
            for bbox in record.boxes:
                x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
                cx = (x + w / 2) / width
                cy = (y + h / 2) / height
                nw = w / width
                nh = h / height
                label_handle.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    obj_data_path = darknet_root / "obj.data"
    with obj_data_path.open("w") as handle:
        handle.write("classes = 2\n")
        handle.write("train = data/train.txt\n")
        handle.write("names = data/obj.names\n")
        handle.write("backup = backup/\n")

    obj_names_path = darknet_root / "obj.names"
    with obj_names_path.open("w") as handle:
        handle.write("dry\n")
        handle.write("wet\n")

    train_txt_path = darknet_root / "train.txt"
    with train_txt_path.open("w") as handle:
        for entry in train_list:
            handle.write(f"{entry}\n")


def export_annotation_outputs(
    args: argparse.Namespace,
    records: list[AnnotationFrame],
    selected_indices: list[int],
    width: int,
    height: int,
) -> None:
    """Export annotations in selected formats (coco, yolov5, darknet)."""
    if not records or not args.annotation_formats or not selected_indices:
        return

    if "coco" in args.annotation_formats:
        export_coco_format(args.output_dir, records, selected_indices, width, height)

    if "yolov5" in args.annotation_formats:
        export_yolov5_format(args.output_dir, records, selected_indices, width, height)

    if "darknet" in args.annotation_formats:
        export_darknet_format(args.output_dir, records, selected_indices, width, height)


def convert_frame_to_colorspace(frame_bgr: np.ndarray, colorspace: str) -> np.ndarray:
    if colorspace == "CIELAB":
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    if colorspace == "RGB":
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if colorspace == "HSV":
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    if colorspace == "GRAYSCALE":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return gray[:, :, np.newaxis]
    raise ValueError(f"Unsupported colorspace: {colorspace}")


def resolve_roi_margins(args: argparse.Namespace) -> Tuple[float, float, float, float]:
    def pick(value: Optional[float]) -> float:
        base = args.roi_margin if value is None else value
        return max(0.0, min(base, 0.49))

    top = pick(args.roi_margin_top)
    bottom = pick(args.roi_margin_bottom)
    left = pick(args.roi_margin_left)
    right = pick(args.roi_margin_right)
    return top, bottom, left, right


def build_roi_mask(
    shape: Tuple[int, int], margins: Tuple[float, float, float, float]
) -> np.ndarray:
    top_frac, bottom_frac, left_frac, right_frac = margins
    height, width = shape
    top = int(top_frac * height)
    bottom = height - int(bottom_frac * height)
    left = int(left_frac * width)
    right = width - int(right_frac * width)
    if bottom <= top:
        bottom = min(height, top + 1)
    if right <= left:
        right = min(width, left + 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 1
    return mask


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def choose_threshold(
    delta_norm: np.ndarray,
    roi_mask: np.ndarray,
    manual_threshold: Optional[float],
    percentile: Optional[float],
    offset: float,
) -> float:
    roi_pixels = delta_norm[roi_mask > 0]
    threshold_value: float

    if percentile is not None and roi_pixels.size:
        percentile = float(np.clip(percentile, 0.0, 100.0))
        threshold_value = float(np.percentile(roi_pixels, percentile))
    elif manual_threshold is not None:
        threshold_value = float(manual_threshold)
    else:
        otsu_value, _ = cv2.threshold(
            delta_norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        threshold_value = float(otsu_value)

    threshold_value = threshold_value + offset
    return float(np.clip(threshold_value, 0.0, 255.0))


def detect_front(
    frame_bgr: np.ndarray,
    frame_converted: np.ndarray,
    reference_converted: np.ndarray,
    roi_mask: np.ndarray,
    blur_kernel: int,
    morph_kernel: int,
    min_area: int,
    manual_threshold: Optional[float],
    percentile_threshold: Optional[float],
    threshold_offset: float,
    channel_weights: np.ndarray,
    blur_enabled: bool,
    morph_shape: str,
    morph_close_iterations: int,
    morph_open_iterations: int,
    darken_only: bool = False,
    peak_brightness_map: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    weights = channel_weights.reshape(1, 1, -1)
    if darken_only:
        # Only detect darkening: reference - frame (positive when frame is darker)
        # For LAB, channel 0 is L* (lightness), for RGB/grayscale lower = darker
        if peak_brightness_map is not None:
            # Use the historical peak brightness as reference
            current_brightness = frame_converted[:, :, 0]
            delta = (peak_brightness_map - current_brightness) * weights[0, 0, 0]
        else:
            diff = (reference_converted - frame_converted) * weights
            # Use first channel (L* for LAB, or intensity) for signed comparison
            delta = diff[:, :, 0] if diff.shape[2] > 0 else diff[:, :, 0]
        # Zero out negative values (brightening)
        delta = np.maximum(delta, 0)
    else:
        diff = (frame_converted - reference_converted) * weights
        delta = np.sqrt(np.sum(diff * diff, axis=2))
    delta *= roi_mask

    if blur_enabled:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        delta_blur = cv2.GaussianBlur(delta, (blur_kernel, blur_kernel), 0)
    else:
        delta_blur = delta

    delta_norm = cv2.normalize(delta_blur, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )

    threshold_value = choose_threshold(
        delta_norm,
        roi_mask,
        manual_threshold,
        percentile_threshold,
        threshold_offset,
    )
    _, binary = cv2.threshold(delta_norm, threshold_value, 255, cv2.THRESH_BINARY)

    kernel_size = morph_kernel + (1 - morph_kernel % 2)
    kernel_shape = MORPH_SHAPE_MAP.get(morph_shape, cv2.MORPH_ELLIPSE)
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    if morph_close_iterations > 0:
        for _ in range(morph_close_iterations):
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if morph_open_iterations > 0:
        for _ in range(morph_open_iterations):
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        filtered = np.zeros_like(binary)
        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == idx] = 255
        binary = filtered

    heatmap = cv2.applyColorMap(delta_norm, cv2.COLORMAP_TURBO)
    return binary, heatmap


def create_overlay(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, edge_kernel)
    overlay = frame_bgr.copy()
    overlay[edges > 0] = (0, 0, 255)
    return overlay


def apply_locking(
    mask: np.ndarray, lock_frames: int, state: Optional[Dict[str, np.ndarray]]
) -> np.ndarray:
    if lock_frames <= 0 or state is None:
        return mask

    filled = mask > 0
    state["counter"][filled] = np.minimum(
        state["counter"][filled] + 1, np.iinfo(np.uint16).max
    )
    state["counter"][~filled] = 0
    state["locked"][state["counter"] >= lock_frames] = 255
    return np.maximum(mask, state["locked"])


def write_run_summary(path: Path, summary: Dict[str, Any]) -> None:
    with path.open("w") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video_path}")

    ret, first_frame_bgr = cap.read()
    if not ret:
        raise RuntimeError("Failed to read reference frame.")

    first_frame_converted = (
        convert_frame_to_colorspace(first_frame_bgr, args.colorspace).astype(np.float32)
    )
    # Initialize peak brightness tracking (first channel, e.g., L* for LAB)
    peak_brightness_map: Optional[np.ndarray] = None
    if args.peak_reference:
        peak_brightness_map = first_frame_converted[:, :, 0].copy()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    reference_mode = args.ref_mode
    absolute_reference = first_frame_converted
    absolute_index = reference_mode.offset if reference_mode.name == "absolute" else None

    if reference_mode.name == "absolute":
        if absolute_index is None:
            raise RuntimeError("absolute reference mode requires a frame index.")
        if total_frames is not None and absolute_index >= total_frames:
            raise RuntimeError("Requested absolute frame index exceeds video length.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, absolute_index)
        ret, absolute_frame_bgr = cap.read()
        if not ret:
            raise RuntimeError(f"Unable to read absolute reference frame {absolute_index}.")
        absolute_reference = convert_frame_to_colorspace(
            absolute_frame_bgr, args.colorspace
        ).astype(np.float32)
    running_reference = first_frame_converted.copy()
    prev_buffer = (
        deque(maxlen=reference_mode.offset)
        if reference_mode.name == "prev"
        else None
    )
    dynamic_state: Optional[Dict[str, Any]] = None
    dynamic_capture: Optional[cv2.VideoCapture] = None
    dynamic_lag_log: list[tuple[int, int]] = []
    annotation_enabled = bool(args.annotation_formats)
    processed_frame_records: list[AnnotationFrame] = []
    if reference_mode.name == "dynamic":
        dynamic_state = {
            "calibration_frames": args.dynamic_calibration_frames,
            "target_fraction": args.dynamic_target_fraction,
            "measurements": [],
            "factor": None,
            "ref_cache": OrderedDict(),
            "ref_cache_capacity": args.dynamic_ref_cache_size,
            "first_lag": None,
            "last_lag": None,
            "lag_scale": args.dynamic_lag_scale,
            "linear_mode": args.dynamic_lag_linear,
            "linear_max": args.dynamic_lag_linear_max,
            "linear_start": args.dynamic_lag_linear_start,
        }
        dynamic_capture = cv2.VideoCapture(str(args.video_path))
        if not dynamic_capture.isOpened():
            raise RuntimeError(
                "Unable to open video for dynamic reference access."
            )
        dynamic_state["total_frames"] = total_frames

    roi_margins = resolve_roi_margins(args)
    roi_mask = build_roi_mask(first_frame_converted.shape[:2], roi_margins)
    lock_frames = max(args.lock_frames, 0)
    lock_state: Optional[Dict[str, np.ndarray]] = None
    if lock_frames > 0:
        lock_state = {
            "counter": np.zeros(first_frame_converted.shape[:2], dtype=np.uint16),
            "locked": np.zeros(first_frame_converted.shape[:2], dtype=np.uint8),
        }
    roi_pixels = int(roi_mask.sum())
    if dynamic_state is not None:
        dynamic_state["roi_pixels"] = roi_pixels

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    processing_time_accum = 0.0
    run_clock_start = time.monotonic()

    mask_dir = args.output_dir / "masks"
    overlay_dir = args.output_dir / "overlays"
    heatmap_dir = args.output_dir / "heatmap"
    outputs_to_write = {
        "mask_png": args.write_mask_pngs,
        "overlay_png": args.write_overlay_pngs,
        "heatmap_png": args.write_heatmap_pngs,
    }
    if outputs_to_write["mask_png"]:
        ensure_dir(mask_dir)
    if outputs_to_write["overlay_png"]:
        ensure_dir(overlay_dir)
    if outputs_to_write["heatmap_png"]:
        ensure_dir(heatmap_dir)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mask_writer = overlay_writer = heat_writer = None
    video_dir = args.output_dir / "videos"
    if args.write_mask_video or args.write_overlay_video or args.write_heatmap_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ensure_dir(video_dir)
        if args.write_mask_video:
            mask_writer = cv2.VideoWriter(
                str(video_dir / "mask.mp4"), fourcc, fps, (width, height), False
            )
            if not mask_writer.isOpened():
                raise RuntimeError("Failed to open mask video writer.")
        if args.write_overlay_video:
            overlay_writer = cv2.VideoWriter(
                str(video_dir / "overlay.mp4"), fourcc, fps, (width, height)
            )
            if not overlay_writer.isOpened():
                raise RuntimeError("Failed to open overlay video writer.")
        if args.write_heatmap_video:
            heat_writer = cv2.VideoWriter(
                str(video_dir / "heatmap.mp4"), fourcc, fps, (width, height)
            )
            if not heat_writer.isOpened():
                raise RuntimeError("Failed to open heatmap video writer.")

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
            frame_converted = convert_frame_to_colorspace(
                frame_bgr, args.colorspace
            ).astype(np.float32)
            should_process = frame_index % args.frame_step == 0

            reference_for_frame: np.ndarray
            reference_frame_index = 1
            if reference_mode.name == "first":
                reference_for_frame = first_frame_converted
            elif reference_mode.name == "absolute":
                reference_for_frame = absolute_reference
                reference_frame_index = absolute_index or 1
            elif reference_mode.name == "running":
                reference_for_frame = running_reference
            elif reference_mode.name == "prev":
                if prev_buffer is not None and len(prev_buffer) >= reference_mode.offset:
                    reference_for_frame = prev_buffer[0]
                else:
                    reference_for_frame = first_frame_converted
            elif reference_mode.name == "dynamic":
                reference_for_frame, reference_frame_index = select_dynamic_reference_frame(
                    dynamic_state,
                    frame_index,
                    fps,
                    roi_pixels,
                    dynamic_capture,
                    first_frame_converted,
                    args.colorspace,
                )
            else:  # pragma: no cover
                reference_for_frame = first_frame_converted

            if should_process:
                compute_start = time.monotonic()
                # Update peak brightness map with current frame (before detection)
                if peak_brightness_map is not None:
                    current_brightness = frame_converted[:, :, 0]
                    peak_brightness_map = np.maximum(peak_brightness_map, current_brightness)
                mask, heatmap = detect_front(
                    frame_bgr,
                    frame_converted,
                    reference_for_frame,
                    roi_mask,
                    args.blur_kernel,
                    args.morph_kernel,
                    args.min_area,
                    args.contrast_threshold,
                    args.contrast_percentile,
                    args.threshold_offset,
                    args.channel_weights,
                    not args.skip_blur,
                    args.morph_shape,
                    args.morph_close_iterations,
                    args.morph_open_iterations,
                    args.darken_only,
                    peak_brightness_map if args.peak_reference else None,
                )
                mask = apply_locking(mask, lock_frames, lock_state)
                overlay = create_overlay(frame_bgr, mask)
                if annotation_enabled:
                    boxes = extract_bounding_boxes(
                        mask,
                        args.annotation_segmentation_tolerance,
                        args.annotation_segmentation_max_edge_length,
                    )
                    processed_frame_records.append(
                        AnnotationFrame(frame_index, frame_bgr.copy(), boxes)
                    )

                basename = f"frame_{frame_index:06d}.png"
                if outputs_to_write["mask_png"]:
                    cv2.imwrite(str(mask_dir / basename), mask)
                if outputs_to_write["overlay_png"]:
                    cv2.imwrite(str(overlay_dir / basename), overlay)
                if outputs_to_write["heatmap_png"]:
                    cv2.imwrite(str(heatmap_dir / basename), heatmap)

                if mask_writer is not None:
                    mask_writer.write(mask)
                if overlay_writer is not None:
                    overlay_writer.write(overlay)
                if heat_writer is not None:
                    heat_writer.write(heatmap)
                if dynamic_state is not None:
                    lag = frame_index - reference_frame_index
                    if dynamic_state["first_lag"] is None:
                        dynamic_state["first_lag"] = lag
                    dynamic_state["last_lag"] = lag
                    dynamic_lag_log.append((frame_index, lag))
                    mask_area = int(np.count_nonzero(mask))
                    record_dynamic_measurement(dynamic_state, frame_index, mask_area, fps)

                processed += 1
                pbar.update(1)
                processing_time_accum += time.monotonic() - compute_start

            if reference_mode.name == "prev" and prev_buffer is not None:
                prev_buffer.append(frame_converted)
            if reference_mode.name == "running":
                running_reference = (
                    (1 - args.ref_running_alpha) * running_reference
                    + args.ref_running_alpha * frame_converted
                )

    cap.release()
    if dynamic_capture is not None:
        dynamic_capture.release()
    if mask_writer is not None:
        mask_writer.release()
    if overlay_writer is not None:
        overlay_writer.release()
    if heat_writer is not None:
        heat_writer.release()
    if annotation_enabled:
        selection = choose_annotation_indices(
            len(processed_frame_records),
            args.annotation_mode,
            args.annotation_count,
            args.annotation_stride,
        )
        export_annotation_outputs(
            args, processed_frame_records, selection, width, height
        )

    run_total_time = time.monotonic() - run_clock_start
    roi_pixels = int(roi_mask.sum())
    roi_fraction = (
        roi_pixels / roi_mask.size if roi_mask.size else 0.0
    )
    avg_compute_time = (
        processing_time_accum / processed if processed else 0.0
    )
    video_duration = (
        float(total_frames) / fps if total_frames and fps else None
    )
    summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "video_path": str(args.video_path),
        "output_dir": str(args.output_dir),
        "frame_step": args.frame_step,
        "total_frames": total_frames,
        "processed_frames": processed,
        "video_fps": fps,
        "video_duration_seconds": video_duration,
        "reference_mode": reference_mode.name,
        "reference_offset": reference_mode.offset,
        "absolute_reference_index": absolute_index,
        "ref_running_alpha": args.ref_running_alpha
        if reference_mode.name == "running"
        else None,
        "dynamic_calibration_frames": args.dynamic_calibration_frames
        if reference_mode.name == "dynamic"
        else None,
        "dynamic_target_fraction": args.dynamic_target_fraction
        if reference_mode.name == "dynamic"
        else None,
        "dynamic_lag_scale": args.dynamic_lag_scale
        if reference_mode.name == "dynamic"
        else None,
        "dynamic_lag_linear": args.dynamic_lag_linear
        if reference_mode.name == "dynamic"
        else None,
        "dynamic_lag_linear_max": args.dynamic_lag_linear_max
        if reference_mode.name == "dynamic" and args.dynamic_lag_linear
        else None,
        "dynamic_lag_linear_start": args.dynamic_lag_linear_start
        if reference_mode.name == "dynamic" and args.dynamic_lag_linear
        else None,
        "dynamic_reference_factor": dynamic_state["factor"]
        if dynamic_state is not None
        else None,
        "dynamic_reference_start_lag": dynamic_state["first_lag"]
        if dynamic_state is not None
        else None,
        "dynamic_reference_end_lag": dynamic_state["last_lag"]
        if dynamic_state is not None
        else None,
        "annotation_formats": args.annotation_formats or None,
        "annotation_mode": args.annotation_mode,
        "annotation_count": args.annotation_count
        if args.annotation_mode == "count"
        else None,
        "annotation_stride": args.annotation_stride
        if args.annotation_mode == "stride"
        else None,
        "annotation_segmentation_tolerance": args.annotation_segmentation_tolerance
        if args.annotation_formats
        else None,
        "annotation_segmentation_max_edge_length": args.annotation_segmentation_max_edge_length
        if args.annotation_formats
        else None,
        "colorspace": args.colorspace,
        "channel_weights": args.channel_weights.tolist(),
        "lock_frames": lock_frames,
        "roi_margin": args.roi_margin,
        "roi_margin_top": args.roi_margin_top,
        "roi_margin_bottom": args.roi_margin_bottom,
        "roi_margin_left": args.roi_margin_left,
        "roi_margin_right": args.roi_margin_right,
        "blur_kernel": args.blur_kernel,
        "skip_blur": args.skip_blur,
        "morph_kernel": args.morph_kernel,
        "morph_shape": args.morph_shape,
        "morph_close_iterations": args.morph_close_iterations,
        "morph_open_iterations": args.morph_open_iterations,
        "min_area": args.min_area,
        "contrast_threshold": args.contrast_threshold,
        "contrast_percentile": args.contrast_percentile,
        "threshold_offset": args.threshold_offset,
        "darken_only": args.darken_only,
        "peak_reference": args.peak_reference,
        "write_mask_pngs": bool(outputs_to_write["mask_png"]),
        "write_overlay_pngs": bool(outputs_to_write["overlay_png"]),
        "write_heatmap_pngs": bool(outputs_to_write["heatmap_png"]),
        "write_videos": args.write_videos,
        "write_mask_video": args.write_mask_video,
        "write_overlay_video": args.write_overlay_video,
        "write_heatmap_video": args.write_heatmap_video,
        "roi_pixel_count": roi_pixels,
        "roi_fraction": roi_fraction,
        "average_compute_time_seconds": avg_compute_time,
        "total_run_time_seconds": run_total_time,
    }
    summary_path = args.output_dir / "run_summary.yaml"
    write_run_summary(summary_path, summary)
    if args.dynamic_lag_log and dynamic_lag_log:
        args.dynamic_lag_log.parent.mkdir(parents=True, exist_ok=True)
        with args.dynamic_lag_log.open("w") as handle:
            handle.write("frame,lag\n")
            for frame_idx_entry, lag_entry in dynamic_lag_log:
                handle.write(f"{frame_idx_entry},{lag_entry}\n")

    print(
        f"Processed {processed} frames. Outputs saved to {args.output_dir}. "
        f"Summary written to {summary_path}"
    )


if __name__ == "__main__":
    main()
