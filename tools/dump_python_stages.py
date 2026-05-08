#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_vrifa_module():
    module_name = "vrifa_reference"
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "vrifa.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load vrifa.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_cli() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", required=True, help="Comma-separated 1-based frame indices.")
    return parser.parse_known_args()


def parse_frames(raw: str) -> list[int]:
    frames = []
    for segment in raw.split(","):
        segment = segment.strip()
        if not segment:
            continue
        frame = int(segment)
        if frame < 1:
            raise ValueError("frame indices are 1-based and must be >= 1")
        frames.append(frame)
    if not frames:
        raise ValueError("at least one frame index is required")
    return frames


def build_reference_args(vrifa: Any, forwarded: list[str]):
    original = sys.argv[:]
    try:
        sys.argv = ["vrifa.py", *forwarded]
        return vrifa.parse_args()
    finally:
        sys.argv = original


def detect_front_debug(
    vrifa: Any,
    frame_converted: np.ndarray,
    reference_converted: np.ndarray,
    roi_mask: np.ndarray,
    blur_kernel: int,
    morph_kernel: int,
    min_area: int,
    manual_threshold: float | None,
    percentile_threshold: float | None,
    threshold_offset: float,
    channel_weights: np.ndarray,
    blur_enabled: bool,
    morph_shape: str,
    morph_close_iterations: int,
    morph_open_iterations: int,
    darken_only: bool,
    peak_brightness_map: np.ndarray | None,
) -> dict[str, np.ndarray]:
    weights = channel_weights.reshape(1, 1, -1)
    if darken_only:
        if peak_brightness_map is not None:
            current_brightness = frame_converted[:, :, 0]
            delta = (peak_brightness_map - current_brightness) * weights[0, 0, 0]
        else:
            diff = (reference_converted - frame_converted) * weights
            delta = diff[:, :, 0]
        delta = np.maximum(delta, 0)
    else:
        diff = (frame_converted - reference_converted) * weights
        delta = np.sqrt(np.sum(diff * diff, axis=2))
    delta = (delta * roi_mask).astype(np.float32, copy=False)

    if blur_enabled:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        delta_blur = cv2.GaussianBlur(delta, (blur_kernel, blur_kernel), 0)
    else:
        delta_blur = delta.copy()

    delta_norm = cv2.normalize(delta_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    threshold_value = vrifa.choose_threshold(
        delta_norm,
        roi_mask,
        manual_threshold,
        percentile_threshold,
        threshold_offset,
    )
    _, binary = cv2.threshold(delta_norm, threshold_value, 255, cv2.THRESH_BINARY)
    binary_pre_morph = binary.copy()

    kernel_size = morph_kernel + (1 - morph_kernel % 2)
    kernel_shape = vrifa.MORPH_SHAPE_MAP.get(morph_shape, cv2.MORPH_ELLIPSE)
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
    return {
        "delta": delta,
        "delta_blur": delta_blur.astype(np.float32, copy=False),
        "delta_norm": delta_norm,
        "binary": binary_pre_morph,
        "mask_pre_lock": binary,
        "heatmap": heatmap,
    }


def save_stage_dump(output_dir: Path, frame_index: int, stages: dict[str, np.ndarray]) -> None:
    frame_dir = output_dir / f"frame_{frame_index:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for name, array in stages.items():
        np.save(frame_dir / f"{name}.npy", array, allow_pickle=False)


def main() -> int:
    cli_args, forwarded = parse_cli()
    frames = parse_frames(cli_args.frames)
    targets = set(frames)
    vrifa = load_vrifa_module()
    args = build_reference_args(vrifa, forwarded)

    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open {args.video_path}")
    ret, first_frame_bgr = cap.read()
    if not ret:
        raise RuntimeError("failed to read first frame")

    first_frame_converted = vrifa.convert_frame_to_colorspace(first_frame_bgr, args.colorspace).astype(
        np.float32
    )
    peak_brightness_map = first_frame_converted[:, :, 0].copy() if args.peak_reference else None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    reference_mode = args.ref_mode
    absolute_reference = first_frame_converted
    absolute_index = reference_mode.offset if reference_mode.name == "absolute" else None
    if reference_mode.name == "absolute":
        if absolute_index is None:
            raise RuntimeError("absolute reference mode requires a frame index")
        if total_frames is not None and absolute_index >= total_frames:
            raise RuntimeError("requested absolute frame index exceeds video length")
        cap.set(cv2.CAP_PROP_POS_FRAMES, absolute_index)
        ret, absolute_frame_bgr = cap.read()
        if not ret:
            raise RuntimeError(f"unable to read absolute reference frame {absolute_index}")
        absolute_reference = vrifa.convert_frame_to_colorspace(
            absolute_frame_bgr, args.colorspace
        ).astype(np.float32)

    running_reference = first_frame_converted.copy()
    prev_buffer = deque(maxlen=reference_mode.offset) if reference_mode.name == "prev" else None
    dynamic_state: dict[str, Any] | None = None
    dynamic_capture: cv2.VideoCapture | None = None
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
            "total_frames": total_frames,
        }
        dynamic_capture = cv2.VideoCapture(str(args.video_path))
        if not dynamic_capture.isOpened():
            raise RuntimeError("unable to open dynamic reference capture")

    roi_margins = vrifa.resolve_roi_margins(args)
    roi_mask = vrifa.build_roi_mask(first_frame_converted.shape[:2], roi_margins)
    roi_pixels = int(roi_mask.sum())
    if dynamic_state is not None:
        dynamic_state["roi_pixels"] = roi_pixels
    lock_state = None
    if args.lock_frames > 0:
        lock_state = {
            "counter": np.zeros(first_frame_converted.shape[:2], dtype=np.uint16),
            "locked": np.zeros(first_frame_converted.shape[:2], dtype=np.uint8),
        }

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    dumped: set[int] = set()
    frame_index = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_index += 1
        frame_converted = vrifa.convert_frame_to_colorspace(frame_bgr, args.colorspace).astype(
            np.float32
        )
        should_process = frame_index % args.frame_step == 0

        if reference_mode.name == "first":
            reference_for_frame = first_frame_converted
            reference_frame_index = 1
        elif reference_mode.name == "absolute":
            reference_for_frame = absolute_reference
            reference_frame_index = absolute_index or 1
        elif reference_mode.name == "running":
            reference_for_frame = running_reference
            reference_frame_index = 1
        elif reference_mode.name == "prev":
            if prev_buffer is not None and len(prev_buffer) >= reference_mode.offset:
                reference_for_frame = prev_buffer[0]
            else:
                reference_for_frame = first_frame_converted
            reference_frame_index = 1
        elif reference_mode.name == "dynamic":
            reference_for_frame, reference_frame_index = vrifa.select_dynamic_reference_frame(
                dynamic_state,
                frame_index,
                cap.get(cv2.CAP_PROP_FPS) or 30.0,
                roi_pixels,
                dynamic_capture,
                first_frame_converted,
                args.colorspace,
            )
        else:
            reference_for_frame = first_frame_converted
            reference_frame_index = 1

        if should_process:
            if peak_brightness_map is not None:
                current_brightness = frame_converted[:, :, 0]
                peak_brightness_map = np.maximum(peak_brightness_map, current_brightness)

            stages = detect_front_debug(
                vrifa,
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
            mask = vrifa.apply_locking(stages["mask_pre_lock"], args.lock_frames, lock_state)
            overlay = vrifa.create_overlay(frame_bgr, mask)

            if frame_index in targets:
                save_stage_dump(
                    cli_args.output_dir,
                    frame_index,
                    {
                        "frame_converted": frame_converted,
                        "delta": stages["delta"],
                        "delta_blur": stages["delta_blur"],
                        "delta_norm": stages["delta_norm"],
                        "binary": stages["binary"],
                        "mask_pre_lock": stages["mask_pre_lock"],
                        "mask": mask,
                        "overlay": overlay,
                        "heatmap": stages["heatmap"],
                    },
                )
                dumped.add(frame_index)
                if dumped == targets:
                    break

            if dynamic_state is not None:
                lag = frame_index - reference_frame_index
                if dynamic_state["first_lag"] is None:
                    dynamic_state["first_lag"] = lag
                dynamic_state["last_lag"] = lag
                mask_area = int(np.count_nonzero(mask))
                vrifa.record_dynamic_measurement(dynamic_state, frame_index, mask_area, cap.get(cv2.CAP_PROP_FPS) or 30.0)

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

    missing = sorted(targets - dumped)
    if missing:
        raise RuntimeError(f"failed to dump frame(s): {missing}")
    print(f"Dumped Python stage intermediates for {sorted(dumped)} to {cli_args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
