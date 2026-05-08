#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = Path(__file__).resolve().parent
MAX_NPY_BYTES = 5 * 1024 * 1024
COMPRESS_ABOVE_BYTES = 1 * 1024 * 1024
STAGE_NAMES = (
    "frame_converted",
    "delta",
    "delta_blur",
    "delta_norm",
    "binary",
    "mask_pre_lock",
    "mask",
    "overlay",
    "heatmap",
)
SPECS = {
    "input_1": {
        "video_path": REPO_ROOT / "data/input_1.mp4",
        "frames": [50, 200, 500],
        "forwarded_args": ["--video-path", "data/input_1.mp4", "--roi-margin", "0.15"],
        "save_peak_fixture": False,
        "save_reference_fixture": False,
        "save_peak_reference_frames": [],
    },
    "input_2": {
        "video_path": REPO_ROOT / "data/input_2.mp4",
        "frames": [30, 60, 90],
        "forwarded_args": ["--video-path", "data/input_2.mp4", "--roi-margin", "0.0"],
        "save_peak_fixture": True,
        "save_reference_fixture": False,
        "save_peak_reference_frames": [30],
    },
}


def load_vrifa_module():
    spec = importlib.util.spec_from_file_location("vrifa_reference", REPO_ROOT / "vrifa.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load vrifa.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["vrifa_reference"] = module
    spec.loader.exec_module(module)
    return module


def parse_args(vrifa: Any, forwarded: list[str]):
    original = sys.argv[:]
    try:
        sys.argv = ["vrifa.py", *forwarded]
        return vrifa.parse_args()
    finally:
        sys.argv = original


def run_stage_dump(output_dir: Path, frames: list[int], forwarded_args: list[str]) -> None:
    cmd = [
        "python3",
        "tools/dump_python_stages.py",
        "--output-dir",
        str(output_dir),
        "--frames",
        ",".join(str(frame) for frame in frames),
        *forwarded_args,
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def save_array(base_path: Path, array: np.ndarray) -> None:
    npy_path = base_path.with_suffix(".npy")
    npz_path = base_path.with_suffix(".npz")
    if npy_path.exists():
        npy_path.unlink()
    if npz_path.exists():
        npz_path.unlink()
    if array.nbytes > MAX_NPY_BYTES or array.nbytes > COMPRESS_ABOVE_BYTES:
        np.savez_compressed(npz_path, data=array)
    else:
        np.save(npy_path, array, allow_pickle=False)


def read_frame_bgr(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"unable to read frame {frame_index} from {video_path}")
        return frame
    finally:
        cap.release()


def write_source_png(video_path: Path, frame_index: int, output_path: Path) -> None:
    frame = read_frame_bgr(video_path, frame_index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 9]):
        raise RuntimeError(f"unable to write {output_path}")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def generate_lock_fixture(vrifa: Any) -> None:
    lock_root = FIXTURE_ROOT / "lock"
    shutil.rmtree(lock_root, ignore_errors=True)
    lock_root.mkdir(parents=True, exist_ok=True)
    sequence = np.zeros((5, 8, 8), dtype=np.uint8)
    sequence[0, 2:5, 2:5] = 255
    sequence[1, 2:5, 2:5] = 255
    sequence[1, 0, 0] = 255
    sequence[2, 2:5, 2:5] = 255
    sequence[2, 5:7, 5:7] = 255
    sequence[3, 2:5, 2:5] = 255
    sequence[3, 5:7, 5:7] = 255
    sequence[4, 5:7, 5:7] = 255

    state = {
        "counter": np.zeros(sequence.shape[1:], dtype=np.uint16),
        "locked": np.zeros(sequence.shape[1:], dtype=np.uint8),
    }
    output = None
    for mask in sequence:
        output = vrifa.apply_locking(mask, 3, state)
    save_array(lock_root / "mask_sequence", sequence)
    save_array(lock_root / "locked_mask", output)
    save_array(lock_root / "lock_frames", np.array(3, dtype=np.int32))


def generate_reference_dynamic_fixture(vrifa: Any) -> None:
    dynamic_root = FIXTURE_ROOT / "reference_dynamic"
    shutil.rmtree(dynamic_root, ignore_errors=True)
    dynamic_root.mkdir(parents=True, exist_ok=True)

    measurements = np.array(
        [
            [0.5, 120.0],
            [1.0, 305.0],
            [1.5, 540.0],
            [2.0, 860.0],
            [2.5, 1200.0],
        ],
        dtype=np.float32,
    )
    factor = float(vrifa.compute_dynamic_factor([tuple(row) for row in measurements.tolist()]))
    params = {
        "factor": factor,
        "frame_index": 180,
        "fps": 29.97,
        "roi_pixels": 1280 * 720,
        "target_fraction": 0.0125,
        "lag_scale": 0.8,
        "linear_mode": False,
        "linear_start": 10,
        "linear_max": 120,
        "total_frames": 706,
    }
    time_current = max(0.0, (params["frame_index"] - 1) / params["fps"])
    target_area = params["target_fraction"] * params["roi_pixels"]
    delta_t = ((target_area / max(factor, 1e-9)) + np.sqrt(time_current)) ** 2 - time_current
    delta_t *= params["lag_scale"]
    delta_t = max(0.0, float(delta_t))
    ref_time = max(0.0, time_current - delta_t)
    ref_index = int(ref_time * params["fps"]) + 1
    ref_index = max(1, min(ref_index, max(1, params["frame_index"] - 1)))

    save_array(dynamic_root / "measurements", measurements)
    save_json(dynamic_root / "params.json", params)
    save_json(
        dynamic_root / "expected.json",
        {
            "factor": factor,
            "delta_t": delta_t,
            "reference_index": ref_index,
        },
    )


def generate_input_fixtures(vrifa: Any, name: str, spec: dict[str, Any]) -> None:
    args = parse_args(vrifa, spec["forwarded_args"])
    destination_root = FIXTURE_ROOT / name
    shutil.rmtree(destination_root, ignore_errors=True)
    destination_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"vrifa_{name}_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        run_stage_dump(tmp_root, spec["frames"], spec["forwarded_args"])

        reference_bgr = read_frame_bgr(spec["video_path"], 1)
        reference_converted = vrifa.convert_frame_to_colorspace(reference_bgr, args.colorspace).astype(
            np.float32
        )
        if spec.get("save_reference_fixture"):
            reference_root = destination_root / "reference"
            reference_root.mkdir(parents=True, exist_ok=True)
            save_array(reference_root / "frame_converted", reference_converted)

        roi_mask = vrifa.build_roi_mask(
            reference_converted.shape[:2],
            vrifa.resolve_roi_margins(args),
        )
        save_array(destination_root / "roi_mask", roi_mask)
        save_json(
            destination_root / "config.json",
            {
                "colorspace": args.colorspace,
                "blur_kernel": args.blur_kernel,
                "morph_kernel": args.morph_kernel,
                "min_area": args.min_area,
                "manual_threshold": args.contrast_threshold,
                "percentile_threshold": args.contrast_percentile,
                "threshold_offset": args.threshold_offset,
                "channel_weights": [float(value) for value in args.channel_weights.tolist()],
                "blur_enabled": not args.skip_blur,
                "morph_shape": args.morph_shape,
                "morph_close_iterations": args.morph_close_iterations,
                "morph_open_iterations": args.morph_open_iterations,
                "darken_only": args.darken_only,
                "peak_reference": args.peak_reference,
                "annotation_segmentation_tolerance": args.annotation_segmentation_tolerance,
                "annotation_segmentation_max_edge_length": args.annotation_segmentation_max_edge_length,
                "lock_frames": args.lock_frames,
            },
        )

        peak_sequence = []
        for frame_index in spec["frames"]:
            frame_dir = destination_root / f"frame_{frame_index:06d}"
            frame_dir.mkdir(parents=True, exist_ok=True)
            write_source_png(spec["video_path"], frame_index, frame_dir / "source.png")

            dumped_frame_dir = tmp_root / f"frame_{frame_index:06d}"
            for stage_name in STAGE_NAMES:
                stage = np.load(dumped_frame_dir / f"{stage_name}.npy")
                save_array(frame_dir / stage_name, stage)

            delta_norm = np.load(dumped_frame_dir / "delta_norm.npy")
            threshold_value = np.array(
                vrifa.choose_threshold(
                    delta_norm,
                    roi_mask,
                    args.contrast_threshold,
                    args.contrast_percentile,
                    args.threshold_offset,
                ),
                dtype=np.float32,
            )
            save_array(frame_dir / "threshold", threshold_value)

            mask = np.load(dumped_frame_dir / "mask.npy")
            boxes = vrifa.extract_bounding_boxes(
                mask,
                args.annotation_segmentation_tolerance,
                args.annotation_segmentation_max_edge_length,
            )
            rows = sorted(
                [
                    [box.x, box.y, box.w, box.h, box.area, len(box.segmentation)]
                    for box in boxes
                ],
                key=lambda row: (row[1], row[0], row[2], row[3]),
            )
            contours = np.asarray(rows, dtype=np.int32).reshape((-1, 6)) if rows else np.zeros((0, 6), dtype=np.int32)
            save_array(frame_dir / "contours_boxes", contours)

            peak_sequence.append(np.load(dumped_frame_dir / "frame_converted.npy"))

        if spec.get("save_peak_fixture"):
            peak_map = np.zeros(peak_sequence[0].shape[:2], dtype=np.float32)
            for frame_converted in peak_sequence:
                peak_map = np.maximum(peak_map, frame_converted[:, :, 0])
            save_array(destination_root / "peak_after_3", peak_map)

        peak_reference_frames = set(spec.get("save_peak_reference_frames", []))
        if args.peak_reference and peak_reference_frames:
            cap = cv2.VideoCapture(str(spec["video_path"]))
            if not cap.isOpened():
                raise RuntimeError(f"unable to open {spec['video_path']}")
            try:
                peak_map = None
                frame_index = 0
                target_limit = max(peak_reference_frames)
                while frame_index < target_limit:
                    ok, frame_bgr = cap.read()
                    if not ok:
                        raise RuntimeError(
                            f"unable to read frame {frame_index + 1} from {spec['video_path']}"
                        )
                    frame_index += 1
                    frame_converted = vrifa.convert_frame_to_colorspace(frame_bgr, args.colorspace).astype(
                        np.float32
                    )
                    if peak_map is None:
                        peak_map = frame_converted[:, :, 0].copy()
                    else:
                        peak_map = np.maximum(peak_map, frame_converted[:, :, 0])
                    if frame_index in peak_reference_frames:
                        save_array(destination_root / f"frame_{frame_index:06d}" / "peak_before", peak_map)
            finally:
                cap.release()


def main() -> int:
    fixture_root = FIXTURE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=fixture_root,
        help="Destination fixture root. Defaults to vrifa-rs/tests/fixtures.",
    )
    args = parser.parse_args()
    fixture_root = args.fixture_root.resolve()
    fixture_root.mkdir(parents=True, exist_ok=True)
    globals()["FIXTURE_ROOT"] = fixture_root

    vrifa = load_vrifa_module()
    for name, spec in SPECS.items():
        generate_input_fixtures(vrifa, name, spec)
    generate_lock_fixture(vrifa)
    generate_reference_dynamic_fixture(vrifa)
    print(f"Wrote stage fixtures to {fixture_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
