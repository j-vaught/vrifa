#!/usr/bin/env python3
"""Parallel parameter sweep for VRIFA input 12-14 annotation tuning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


VIDEOS = {
    "input_12": [49, 243, 486, 730, 924],
    "input_13": [60, 302, 604, 905, 1147],
    "input_14": [27, 137, 274, 412, 522],
}


@dataclass(frozen=True)
class Case:
    case_id: str
    params: dict[str, Any]
    args: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    home = pathlib.Path.home()
    default_root = home / "vrifa_eval" / "vrifa"
    parser.add_argument("--root", type=pathlib.Path, default=default_root)
    parser.add_argument(
        "--bin",
        type=pathlib.Path,
        default=default_root / "vrifa-rs" / "target" / "release" / "vrifa",
    )
    parser.add_argument(
        "--video-root", type=pathlib.Path, default=default_root / "data"
    )
    parser.add_argument(
        "--annot-root",
        type=pathlib.Path,
        default=default_root / "annotation_work" / "input_12_14_final",
    )
    parser.add_argument(
        "--roi-root",
        type=pathlib.Path,
        default=default_root / "annotation_work" / "roi_from_95",
    )
    parser.add_argument(
        "--out-root",
        type=pathlib.Path,
        default=home / "vrifa_eval" / "full_sweep_roi95",
    )
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8) // 4))
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--only-video", choices=tuple(VIDEOS), default=None)
    parser.add_argument("--render-best", type=int, default=0)
    parser.add_argument("--keep-dumps", action="store_true")
    return parser.parse_args()


def cli_pair(flag: str, value: Any) -> list[str]:
    return [f"{flag}={value}"]


def flag(name: str, enabled: bool) -> list[str]:
    return [name] if enabled else []


def base_params() -> dict[str, Any]:
    return {
        "ref_mode": ("first",),
        "ref_running_alpha": 0.05,
        "dynamic_calibration_frames": 10,
        "dynamic_target_fraction": 0.2,
        "dynamic_lag_scale": 1.0,
        "dynamic_lag_linear": False,
        "dynamic_lag_linear_max": 60,
        "dynamic_lag_linear_start": 0,
        "colorspace": "CIELAB",
        "channel_weights": None,
        "pre_delta_blur": "none",
        "blur": "gaussian:9",
        "morph_kernel": 13,
        "morph_shape": "ellipse",
        "morph_close_iterations": 1,
        "morph_open_iterations": 1,
        "min_area": 400,
        "threshold": "otsu",
        "threshold_offset": 0,
        "darken_only": True,
        "peak_reference": True,
        "camera_stable": False,
        "motion_per_frame_threshold": 1.5,
        "cumulative_motion_threshold": 3.0,
        "motion_model": "affine",
        "peak_on_shift": "reset",
        "lock_frames": 3,
    }


def build_args(params: dict[str, Any]) -> tuple[str, ...]:
    args: list[str] = []
    args += ["--ref-mode", str(params["ref_mode"][0])]
    if len(params["ref_mode"]) > 1:
        args.append(str(params["ref_mode"][1]))
    args += cli_pair("--ref-running-alpha", params["ref_running_alpha"])
    args += cli_pair(
        "--dynamic-calibration-frames", params["dynamic_calibration_frames"]
    )
    args += cli_pair("--dynamic-target-fraction", params["dynamic_target_fraction"])
    args += cli_pair("--dynamic-lag-scale", params["dynamic_lag_scale"])
    args += flag("--dynamic-lag-linear", params["dynamic_lag_linear"])
    args += cli_pair("--dynamic-lag-linear-max", params["dynamic_lag_linear_max"])
    args += cli_pair("--dynamic-lag-linear-start", params["dynamic_lag_linear_start"])
    args += cli_pair("--colorspace", params["colorspace"])
    if params["channel_weights"]:
        args += cli_pair("--channel-weights", params["channel_weights"])
    args += cli_pair("--pre-delta-blur", params["pre_delta_blur"])
    args += cli_pair("--blur", params["blur"])
    args += cli_pair("--morph-kernel", params["morph_kernel"])
    args += cli_pair("--morph-shape", params["morph_shape"])
    args += cli_pair("--morph-close-iterations", params["morph_close_iterations"])
    args += cli_pair("--morph-open-iterations", params["morph_open_iterations"])
    args += cli_pair("--min-area", params["min_area"])
    args += cli_pair("--threshold", params["threshold"])
    args += cli_pair("--threshold-offset", params["threshold_offset"])
    args += flag("--no-darken-only", not params["darken_only"])
    args += flag("--no-peak-reference", not params["peak_reference"])
    args += flag("--camera-stable", params["camera_stable"])
    args += cli_pair(
        "--motion-per-frame-threshold", params["motion_per_frame_threshold"]
    )
    args += cli_pair(
        "--cumulative-motion-threshold", params["cumulative_motion_threshold"]
    )
    args += cli_pair("--motion-model", params["motion_model"])
    args += cli_pair("--peak-on-shift", params["peak_on_shift"])
    args += cli_pair("--lock-frames", params["lock_frames"])
    args += cli_pair("--annotation-formats", "")
    return tuple(args)


def case_from_params(params: dict[str, Any]) -> Case:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return Case(case_id=digest, params=params, args=build_args(params))


def merged(**updates: Any) -> dict[str, Any]:
    params = base_params()
    params.update(updates)
    return params


def generate_cases() -> list[Case]:
    raw: list[dict[str, Any]] = []

    thresholds = []
    for mode in ("otsu", "triangle"):
        for offset in (-60, -45, -30, -15, 0, 15, 30, 45):
            thresholds.append({"threshold": mode, "threshold_offset": offset})
    for value in (35, 50, 65, 80, 95, 110, 125, 140, 160):
        thresholds.append({"threshold": f"manual:{value}", "threshold_offset": 0})
    for value in (60, 70, 80, 85, 90, 95, 98):
        thresholds.append({"threshold": f"percentile:{value}", "threshold_offset": 0})
    for family, block, c_value in itertools.product(
        ("adaptive-mean", "adaptive-gaussian"), (15, 21, 31, 41), (-5, 0, 5, 10, 15)
    ):
        thresholds.append(
            {"threshold": f"{family}:{block}:{c_value}", "threshold_offset": 0}
        )

    references = [
        {"ref_mode": ("first",)},
        {"ref_mode": ("absolute", 0)},
        {"ref_mode": ("absolute", 5)},
        {"ref_mode": ("absolute", 25)},
        {"ref_mode": ("prev", 1)},
        {"ref_mode": ("prev", 3)},
        {"ref_mode": ("prev", 5)},
        {"ref_mode": ("prev", 10)},
        {"ref_mode": ("prev", 20)},
    ]
    references += [
        {"ref_mode": ("running",), "ref_running_alpha": alpha}
        for alpha in (0.01, 0.03, 0.05, 0.10, 0.20)
    ]
    for calibration, target, scale, linear in itertools.product(
        (5, 10, 20), (0.1, 0.2, 0.35), (0.5, 1.0, 2.0), (False, True)
    ):
        references.append(
            {
                "ref_mode": ("dynamic",),
                "dynamic_calibration_frames": calibration,
                "dynamic_target_fraction": target,
                "dynamic_lag_scale": scale,
                "dynamic_lag_linear": linear,
                "dynamic_lag_linear_max": 60,
                "dynamic_lag_linear_start": 0,
            }
        )

    colors = [
        {"colorspace": "CIELAB", "channel_weights": None},
        {"colorspace": "CIELAB", "channel_weights": "1,0,0"},
        {"colorspace": "CIELAB", "channel_weights": "1,0.5,0.5"},
        {"colorspace": "CIELAB", "channel_weights": "2,1,1"},
        {"colorspace": "RGB", "channel_weights": None},
        {"colorspace": "RGB", "channel_weights": "0.5,1,1"},
        {"colorspace": "HSV", "channel_weights": None},
        {"colorspace": "HSV", "channel_weights": "0.25,1,1"},
        {"colorspace": "GRAYSCALE", "channel_weights": None},
    ]

    blurs = []
    for pre_delta_blur in ("none", "gaussian:3", "gaussian:5", "median:3"):
        for blur in (
            "none",
            "gaussian:5",
            "gaussian:9",
            "gaussian:13",
            "flat:7",
            "triangle:9",
            "median:3",
            "bilateral:5",
        ):
            blurs.append({"pre_delta_blur": pre_delta_blur, "blur": blur})

    morphs = []
    for kernel, close_iter, open_iter, shape in itertools.product(
        (1, 5, 9, 13, 17, 21, 31), (0, 1, 2), (0, 1, 2), ("ellipse", "rect", "cross")
    ):
        if kernel == 1 and (close_iter or open_iter):
            continue
        morphs.append(
            {
                "morph_kernel": kernel,
                "morph_close_iterations": close_iter,
                "morph_open_iterations": open_iter,
                "morph_shape": shape,
            }
        )

    min_areas = [
        {"min_area": value} for value in (0, 50, 100, 200, 400, 800, 1500, 3000)
    ]
    locks = [{"lock_frames": value} for value in (0, 1, 3, 5, 8)]
    polarity_peak = [
        {"darken_only": darken, "peak_reference": peak}
        for darken, peak in itertools.product((True, False), (True, False))
    ]
    camera = [{"camera_stable": False}]
    for model, per_frame, cumulative, peak_shift in itertools.product(
        ("translation", "euclidean", "affine"),
        (0.75, 1.5, 3.0),
        (1.5, 3.0, 6.0),
        ("reset", "warp"),
    ):
        camera.append(
            {
                "camera_stable": True,
                "motion_model": model,
                "motion_per_frame_threshold": per_frame,
                "cumulative_motion_threshold": cumulative,
                "peak_on_shift": peak_shift,
            }
        )

    groups = [
        thresholds,
        references,
        colors,
        blurs,
        morphs,
        min_areas,
        locks,
        polarity_peak,
        camera,
    ]
    for group in groups:
        raw.extend(merged(**variant) for variant in group)

    focused_refs = [
        {"ref_mode": ("first",)},
        {"ref_mode": ("prev", 3)},
        {"ref_mode": ("prev", 5)},
        {"ref_mode": ("running",), "ref_running_alpha": 0.05},
        {
            "ref_mode": ("dynamic",),
            "dynamic_calibration_frames": 10,
            "dynamic_target_fraction": 0.2,
            "dynamic_lag_scale": 1.0,
        },
    ]
    focused_thresholds = (
        [
            {"threshold": "otsu", "threshold_offset": offset}
            for offset in (-30, -15, 0, 15)
        ]
        + [
            {"threshold": "triangle", "threshold_offset": offset}
            for offset in (-15, 0, 15)
        ]
        + [
            {"threshold": f"percentile:{value}", "threshold_offset": 0}
            for value in (80, 85, 90, 95)
        ]
    )
    focused_morphs = [
        {
            "morph_kernel": kernel,
            "morph_close_iterations": close_iter,
            "morph_open_iterations": open_iter,
            "morph_shape": shape,
        }
        for kernel, close_iter, open_iter, shape in itertools.product(
            (5, 9, 13, 17, 21), (0, 1), (0, 1), ("ellipse", "rect")
        )
    ]
    focused_blurs = [
        {"pre_delta_blur": pre, "blur": post}
        for pre, post in itertools.product(
            ("none", "gaussian:3"),
            ("gaussian:5", "gaussian:9", "triangle:9", "bilateral:5"),
        )
    ]
    focused_colors = colors[:5]
    focused_locks = [{"lock_frames": value} for value in (0, 1, 3, 5)]
    focused_peaks = polarity_peak
    focused_groups = [
        focused_refs,
        focused_thresholds,
        focused_morphs,
        focused_blurs,
        focused_colors,
        focused_locks,
        focused_peaks,
        min_areas,
        camera[:8],
    ]

    # Pairwise coverage keeps every setting involved in combinations without
    # turning the sweep into an intractable full Cartesian product.
    for left_index, right_index in itertools.combinations(
        range(len(focused_groups)), 2
    ):
        for left, right in itertools.product(
            focused_groups[left_index], focused_groups[right_index]
        ):
            params = merged(**left)
            params.update(right)
            raw.append(params)

    # Higher-order passes target the interactions that matter most for this
    # algorithm's behavior.
    for ref, thresh, morph, peak in itertools.product(
        focused_refs,
        focused_thresholds,
        focused_morphs,
        focused_peaks,
    ):
        params = merged(**ref)
        for update in (thresh, morph, peak):
            params.update(update)
        raw.append(params)
    for ref, thresh, blur, color in itertools.product(
        focused_refs,
        focused_thresholds,
        focused_blurs,
        focused_colors,
    ):
        params = merged(**ref)
        for update in (thresh, blur, color):
            params.update(update)
        raw.append(params)

    by_id: dict[str, Case] = {}
    for params in raw:
        case = case_from_params(params)
        by_id[case.case_id] = case
    return sorted(by_id.values(), key=lambda case: case.case_id)


def load_gt_masks(annot_root: pathlib.Path) -> dict[tuple[str, int], np.ndarray]:
    masks: dict[tuple[str, int], np.ndarray] = {}
    for video, frames in VIDEOS.items():
        for frame in frames:
            stem = f"{video}__frame_{frame:06d}"
            image_path = annot_root / "images" / f"{stem}.png"
            state_path = annot_root / "labels" / f"{stem}.state.json"
            with Image.open(image_path) as image:
                width, height = image.size
            mask_image = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask_image)
            data = json.loads(state_path.read_text())
            for item in data:
                vertices = item.get("vertices", "")
                points = []
                for pair in vertices.split(";"):
                    if not pair:
                        continue
                    x_text, y_text = pair.split(",")
                    points.append((float(x_text), float(y_text)))
                if len(points) >= 3:
                    draw.polygon(points, fill=1)
            masks[(video, frame)] = np.array(mask_image, dtype=np.uint8) > 0
    return masks


def score_case(
    case_dir: pathlib.Path,
    videos: dict[str, list[int]],
    gt_masks: dict[tuple[str, int], np.ndarray],
) -> dict[str, Any]:
    rows = []
    all_ious = []
    for video, frames in videos.items():
        frame_ious = []
        for frame in frames:
            mask_path = case_dir / video / f"frame_{frame:06d}" / "mask.npy"
            pred = np.load(mask_path) > 0
            gt = gt_masks[(video, frame)]
            inter = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            iou = float(inter / union) if union else 1.0
            frame_ious.append(iou)
            all_ious.append(iou)
        rows.append(
            {
                "video": video,
                "mean_iou": float(statistics.mean(frame_ious)),
                "median_iou": float(statistics.median(frame_ious)),
                "frame_ious": frame_ious,
            }
        )
    return {
        "overall_mean": float(statistics.mean(all_ious)),
        "overall_median": float(statistics.median(all_ious)),
        "worst_frame_iou": float(min(all_ious)),
        "rows": rows,
    }


def cleanup_debug_dumps(case_dir: pathlib.Path, videos: dict[str, list[int]]) -> None:
    for video, frames in videos.items():
        for frame in frames:
            shutil.rmtree(
                case_dir / video / f"frame_{frame:06d}",
                ignore_errors=True,
            )


def run_video(
    case: Case,
    video: str,
    frames: list[int],
    args: argparse.Namespace,
    case_dir: pathlib.Path,
) -> None:
    video_out = case_dir / video
    video_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.bin),
        "--video-path",
        str(args.video_root / f"{video}.mp4"),
        "--output-dir",
        str(video_out),
        "--roi-mask",
        str(args.roi_root / f"{video}.png"),
        "--debug-dump-frames",
        ",".join(str(frame) for frame in frames),
        "--debug-dump-dir",
        str(video_out),
        *case.args,
    ]
    env = os.environ.copy()
    env.setdefault("OPENCV_FOR_THREADS_NUM", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    with (video_out / "run.log").open("w") as log:
        proc = subprocess.run(
            cmd, stdout=log, stderr=subprocess.STDOUT, check=False, env=env
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{case.case_id} failed on {video}; see {video_out / 'run.log'}"
        )


def run_case(
    payload: tuple[Case, str, dict[str, list[int]], str, str, str, str, str, bool],
) -> dict[str, Any]:
    (
        case,
        out_root_text,
        videos,
        bin_text,
        video_root_text,
        roi_root_text,
        annot_root_text,
        only_video,
        keep_dumps,
    ) = payload
    args = argparse.Namespace(
        bin=pathlib.Path(bin_text),
        video_root=pathlib.Path(video_root_text),
        roi_root=pathlib.Path(roi_root_text),
    )
    out_root = pathlib.Path(out_root_text)
    case_dir = out_root / "cases" / case.case_id
    done_path = case_dir / "done.ok"
    score_path = case_dir / "score.json"
    if done_path.exists() and score_path.exists():
        return json.loads(score_path.read_text())

    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "params.json").write_text(
        json.dumps(case.params, indent=2, sort_keys=True) + "\n"
    )
    selected_videos = {only_video: videos[only_video]} if only_video else videos
    try:
        for video, frames in selected_videos.items():
            run_video(case, video, frames, args, case_dir)
        gt_masks = load_gt_masks(pathlib.Path(annot_root_text))
        score = score_case(case_dir, selected_videos, gt_masks)
        result = {
            "case_id": case.case_id,
            "params": case.params,
            "args": list(case.args),
            **score,
        }
        score_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if not keep_dumps:
            cleanup_debug_dumps(case_dir, selected_videos)
        done_path.write_text("ok\n")
        return result
    except Exception as exc:
        (case_dir / "FAILED.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


def write_leaderboard(out_root: pathlib.Path, results: list[dict[str, Any]]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    results = sorted(
        results,
        key=lambda row: (
            row["overall_mean"],
            row["overall_median"],
            row["worst_frame_iou"],
        ),
        reverse=True,
    )
    (out_root / "leaderboard.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    with (out_root / "leaderboard.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "rank",
                "case_id",
                "overall_mean",
                "overall_median",
                "worst_frame_iou",
                "reference",
                "threshold",
                "threshold_offset",
                "colorspace",
                "channel_weights",
                "pre_delta_blur",
                "blur",
                "morph_kernel",
                "morph_shape",
                "morph_close_iterations",
                "morph_open_iterations",
                "min_area",
                "lock_frames",
                "darken_only",
                "peak_reference",
                "camera_stable",
                "motion_model",
            ]
        )
        for rank, result in enumerate(results, start=1):
            params = result["params"]
            writer.writerow(
                [
                    rank,
                    result["case_id"],
                    f"{result['overall_mean']:.6f}",
                    f"{result['overall_median']:.6f}",
                    f"{result['worst_frame_iou']:.6f}",
                    " ".join(str(part) for part in params["ref_mode"]),
                    params["threshold"],
                    params["threshold_offset"],
                    params["colorspace"],
                    params["channel_weights"] or "",
                    params["pre_delta_blur"],
                    params["blur"],
                    params["morph_kernel"],
                    params["morph_shape"],
                    params["morph_close_iterations"],
                    params["morph_open_iterations"],
                    params["min_area"],
                    params["lock_frames"],
                    params["darken_only"],
                    params["peak_reference"],
                    params["camera_stable"],
                    params["motion_model"],
                ]
            )


def render_best(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    for result in results[: args.render_best]:
        case = case_from_params(result["params"])
        render_root = args.out_root / "best_renders" / case.case_id
        for video in VIDEOS:
            out_dir = render_root / video
            if (out_dir / "videos" / "overlay.mp4").exists():
                continue
            cmd = [
                str(args.bin),
                "--video-path",
                str(args.video_root / f"{video}.mp4"),
                "--output-dir",
                str(out_dir),
                "--roi-mask",
                str(args.roi_root / f"{video}.png"),
                "--write-mask-video",
                "true",
                "--write-overlay-video",
                "true",
                *case.args,
            ]
            out_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.setdefault("OPENCV_FOR_THREADS_NUM", "1")
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("NUMEXPR_NUM_THREADS", "1")
            with (out_dir / "run.log").open("w") as log:
                subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT, check=True, env=env
                )


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    cases = generate_cases()
    if args.start:
        cases = cases[args.start :]
    if args.max_cases:
        cases = cases[: args.max_cases]

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_count": len(cases),
        "jobs": args.jobs,
        "videos": VIDEOS
        if not args.only_video
        else {args.only_video: VIDEOS[args.only_video]},
    }
    (args.out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    payloads = [
        (
            case,
            str(args.out_root),
            VIDEOS,
            str(args.bin),
            str(args.video_root),
            str(args.roi_root),
            str(args.annot_root),
            args.only_video or "",
            args.keep_dumps,
        )
        for case in cases
    ]

    results: list[dict[str, Any]] = []
    failures = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        pending = {
            executor.submit(run_case, payload): payload[0].case_id
            for payload in payloads
        }
        completed = 0
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                case_id = pending.pop(future)
                completed += 1
                try:
                    results.append(future.result())
                except Exception as exc:
                    failures += 1
                    print(f"FAILED {case_id}: {exc}", flush=True)
                if completed == 1 or completed % 10 == 0:
                    write_leaderboard(args.out_root, results)
                    best = max((row["overall_mean"] for row in results), default=0.0)
                    print(
                        f"progress {completed}/{len(payloads)} failures={failures} best_mean={best:.4f}",
                        flush=True,
                    )

    write_leaderboard(args.out_root, results)
    if args.render_best:
        render_best(
            args, sorted(results, key=lambda row: row["overall_mean"], reverse=True)
        )
    print(f"complete results={len(results)} failures={failures}", flush=True)
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
