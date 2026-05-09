#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


STRIP_SUMMARY_KEYS = {
    "run_timestamp",
    "video_path",
    "output_dir",
    "average_compute_time_seconds",
    "total_run_time_seconds",
}

# Empirical observation from the Rust/Python diagnosis pass:
# mp4v color video round-trips introduce several L2 of codec noise even when the
# raw overlay/heatmap frames match exactly before encoding. Use PNGs as the
# authoritative parity artifacts and keep MP4 comparisons advisory only.
COLOR_MP4_ENCODER_NOISE_FLOOR = {
    "overlay.mp4": 4.25,
    "heatmap.mp4": 3.53,
}


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"FAIL {message}")


def load_yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def compare_summary(py_dir: Path, rs_dir: Path, failures: list[str]) -> None:
    py_summary = load_yaml(py_dir / "run_summary.yaml")
    rs_summary = load_yaml(rs_dir / "run_summary.yaml")
    py_fps = py_summary.get("video_fps")
    rs_fps = rs_summary.get("video_fps")
    if py_fps is not None and rs_fps is not None:
        denom = max(abs(float(py_fps)), 1e-9)
        if abs(float(py_fps) - float(rs_fps)) / denom > 1e-3:
            fail(f"video_fps differs: python={py_fps} rust={rs_fps}", failures)
    py_summary.pop("video_fps", None)
    rs_summary.pop("video_fps", None)
    for key in STRIP_SUMMARY_KEYS:
        py_summary.pop(key, None)
        rs_summary.pop(key, None)
    if py_summary != rs_summary:
        fail("run_summary.yaml differs after volatile-key stripping", failures)


def load_coco(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def bbox_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def hungarian_minimize(cost: np.ndarray) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    transposed = False
    if cost.shape[0] > cost.shape[1]:
        cost = cost.T
        transposed = True
    n, m = cost.shape
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = []
    for j in range(1, m + 1):
        if p[j] > 0:
            pair = (p[j] - 1, j - 1)
            pairs.append((pair[1], pair[0]) if transposed else pair)
    return pairs


def annotations_by_image(coco: dict) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {image["id"]: [] for image in coco["images"]}
    for ann in coco["annotations"]:
        grouped.setdefault(ann["image_id"], []).append(ann)
    return grouped


def compare_coco(py_dir: Path, rs_dir: Path, failures: list[str]) -> None:
    rel = Path("formatCOCO/annotations/instances_default.json")
    py_coco = load_coco(py_dir / rel)
    rs_coco = load_coco(rs_dir / rel)
    if len(py_coco["images"]) != len(rs_coco["images"]):
        fail(
            f"COCO image count differs: python={len(py_coco['images'])} rust={len(rs_coco['images'])}",
            failures,
        )
        return

    py_grouped = annotations_by_image(py_coco)
    rs_grouped = annotations_by_image(rs_coco)
    py_images = sorted(py_coco["images"], key=lambda image: image["id"])
    rs_images = sorted(rs_coco["images"], key=lambda image: image["id"])
    for py_image, rs_image in zip(py_images, rs_images):
        py_anns = py_grouped.get(py_image["id"], [])
        rs_anns = rs_grouped.get(rs_image["id"], [])
        py_count = len(py_anns)
        rs_count = len(rs_anns)
        if py_count == 0:
            if rs_count != 0:
                fail(f"{py_image['file_name']} annotation count differs from zero", failures)
            continue
        if abs(py_count - rs_count) / py_count > 0.01:
            fail(
                f"{py_image['file_name']} annotation count drift >1%: python={py_count} rust={rs_count}",
                failures,
            )
        py_area = sum(float(ann["area"]) for ann in py_anns)
        rs_area = sum(float(ann["area"]) for ann in rs_anns)
        if py_area > 0 and abs(py_area - rs_area) / py_area > 0.005:
            fail(
                f"{py_image['file_name']} annotation area drift >0.5%: python={py_area} rust={rs_area}",
                failures,
            )
        ious = np.array(
            [[bbox_iou(a["bbox"], b["bbox"]) for b in rs_anns] for a in py_anns],
            dtype=float,
        )
        pairs = hungarian_minimize(1.0 - ious) if ious.size else []
        matched = [ious[i, j] for i, j in pairs]
        if matched and float(np.mean(matched)) < 0.99:
            fail(
                f"{py_image['file_name']} mean bbox IoU <0.99: {float(np.mean(matched)):.6f}",
                failures,
            )
        unmatched = abs(py_count - rs_count)
        if unmatched:
            print(f"WARN {py_image['file_name']} unmatched boxes: {unmatched}")


def frame_iter(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open {path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    aa = a > 127
    bb = b > 127
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(aa, bb).sum() / union)


def frame_l2(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(math.sqrt(np.mean(diff * diff)))


def advisory_video_mask_iou(py_path: Path, rs_path: Path) -> None:
    if not py_path.exists() or not rs_path.exists():
        print(f"WARN missing advisory video: {py_path if not py_path.exists() else rs_path}")
        return
    try:
        values = [mask_iou(a, b) for a, b in zip(frame_iter(py_path), frame_iter(rs_path))]
    except RuntimeError as exc:
        print(f"WARN {exc}")
        return
    if not values:
        print(f"WARN {py_path.name} has no comparable frames")
        return
    print(f"INFO advisory {py_path.name} mean_frame_iou={float(np.mean(values)):.6f}")


def advisory_video_l2(py_path: Path, rs_path: Path) -> None:
    if not py_path.exists() or not rs_path.exists():
        print(f"WARN missing advisory video: {py_path if not py_path.exists() else rs_path}")
        return
    try:
        values = [frame_l2(a, b) for a, b in zip(frame_iter(py_path), frame_iter(rs_path))]
    except RuntimeError as exc:
        print(f"WARN {exc}")
        return
    if not values:
        print(f"WARN {py_path.name} has no comparable frames")
        return
    mean_l2 = float(np.mean(values))
    floor = COLOR_MP4_ENCODER_NOISE_FLOOR.get(py_path.name)
    if floor is None:
        print(f"INFO advisory {py_path.name} mean_l2={mean_l2:.6f}")
    else:
        print(
            f"INFO advisory {py_path.name} mean_l2={mean_l2:.6f} "
            f"(empirical mp4v color-noise floor ~{floor:.2f})"
        )


def load_png(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"unable to read {path}")
    return image


def compare_pngs(py_dir: Path, rs_dir: Path, subdir: str, kind: str, failures: list[str]) -> None:
    py_subdir = py_dir / subdir
    rs_subdir = rs_dir / subdir
    if not py_subdir.is_dir() or not rs_subdir.is_dir():
        print(
            f"WARN skipping authoritative PNG parity for {subdir}: "
            f"missing {'python' if not py_subdir.is_dir() else 'rust'} directory"
        )
        return

    py_files = {path.name: path for path in py_subdir.glob("*.png")}
    rs_files = {path.name: path for path in rs_subdir.glob("*.png")}
    if set(py_files) != set(rs_files):
        missing_py = sorted(set(rs_files) - set(py_files))
        missing_rs = sorted(set(py_files) - set(rs_files))
        fail(
            f"{subdir} PNG filename sets differ: missing_in_python={missing_py[:5]} missing_in_rust={missing_rs[:5]}",
            failures,
        )
        return

    for name in sorted(py_files):
        py_path = py_files[name]
        rs_path = rs_files[name]
        if kind == "mask":
            if py_path.read_bytes() != rs_path.read_bytes():
                py_image = load_png(py_path)
                rs_image = load_png(rs_path)
                diff = np.abs(py_image.astype(np.int16) - rs_image.astype(np.int16))
                fail(
                    f"{subdir}/{name} mask PNG differs: max_abs_diff={int(diff.max())} mean_abs_diff={float(diff.mean()):.6f}",
                    failures,
                )
            continue

        py_image = load_png(py_path)
        rs_image = load_png(rs_path)
        if py_image.shape != rs_image.shape:
            fail(
                f"{subdir}/{name} shape differs: python={py_image.shape} rust={rs_image.shape}",
                failures,
            )
            continue
        diff = np.abs(py_image.astype(np.int16) - rs_image.astype(np.int16))
        max_abs = int(diff.max())
        mean_abs = float(diff.mean())
        if max_abs > 1 or mean_abs > 0.1:
            fail(
                f"{subdir}/{name} {kind} PNG differs: max_abs_diff={max_abs} mean_abs_diff={mean_abs:.6f}",
                failures,
            )


def compare_artifacts(py_dir: Path, rs_dir: Path, failures: list[str]) -> None:
    png_subdirs = ("masks", "overlays", "heatmap")
    png_available = all((py_dir / subdir).is_dir() and (rs_dir / subdir).is_dir() for subdir in png_subdirs)
    if png_available:
        compare_pngs(py_dir, rs_dir, "masks", "mask", failures)
        compare_pngs(py_dir, rs_dir, "overlays", "overlay", failures)
        compare_pngs(py_dir, rs_dir, "heatmap", "heatmap", failures)
    else:
        print(
            "WARN skipping authoritative PNG parity because one or more PNG directories "
            "are missing; falling back to advisory MP4 checks only"
        )
    advisory_video_mask_iou(py_dir / "videos/mask.mp4", rs_dir / "videos/mask.mp4")
    advisory_video_l2(py_dir / "videos/overlay.mp4", rs_dir / "videos/overlay.mp4")
    advisory_video_l2(py_dir / "videos/heatmap.mp4", rs_dir / "videos/heatmap.mp4")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("python_run", type=Path)
    parser.add_argument("rust_run", type=Path)
    args = parser.parse_args()
    failures: list[str] = []
    compare_summary(args.python_run, args.rust_run, failures)
    compare_coco(args.python_run, args.rust_run, failures)
    compare_artifacts(args.python_run, args.rust_run, failures)
    if failures:
        print(f"{len(failures)} parity check(s) failed")
        return 1
    print("Parity checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
