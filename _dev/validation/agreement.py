#!/usr/bin/env python3
"""Evaluate mask agreement against ``data/labels.json``.

The expected run layout is:

    <runs-dir>/<sample>/masks/frame_NNNNNN.png

where ``sample`` is ``input_1`` through ``input_11`` and the PNGs are
single-channel masks at source resolution.

Metrics:

- mask IoU
- Sorensen-Dice
- boundary F1 at tau = 1, 3, and 5 pixels
- mean boundary distance in pixels
- bounding-box IoU

Bootstrap confidence intervals are computed from frame-level means with
replacement. When exactly one boundary is empty, the mean boundary
distance is set to the image diagonal so the metric remains finite and
JSON-serializable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

FRAME_NAME_RE = re.compile(r"^(?P<slug>input_\d+)__frame_(?P<idx>\d+)\.png$")
BOUNDARY_TOLERANCES_PX = (1, 3, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--labels", default=Path("data/labels.json"), type=Path)
    parser.add_argument("--bootstrap-samples", default=10_000, type=int)
    parser.add_argument("--bootstrap-seed", default=0, type=int)
    return parser.parse_args()


def parse_frame_name(name: str) -> tuple[str, int]:
    match = FRAME_NAME_RE.match(name)
    if not match:
        raise ValueError(f"unexpected label image filename: {name!r}")
    return match.group("slug"), int(match.group("idx"))


def sample_sort_key(sample: str) -> tuple[int, str]:
    try:
        return int(sample.split("_", 1)[1]), sample
    except (IndexError, ValueError):
        return (10**9, sample)


def rasterize_polygon(segmentation: list[list[float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        if not polygon:
            continue
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def load_ground_truth(labels: dict[str, Any]) -> dict[str, dict[int, np.ndarray]]:
    images_by_id: dict[int, dict[str, Any]] = {}
    gt_by_sample: dict[str, dict[int, np.ndarray]] = {}

    for image in labels["images"]:
        slug, frame_index = parse_frame_name(image["file_name"])
        images_by_id[image["id"]] = {
            **image,
            "sample": slug,
            "frame_index": frame_index,
        }

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in labels["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    for image_id, image in images_by_id.items():
        h = int(image["height"])
        w = int(image["width"])
        merged = np.zeros((h, w), dtype=np.uint8)
        for ann in annotations_by_image.get(image_id, []):
            segmentation = ann.get("segmentation", [])
            if not isinstance(segmentation, list) or not segmentation:
                continue
            merged |= rasterize_polygon(segmentation, h, w)
        gt_by_sample.setdefault(image["sample"], {})[image["frame_index"]] = merged

    return gt_by_sample


def load_predicted_mask(masks_dir: Path, frame_index: int) -> np.ndarray | None:
    path = masks_dir / f"frame_{frame_index:06d}.png"
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return (img > 127).astype(np.uint8)


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    return (cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0).astype(np.uint8)


def boundary_distances(source_edge: np.ndarray, target_edge: np.ndarray) -> np.ndarray:
    if source_edge.sum() == 0 or target_edge.sum() == 0:
        return np.array([], dtype=np.float32)
    inv_target = (1 - target_edge.astype(np.uint8)) * 255
    dist = cv2.distanceTransform(inv_target, cv2.DIST_L2, 5)
    return dist[source_edge > 0].astype(np.float32)


def compute_box_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        ys, xs = np.where(mask)
        if ys.size == 0:
            return None
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())

    box_a = bbox(gt)
    box_b = bbox(pred)
    if box_a is None and box_b is None:
        return 1.0
    if box_a is None or box_b is None:
        return 0.0

    iy0 = max(box_a[0], box_b[0])
    ix0 = max(box_a[1], box_b[1])
    iy1 = min(box_a[2], box_b[2])
    ix1 = min(box_a[3], box_b[3])
    inter_h = max(0, iy1 - iy0 + 1)
    inter_w = max(0, ix1 - ix0 + 1)
    inter = inter_h * inter_w
    area_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def per_frame_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    gt_bool = gt > 0
    pred_bool = pred > 0

    inter = int(np.logical_and(gt_bool, pred_bool).sum())
    union = int(np.logical_or(gt_bool, pred_bool).sum())
    iou = inter / union if union > 0 else 1.0

    gt_sum = int(gt_bool.sum())
    pred_sum = int(pred_bool.sum())
    dice = 2 * inter / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 1.0

    gt_edge = extract_boundary(gt_bool.astype(np.uint8))
    pred_edge = extract_boundary(pred_bool.astype(np.uint8))
    pred_to_gt = boundary_distances(pred_edge, gt_edge)
    gt_to_pred = boundary_distances(gt_edge, pred_edge)

    boundary_scores: dict[str, float] = {}
    for tol in BOUNDARY_TOLERANCES_PX:
        if pred_edge.sum() == 0 and gt_edge.sum() == 0:
            f1 = 1.0
        elif pred_edge.sum() == 0 or gt_edge.sum() == 0:
            f1 = 0.0
        else:
            precision = float((pred_to_gt <= tol).mean())
            recall = float((gt_to_pred <= tol).mean())
            f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        boundary_scores[f"boundary_f1_at_{tol}px"] = f1

    if pred_edge.sum() == 0 and gt_edge.sum() == 0:
        mean_boundary_distance = 0.0
    elif pred_edge.sum() == 0 or gt_edge.sum() == 0:
        mean_boundary_distance = float(math.hypot(gt.shape[0], gt.shape[1]))
    else:
        mean_boundary_distance = float(np.concatenate((pred_to_gt, gt_to_pred)).mean())

    boundary_f1_mean = float(np.mean([boundary_scores[f"boundary_f1_at_{tol}px"] for tol in BOUNDARY_TOLERANCES_PX]))

    return {
        "iou": iou,
        "dice": dice,
        **boundary_scores,
        "boundary_f1_mean": boundary_f1_mean,
        "mean_boundary_distance_px": mean_boundary_distance,
        "box_iou": compute_box_iou(gt_bool, pred_bool),
        "gt_pixels": float(gt_sum),
        "pred_pixels": float(pred_sum),
    }


def bootstrap_mean_summary(values: list[float], bootstrap_samples: int, rng: np.random.Generator) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot summarize an empty metric list")
    if arr.size == 1:
        mean = float(arr[0])
        return {
            "mean": mean,
            "std": 0.0,
            "ci95": [mean, mean],
            "n": 1,
        }

    draws = rng.choice(arr, size=(bootstrap_samples, arr.size), replace=True)
    means = draws.mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "n": int(arr.size),
    }


def summarize_frames(frames: list[dict[str, Any]], bootstrap_samples: int, rng: np.random.Generator) -> dict[str, Any]:
    metric_keys = (
        "iou",
        "dice",
        "boundary_f1_at_1px",
        "boundary_f1_at_3px",
        "boundary_f1_at_5px",
        "boundary_f1_mean",
        "mean_boundary_distance_px",
        "box_iou",
    )
    return {
        key: bootstrap_mean_summary([float(frame[key]) for frame in frames], bootstrap_samples, rng)
        for key in metric_keys
    }


def evaluate_runs(
    runs_dir: Path,
    labels_path: Path,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    labels = json.loads(labels_path.read_text())
    gt_by_sample = load_ground_truth(labels)

    sample_results: dict[str, Any] = {}
    all_valid_frames: list[dict[str, Any]] = []
    total_expected = 0
    total_missing = 0

    for sample in sorted(gt_by_sample, key=sample_sort_key):
        gts = gt_by_sample[sample]
        masks_dir = runs_dir / sample / "masks"
        frame_results: list[dict[str, Any]] = []

        for frame_index, gt in sorted(gts.items()):
            total_expected += 1
            pred = load_predicted_mask(masks_dir, frame_index)
            if pred is None:
                frame_results.append({
                    "sample": sample,
                    "frame_index": frame_index,
                    "missing_pred": True,
                })
                total_missing += 1
                continue
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
                pred = (pred > 0).astype(np.uint8)
            metrics = per_frame_metrics(gt, pred)
            frame_record = {
                "sample": sample,
                "frame_index": frame_index,
                **metrics,
            }
            frame_results.append(frame_record)
            all_valid_frames.append(frame_record)

        valid_frames = [frame for frame in frame_results if not frame.get("missing_pred", False)]
        sample_rng = np.random.default_rng(bootstrap_seed + sample_sort_key(sample)[0])
        sample_result: dict[str, Any] = {
            "n_expected": len(gts),
            "n_matched": len(valid_frames),
            "n_missing": len(gts) - len(valid_frames),
            "frames": frame_results,
        }
        if valid_frames:
            sample_result["metrics"] = summarize_frames(valid_frames, bootstrap_samples, sample_rng)
        sample_results[sample] = sample_result

    overall: dict[str, Any] = {
        "n_samples": len(gt_by_sample),
        "n_expected": total_expected,
        "n_matched": len(all_valid_frames),
        "n_missing": total_missing,
    }
    if all_valid_frames:
        overall_rng = np.random.default_rng(bootstrap_seed)
        overall["metrics"] = summarize_frames(all_valid_frames, bootstrap_samples, overall_rng)

    return {
        "runs_dir": str(runs_dir),
        "labels": str(labels_path),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "overall": overall,
        "samples": sample_results,
    }


def main() -> None:
    args = parse_args()
    results = evaluate_runs(
        runs_dir=args.runs_dir,
        labels_path=args.labels,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
