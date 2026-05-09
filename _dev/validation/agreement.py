#!/usr/bin/env python3
"""Compute agreement metrics between labeled frames and Rust output masks.

Inputs:
  --labels        paper/data/labels_55.json (COCO export from makesense.ai)
  --runs-dir      directory containing per-sample Rust runs, where each
                  per-sample subdirectory has a masks/ folder of PNGs.
                  e.g. outputs_paper/<sample_slug>/masks/frame_<idx>.png
  --output        paper/data/agreement_metrics.json (default)

For each labeled image (filename `<slug>__frame_<idx>.png`):
  - rasterize the COCO polygon(s) into a binary ground-truth mask
  - load the matching predicted mask from <runs-dir>/<slug>/masks/frame_<idx>.png
  - compute mask IoU, Dice, boundary F1 (τ ∈ {1, 3, 5} px), mean boundary
    distance, box IoU

Reports:
  - overall: mean ± bootstrap 95 % CI for every metric (10 000 resamples)
  - per-sample: same metrics grouped by slug

Output JSON shape:
  {"overall": {metric: {"mean": ..., "ci_low": ..., "ci_high": ..., "n": N}},
   "per_sample": {slug: {metric: ..., "n": k}, ...},
   "frames": [{"slug": ..., "frame": ..., metric: ...}, ...]}
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = REPO_ROOT / "paper" / "data" / "labels_55.json"
DEFAULT_OUTPUT = REPO_ROOT / "paper" / "data" / "agreement_metrics.json"

BOUNDARY_TOLERANCES_PX = (1, 3, 5)
BOOTSTRAP_RESAMPLES = 10_000

FRAME_NAME_RE = re.compile(r"(?P<slug>.+)__frame_(?P<idx>\d+)\.png$")


def parse_frame_name(name: str) -> tuple[str, int]:
    match = FRAME_NAME_RE.search(name)
    if not match:
        raise ValueError(f"unexpected label image filename: {name!r}")
    return match.group("slug"), int(match.group("idx"))


def rasterize_coco_image(coco: dict[str, Any], image_id: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in coco["annotations"]:
        if ann["image_id"] != image_id:
            continue
        for segment in ann.get("segmentation", []):
            if not segment:
                continue
            pts = np.asarray(segment, dtype=np.float32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def load_predicted(runs_dir: Path, slug: str, frame_idx: int) -> np.ndarray | None:
    candidate = runs_dir / slug / "masks" / f"frame_{frame_idx + 1:06d}.png"
    # Vrifa writes 1-indexed (frame_index starts at 1 in run loop), source
    # video is 0-indexed; the extract script used 0-indexed names. Try both.
    if not candidate.exists():
        candidate = runs_dir / slug / "masks" / f"frame_{frame_idx:06d}.png"
    if not candidate.exists():
        return None
    arr = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return None
    return (arr > 127).astype(np.uint8)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return inter / union


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a_sum = int(a.sum())
    b_sum = int(b.sum())
    if a_sum + b_sum == 0:
        return 1.0
    inter = int(np.logical_and(a, b).sum())
    return 2.0 * inter / (a_sum + b_sum)


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros_like(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)


def boundary_f1_and_distance(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    gt_b = extract_boundary(gt)
    pred_b = extract_boundary(pred)
    if gt_b.sum() == 0 and pred_b.sum() == 0:
        return 1.0, 0.0
    if gt_b.sum() == 0 or pred_b.sum() == 0:
        return 0.0, float("inf")

    # Distance transform expects 0 on object pixels in some signatures;
    # we want distance to nearest boundary pixel, so build inverted maps.
    dt_gt = cv2.distanceTransform((gt_b == 0).astype(np.uint8), cv2.DIST_L2, 3)
    dt_pred = cv2.distanceTransform((pred_b == 0).astype(np.uint8), cv2.DIST_L2, 3)

    f1_per_tol: list[float] = []
    for tol in BOUNDARY_TOLERANCES_PX:
        precision = float((dt_gt[pred_b > 0] <= tol).mean())
        recall = float((dt_pred[gt_b > 0] <= tol).mean())
        if precision + recall == 0:
            f1_per_tol.append(0.0)
        else:
            f1_per_tol.append(2.0 * precision * recall / (precision + recall))
    f1_mean = float(np.mean(f1_per_tol))

    # Symmetric mean boundary distance.
    d_gt_to_pred = float(dt_pred[gt_b > 0].mean())
    d_pred_to_gt = float(dt_gt[pred_b > 0].mean())
    mean_distance = 0.5 * (d_gt_to_pred + d_pred_to_gt)
    return f1_mean, mean_distance


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask.sum() == 0:
        return None
    ys, xs = np.where(mask > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ba = bbox(a)
    bb = bbox(b)
    if ba is None and bb is None:
        return 1.0
    if ba is None or bb is None:
        return 0.0
    ax1, ay1, ax2, ay2 = ba
    bx1, by1, bx2, by2 = bb
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union


def bootstrap_ci(values: list[float], resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed=42)
    means = np.empty(resamples, dtype=float)
    for k in range(resamples):
        sample = rng.choice(finite, size=finite.size, replace=True)
        means[k] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    mean = float(finite.mean()) if finite.size else float("nan")
    low, high = bootstrap_ci(values)
    return {"mean": mean, "ci_low": low, "ci_high": high, "n": int(finite.size)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--runs-dir", type=Path, required=True,
                        help="directory of per-sample Rust runs, each with masks/")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.labels.exists():
        print(f"error: labels file not found at {args.labels}", file=sys.stderr)
        return 1

    coco = json.loads(args.labels.read_text())
    image_lookup = {img["id"]: img for img in coco["images"]}

    per_frame: list[dict[str, Any]] = []
    skipped = 0
    for image_id, image in image_lookup.items():
        try:
            slug, frame_idx = parse_frame_name(image["file_name"])
        except ValueError as exc:
            print(f"  WARN {exc}", file=sys.stderr)
            skipped += 1
            continue
        gt = rasterize_coco_image(coco, image_id, image["height"], image["width"])
        pred = load_predicted(args.runs_dir, slug, frame_idx)
        if pred is None:
            print(f"  WARN no prediction for {image['file_name']}", file=sys.stderr)
            skipped += 1
            continue
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            pred = (pred > 0).astype(np.uint8)

        bf1, bd = boundary_f1_and_distance(gt, pred)
        per_frame.append({
            "slug": slug,
            "frame": frame_idx,
            "iou": mask_iou(gt, pred),
            "dice": dice(gt, pred),
            "boundary_f1": bf1,
            "boundary_distance_px": bd,
            "box_iou": box_iou(gt, pred),
        })

    if not per_frame:
        print("no frames matched between labels and runs-dir", file=sys.stderr)
        return 1

    metric_keys = ["iou", "dice", "boundary_f1", "boundary_distance_px", "box_iou"]
    overall = {key: summarize([f[key] for f in per_frame]) for key in metric_keys}

    per_sample: dict[str, dict[str, Any]] = {}
    by_slug: dict[str, list[dict[str, Any]]] = {}
    for entry in per_frame:
        by_slug.setdefault(entry["slug"], []).append(entry)
    for slug, entries in by_slug.items():
        per_sample[slug] = {
            key: summarize([e[key] for e in entries]) for key in metric_keys
        }
        per_sample[slug]["n_frames"] = len(entries)

    output = {
        "overall": overall,
        "per_sample": per_sample,
        "frames": per_frame,
        "n_total": len(per_frame),
        "n_skipped": skipped,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "boundary_tolerances_px": list(BOUNDARY_TOLERANCES_PX),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"wrote {args.output}")
    print(f"  matched {len(per_frame)} frames, skipped {skipped}")
    print(f"  overall mean IoU = {overall['iou']['mean']:.4f}"
          f" (95% CI [{overall['iou']['ci_low']:.4f}, {overall['iou']['ci_high']:.4f}])")
    print(f"  overall mean boundary F1 = {overall['boundary_f1']['mean']:.4f}"
          f" (95% CI [{overall['boundary_f1']['ci_low']:.4f}, {overall['boundary_f1']['ci_high']:.4f}])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
