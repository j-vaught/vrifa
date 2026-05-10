"""IoU / Dice / boundary-F1 against the COCO ground truth, restricted to
one sample's labeled frames.

Reads the trimmed labels JSON (each image entry's `file_name` already
points at the trimmed video's frame index) and the matching mask PNGs
written by vrifa. Returns per-frame metrics and an aggregate summary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

FRAME_NAME_RE = re.compile(r"^(?P<slug>input_\d+)__frame_(?P<idx>\d+)\.png$")
BOUNDARY_TOLERANCES_PX = (1, 3, 5)


def parse_frame_name(name: str) -> tuple[str, int]:
    m = FRAME_NAME_RE.match(name)
    if not m:
        raise ValueError(f"unexpected label image filename: {name!r}")
    return m.group("slug"), int(m.group("idx"))


def rasterize_polygon(segmentation: list[list[float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        if not polygon:
            continue
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def load_ground_truth(labels: dict[str, Any], sample: str) -> dict[int, np.ndarray]:
    """Return {frame_index: gt_mask} for the requested sample."""
    images_by_id: dict[int, dict[str, Any]] = {}
    for img in labels["images"]:
        slug, idx = parse_frame_name(img["file_name"])
        if slug == sample:
            images_by_id[img["id"]] = {**img, "frame_index": idx}

    gts: dict[int, np.ndarray] = {}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in labels["annotations"]:
        if ann["image_id"] in images_by_id:
            annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    for image_id, img in images_by_id.items():
        anns = annotations_by_image.get(image_id, [])
        h, w = img["height"], img["width"]
        merged = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            seg = ann.get("segmentation", [])
            if not isinstance(seg, list) or not seg:
                continue
            merged |= rasterize_polygon(seg, h, w)
        gts[img["frame_index"]] = merged

    return gts


def load_predicted_mask(masks_dir: Path, frame_index: int) -> np.ndarray | None:
    """Load the predicted mask for a frame, or None if missing."""
    candidate = masks_dir / f"frame_{frame_index:06d}.png"
    if not candidate.exists():
        return None
    img = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return (img > 127).astype(np.uint8)


def boundary_distance_pixels(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt_edge = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    pred_edge = cv2.morphologyEx(pred, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    if gt_edge.sum() == 0 or pred_edge.sum() == 0:
        return np.array([], dtype=np.float32)
    inv_gt = (1 - (gt_edge > 0).astype(np.uint8)) * 255
    dt_to_gt = cv2.distanceTransform(inv_gt, cv2.DIST_L2, 5)
    return dt_to_gt[pred_edge > 0].astype(np.float32)


def per_frame_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    gt_b = gt > 0
    pred_b = pred > 0
    inter = int(np.logical_and(gt_b, pred_b).sum())
    union = int(np.logical_or(gt_b, pred_b).sum())
    iou = inter / union if union > 0 else 1.0
    gt_sum = int(gt_b.sum())
    pred_sum = int(pred_b.sum())
    dice = 2 * inter / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 1.0

    # Boundary F1 averaged across tolerance thresholds.
    distances = boundary_distance_pixels(gt.astype(np.uint8), pred.astype(np.uint8))
    boundary_f1s = []
    for tol in BOUNDARY_TOLERANCES_PX:
        if distances.size == 0:
            boundary_f1s.append(1.0 if (gt_sum == 0 and pred_sum == 0) else 0.0)
            continue
        precision = float((distances <= tol).mean())
        # Recall: distance from gt edge to pred edge (transpose).
        rev_distances = boundary_distance_pixels(pred.astype(np.uint8), gt.astype(np.uint8))
        if rev_distances.size == 0:
            recall = 0.0
        else:
            recall = float((rev_distances <= tol).mean())
        if precision + recall == 0:
            boundary_f1s.append(0.0)
        else:
            boundary_f1s.append(2 * precision * recall / (precision + recall))
    boundary_f1 = float(np.mean(boundary_f1s)) if boundary_f1s else 0.0

    # Bounding-box IoU (cheaper-than-mask aggregate).
    box_iou = compute_box_iou(gt_b, pred_b)

    return {
        "iou": iou,
        "dice": dice,
        "boundary_f1": boundary_f1,
        "box_iou": box_iou,
        "gt_pixels": gt_sum,
        "pred_pixels": pred_sum,
    }


def compute_box_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        ys, xs = np.where(mask)
        if ys.size == 0:
            return None
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())

    a = bbox(gt)
    b = bbox(pred)
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    iy0 = max(a[0], b[0])
    ix0 = max(a[1], b[1])
    iy1 = min(a[2], b[2])
    ix1 = min(a[3], b[3])
    inter_h = max(0, iy1 - iy0 + 1)
    inter_w = max(0, ix1 - ix0 + 1)
    inter = inter_h * inter_w
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_sample(
    masks_dir: Path,
    labels: dict[str, Any],
    sample: str,
) -> dict[str, Any]:
    """Compute per-frame and aggregate metrics for one sample."""
    gts = load_ground_truth(labels, sample)
    if not gts:
        raise ValueError(f"no labels found for sample {sample!r}")

    frame_results = []
    for frame_idx, gt in sorted(gts.items()):
        pred = load_predicted_mask(masks_dir, frame_idx)
        if pred is None:
            frame_results.append({
                "frame_index": frame_idx,
                "missing_pred": True,
            })
            continue
        if pred.shape != gt.shape:
            # Defensive: vrifa should write masks at the source-video
            # resolution, which matches the label's `width` and `height`.
            # If they differ, scale ground truth to predicted size.
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        m = per_frame_metrics(gt, pred)
        frame_results.append({"frame_index": frame_idx, **m})

    valid = [f for f in frame_results if "iou" in f]
    n_valid = len(valid)
    n_missing = len(frame_results) - n_valid

    aggregate: dict[str, Any] = {"n_valid": n_valid, "n_missing": n_missing}
    if valid:
        for key in ("iou", "dice", "boundary_f1", "box_iou"):
            values = np.array([f[key] for f in valid], dtype=np.float64)
            aggregate[f"{key}_mean"] = float(values.mean())
            if values.size >= 2:
                aggregate[f"{key}_std"] = float(values.std(ddof=1))
            else:
                aggregate[f"{key}_std"] = 0.0

    return {"frames": frame_results, "aggregate": aggregate}
