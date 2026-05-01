#!/usr/bin/env python3
"""Build reproducible data tables and raster assets for the VRIFA paper drafts."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper"
DATA_DIR = PAPER_DIR / "data"
ASSET_DIR = PAPER_DIR / "assets" / "generated"

GARNET = (115, 0, 10)
ATLANTIC = (70, 106, 159)
BLACK = (0, 0, 0)
WARM_GRAY = (103, 97, 86)
WHITE = (255, 255, 255)

FRAME_RE = re.compile(r"frame_(\d+)\.png")


RUNS = [
    {
        "slug": "input1",
        "display_name": "Run A",
        "video": ROOT / "data" / "input_1.mp4",
        "summary": ROOT / "outputs_run" / "run_summary.yaml",
        "coco": ROOT / "outputs_run" / "formatCOCO" / "annotations" / "instances_default.json",
    },
    {
        "slug": "input2",
        "display_name": "Run B",
        "video": ROOT / "data" / "input_2.mp4",
        "summary": ROOT / "outputs_run2" / "run_summary.yaml",
        "coco": ROOT / "outputs_run2" / "formatCOCO" / "annotations" / "instances_default.json",
    },
    {
        "slug": "input3",
        "display_name": "Run C",
        "video": ROOT / "data" / "input_3.mp4",
        "summary": ROOT / "outputs_run3" / "run_summary.yaml",
        "coco": ROOT / "outputs_run3" / "formatCOCO" / "annotations" / "instances_default.json",
    },
]


PERFORMANCE = {
    "baseline": {
        "objective_score": 0.583,
        "mask_iou": 0.737,
        "dice_f1": 0.847,
        "boundary_f1": 0.206,
        "boundary_distance_px": 138.8,
        "box_iou": 0.837,
        "predicted_area_fraction": 0.769,
        "runtime_ms": 131.6,
    },
    "optimized": {
        "objective_score": 0.807,
        "mask_iou": 0.935,
        "dice_f1": 0.966,
        "boundary_f1": 0.559,
        "boundary_distance_px": 61.5,
        "box_iou": 0.902,
        "predicted_area_fraction": 0.583,
        "runtime_ms": 120.7,
    },
    "stage1_best": {
        "objective_score": 0.777,
        "description": "Threshold offset family with offset -12",
    },
    "stage2_best": {
        "objective_score": 0.595,
        "description": "Threshold offset -30, morphology kernel 7, minimum area 500",
    },
    "best_configuration": {
        "reference": "Peak brightness",
        "colorspace": "RGB",
        "darken_only": True,
        "threshold_offset": -33.812,
        "blur_kernel": 7,
        "morph_kernel": 11,
        "min_area": 525,
    },
}


RUNTIME = [
    {"stage": "Baseline pipeline", "trials": 1, "seconds": 59.8},
    {"stage": "Stage 1 ablation", "trials": 23, "seconds": 21 * 60 + 52},
    {"stage": "Stage 2 ablation", "trials": 27, "seconds": 26 * 60 + 42},
    {"stage": "Mixed-variable optimization", "trials": 40, "seconds": 31 * 60 + 29},
]


def frame_number(file_name: str) -> int:
    match = FRAME_RE.fullmatch(file_name)
    if not match:
        raise ValueError(f"Unexpected frame name: {file_name}")
    return int(match.group(1))


def resize_frame(frame: np.ndarray, max_width: int = 1200) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def parse_polygon(segmentation: list[float]) -> np.ndarray:
    pts = np.array(segmentation, dtype=np.float32).reshape(-1, 2)
    return np.round(pts).astype(np.int32)


def polygon_area(segmentation: list[float]) -> float:
    pts = np.array(segmentation, dtype=np.float64).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def read_video_frame(video_path: Path, frame_idx_1based: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_1based - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx_1based} from {video_path}")
    return frame


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def overlay_annotations(
    frame: np.ndarray,
    annotations: list[dict],
    scale: float,
) -> np.ndarray:
    overlay = frame.copy()
    drawn = frame.copy()

    for ann in annotations:
        if not ann.get("segmentation"):
            continue
        fill_color = GARNET
        edge_color = BLACK
        thickness = max(2, int(3 * scale))
        for seg in ann["segmentation"]:
            polygon = parse_polygon(seg)
            cv2.fillPoly(overlay, [polygon], fill_color)
            cv2.polylines(drawn, [polygon], True, edge_color, thickness)

        x, y, w, h = [int(v) for v in ann["bbox"]]
        cv2.rectangle(drawn, (x, y), (x + w, y + h), ATLANTIC, thickness)

    merged = cv2.addWeighted(overlay, 0.28, drawn, 0.72, 0)
    return merged


def make_progression_rows(run: dict, summary: dict, coco: dict) -> list[dict]:
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    rows: list[dict] = []
    roi_pixels = float(summary["roi_pixel_count"])
    fps = float(summary["video_fps"])
    for image in sorted(coco["images"], key=lambda item: item["id"]):
        image_id = image["id"]
        frame_idx = frame_number(image["file_name"])
        anns = anns_by_image.get(image_id, [])
        region_area = sum(float(ann.get("area") or 0.0) for ann in anns)
        if region_area <= 0.0:
            region_area = sum(
                polygon_area(seg)
                for ann in anns
                for seg in ann.get("segmentation", [])
            )
        rows.append(
            {
                "slug": run["slug"],
                "display_name": run["display_name"],
                "frame_index": frame_idx,
                "time_seconds": round((frame_idx - 1) / fps, 5),
                "width": image["width"],
                "height": image["height"],
                "wet_area_px": round(region_area, 3),
                "dry_area_px": 0.0,
                "wet_fraction_roi": round(region_area / roi_pixels, 6),
                "wet_fraction_frame": round(region_area / (image["width"] * image["height"]), 6),
                "annotation_count": len(anns),
            }
        )
    return rows


def sample_progression(rows: list[dict], target_points: int = 90) -> list[dict]:
    step = max(1, math.ceil(len(rows) / target_points))
    sampled = rows[::step]
    if sampled[-1]["frame_index"] != rows[-1]["frame_index"]:
        sampled.append(rows[-1])
    max_time = rows[-1]["time_seconds"] if rows else 1.0
    max_wet = max((row["wet_fraction_roi"] for row in rows), default=1.0)
    reduced = []
    for row in sampled:
        reduced.append(
            {
                "slug": row["slug"],
                "display_name": row["display_name"],
                "frame_index": row["frame_index"],
                "time_norm": round(row["time_seconds"] / max_time if max_time else 0.0, 6),
                "wet_norm": round(row["wet_fraction_roi"] / max_wet if max_wet else 0.0, 6),
                "wet_fraction_roi": row["wet_fraction_roi"],
            }
        )
    return reduced


def choose_showcase_frames(rows: list[dict]) -> list[dict]:
    positive_rows = [row for row in rows if row["wet_fraction_roi"] > 0.005]
    pool = positive_rows if positive_rows else rows
    max_wet = max(row["wet_fraction_roi"] for row in pool)
    targets = [0.15 * max_wet, 0.5 * max_wet, 0.85 * max_wet]
    labels = ["Early wetting", "Mid infusion", "Late fill"]
    chosen: list[dict] = []
    used = set()
    for label, target in zip(labels, targets):
        ordered = sorted(pool, key=lambda row: abs(row["wet_fraction_roi"] - target))
        for row in ordered:
            if row["frame_index"] in used:
                continue
            chosen.append({"label": label, **row})
            used.add(row["frame_index"])
            break
    return chosen


def build_showcase_assets(run: dict, coco: dict, selected: list[dict]) -> list[dict]:
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    file_to_id: dict[str, int] = {}
    for image in coco["images"]:
        file_to_id[image["file_name"]] = image["id"]
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    generated = []
    for item in selected:
        frame_idx = item["frame_index"]
        file_name = f"frame_{frame_idx:06d}.png"
        image_id = file_to_id[file_name]
        frame = read_video_frame(run["video"], frame_idx)
        scale = resize_frame(frame).shape[1] / frame.shape[1]
        frame = resize_frame(frame)

        annotations = anns_by_image.get(image_id, [])
        scaled_annotations = []
        for ann in annotations:
            scaled = dict(ann)
            scaled["bbox"] = [value * scale for value in ann["bbox"]]
            scaled["segmentation"] = [
                [
                    coord * scale
                    for coord in seg
                ]
                for seg in ann.get("segmentation", [])
            ]
            scaled_annotations.append(scaled)

        overlay = overlay_annotations(frame, scaled_annotations, scale)

        raw_rel = Path("assets/generated") / f"{run['slug']}_{frame_idx:06d}_raw.png"
        overlay_rel = Path("assets/generated") / f"{run['slug']}_{frame_idx:06d}_overlay.png"
        cv2.imwrite(str(PAPER_DIR / raw_rel), frame)
        cv2.imwrite(str(PAPER_DIR / overlay_rel), overlay)

        generated.append(
            {
                "label": item["label"],
                "frame_index": frame_idx,
                "time_seconds": item["time_seconds"],
                "wet_fraction_roi": item["wet_fraction_roi"],
                "raw": raw_rel.as_posix(),
                "overlay": overlay_rel.as_posix(),
            }
        )
    return generated


def extract_yolo_demo_frame() -> str:
    demo_video = ROOT / "yolo_overlay_input4.mp4"
    frame = read_video_frame(demo_video, 120)
    frame = resize_frame(frame)
    rel = Path("assets/generated") / "yolo_overlay_demo.png"
    cv2.imwrite(str(PAPER_DIR / rel), frame)
    return rel.as_posix()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    repo_runs = []
    sampled_progression = []
    showcase_assets = []

    total_images = 0
    total_annotations = 0

    for run in RUNS:
        summary = load_yaml(run["summary"])
        coco = load_json(run["coco"])
        image_count = len(coco["images"])
        annotation_count = len(coco["annotations"])
        total_images += image_count
        total_annotations += annotation_count

        repo_runs.append(
            {
                "slug": run["slug"],
                "display_name": run["display_name"],
                "video": run["video"].name,
                "processed_frames": summary["processed_frames"],
                "duration_seconds": round(float(summary["video_duration_seconds"]), 3),
                "video_fps": round(float(summary["video_fps"]), 3),
                "colorspace": summary["colorspace"],
                "roi_fraction": round(float(summary["roi_fraction"]), 3),
                "roi_pixel_count": int(summary["roi_pixel_count"]),
                "threshold_offset": float(summary["threshold_offset"]),
                "darken_only": bool(summary["darken_only"]),
                "peak_reference": bool(summary["peak_reference"]),
                "annotation_formats": summary["annotation_formats"],
                "image_count": image_count,
                "annotation_count": annotation_count,
                "average_compute_time_seconds": round(float(summary["average_compute_time_seconds"]), 4),
            }
        )

        progression_rows = make_progression_rows(run, summary, coco)
        csv_path = DATA_DIR / f"{run['slug']}_progression.csv"
        write_csv(
            csv_path,
            progression_rows,
            [
                "slug",
                "display_name",
                "frame_index",
                "time_seconds",
                "width",
                "height",
                "wet_area_px",
                "dry_area_px",
                "wet_fraction_roi",
                "wet_fraction_frame",
                "annotation_count",
            ],
        )

        sampled_progression.extend(sample_progression(progression_rows))

        if run["slug"] == "input1":
            selected = choose_showcase_frames(progression_rows)
            showcase_assets = build_showcase_assets(run, coco, selected)

    paper_data = {
        "performance": PERFORMANCE,
        "runtime": RUNTIME,
        "repo_runs": repo_runs,
        "repo_totals": {
            "image_count": total_images,
            "annotation_count": total_annotations,
            "run_count": len(repo_runs),
            "trial_count": sum(item["trials"] for item in RUNTIME),
        },
        "showcase_assets": showcase_assets,
        "sampled_progression": sampled_progression,
        "yolo_demo": extract_yolo_demo_frame(),
        "sources": {
            "draft_pdf": "../../Downloads/vrifa_aiaa_extended_abstract.pdf",
            "repo_root": str(ROOT),
        },
    }

    with (DATA_DIR / "paper_data.json").open("w") as handle:
        json.dump(paper_data, handle, indent=2)

    print(f"Wrote paper data to {DATA_DIR / 'paper_data.json'}")


if __name__ == "__main__":
    main()
