#!/usr/bin/env python3
"""Ablation driver for input_1 against the cleaned labels in labels_55.json.

For each trial, runs the locally-built vrifa CPU binary on input_1 with the
trial-specific flags, computes mask agreement against the labeled frames, and
deletes the per-trial mask output to keep disk usage bounded. Trial results
land in paper/data/ablation_results.json plus a markdown summary at
paper/data/ablation_summary.md.

Storage budget per trial: ~706 mask PNGs at single-channel 1920x1080 ~ 20 MB
peak before cleanup; agreement.py reads them, then the trial output dir is
removed. Total ablation never exceeds the cost of one trial.

Run from the repo root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
VRIFA = REPO_ROOT / "vrifa-rs" / "target" / "release" / "vrifa"
LABELS = REPO_ROOT / "paper" / "data" / "labels_55.json"
VIDEO = REPO_ROOT / "data" / "input_1.mp4"
RESULTS_JSON = REPO_ROOT / "paper" / "data" / "ablation_results.json"
SUMMARY_MD = REPO_ROOT / "paper" / "data" / "ablation_summary.md"
TRIAL_ROOT = Path("/tmp/vrifa_ablation")


def trial_grid() -> list[tuple[str, list[str]]]:
    """Build the trial list. Each entry: (label, extra_flags)."""
    trials: list[tuple[str, list[str]]] = []

    trials.append(("baseline_defaults", []))

    # Phase A — single-axis sweeps. Defaults: threshold-offset=-30, min-area=400,
    # morph-kernel=13, blur-kernel=9, lock-frames=3.
    for v in (-50, -45, -40, -35, -25, -20, -15, -10):
        trials.append((f"threshold_offset_{v:+d}", ["--threshold-offset", str(v)]))
    for v in (0, 50, 100, 200, 600, 800, 1200, 1600):
        trials.append((f"min_area_{v}", ["--min-area", str(v)]))
    for v in (3, 5, 7, 9, 11, 15, 17, 19, 21):
        trials.append((f"morph_kernel_{v}", ["--morph-kernel", str(v)]))
    for v in (3, 5, 7, 11, 13, 15):
        trials.append((f"blur_kernel_{v}", ["--blur-kernel", str(v)]))
    for v in (0, 1, 2, 5, 10, 20):
        trials.append((f"lock_frames_{v}", ["--lock-frames", str(v)]))

    # Phase B — categorical / on-off.
    trials.append(("colorspace_RGB", ["--colorspace", "RGB"]))
    trials.append(("colorspace_HSV", ["--colorspace", "HSV"]))
    trials.append(("colorspace_GRAYSCALE", ["--colorspace", "GRAYSCALE"]))
    trials.append(("no_darken_only", ["--no-darken-only"]))
    trials.append(("no_peak_reference", ["--no-peak-reference"]))
    trials.append(("ref_mode_running", ["--ref-mode", "running"]))
    trials.append(("morph_shape_rect", ["--morph-shape", "rect"]))
    trials.append(("morph_shape_cross", ["--morph-shape", "cross"]))
    trials.append(("skip_blur", ["--skip-blur"]))

    return trials


def run_trial(label: str, extra_flags: list[str]) -> dict[str, Any]:
    trial_dir = TRIAL_ROOT / label
    sample_dir = trial_dir / "input_1"
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(VRIFA),
        "--video-path", str(VIDEO),
        "--output-dir", str(sample_dir),
        "--roi-margin", "0.0",
        "--write-mask-pngs", "true",
        "--write-overlay-pngs", "false",
        "--write-heatmap-pngs", "false",
    ] + extra_flags

    start = time.monotonic()
    res = subprocess.run(cmd, capture_output=True, text=True)
    runtime = time.monotonic() - start
    if res.returncode != 0:
        shutil.rmtree(trial_dir, ignore_errors=True)
        return {
            "label": label,
            "flags": extra_flags,
            "ok": False,
            "error": (res.stderr or res.stdout).strip().splitlines()[-1:],
            "runtime_s": runtime,
        }

    agree_cmd = [
        sys.executable,
        str(REPO_ROOT / "_dev" / "validation" / "agreement.py"),
        "--labels", str(LABELS),
        "--runs-dir", str(trial_dir),
        "--output", str(trial_dir / "agreement.json"),
    ]
    agree = subprocess.run(agree_cmd, capture_output=True, text=True)
    if agree.returncode != 0:
        shutil.rmtree(trial_dir, ignore_errors=True)
        return {
            "label": label,
            "flags": extra_flags,
            "ok": False,
            "error": (agree.stderr or agree.stdout).strip().splitlines()[-1:],
            "runtime_s": runtime,
        }

    metrics = json.loads((trial_dir / "agreement.json").read_text())
    overall = metrics.get("overall", {})
    summary = {
        "label": label,
        "flags": extra_flags,
        "ok": True,
        "runtime_s": runtime,
        "n": metrics.get("n_total", 0),
        "iou_mean": overall.get("iou", {}).get("mean"),
        "iou_ci_low": overall.get("iou", {}).get("ci_low"),
        "iou_ci_high": overall.get("iou", {}).get("ci_high"),
        "dice_mean": overall.get("dice", {}).get("mean"),
        "boundary_f1_mean": overall.get("boundary_f1", {}).get("mean"),
        "boundary_distance_mean": overall.get("boundary_distance_px", {}).get("mean"),
        "box_iou_mean": overall.get("box_iou", {}).get("mean"),
    }
    shutil.rmtree(trial_dir, ignore_errors=True)
    return summary


def write_summary(results: list[dict[str, Any]]) -> None:
    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    lines: list[str] = []
    lines.append("# input_1 ablation summary")
    lines.append("")
    lines.append(f"Trials run: {len(results)}  ·  succeeded: {len(ok)}  ·  failed: {len(failed)}")
    lines.append("")

    if ok:
        sorted_iou = sorted(ok, key=lambda r: r["iou_mean"] or 0, reverse=True)
        lines.append("## Top-10 by mean IoU")
        lines.append("")
        lines.append("| Rank | Trial | IoU | 95% CI | Dice | Boundary F1 | Mean boundary px | Runtime s |")
        lines.append("|---:|---|---:|---|---:|---:|---:|---:|")
        for i, r in enumerate(sorted_iou[:10], 1):
            lines.append(
                f"| {i} | {r['label']} | "
                f"{r['iou_mean']:.4f} | "
                f"[{r['iou_ci_low']:.4f}, {r['iou_ci_high']:.4f}] | "
                f"{r['dice_mean']:.4f} | "
                f"{r['boundary_f1_mean']:.4f} | "
                f"{r['boundary_distance_mean']:.1f} | "
                f"{r['runtime_s']:.1f} |"
            )
        lines.append("")
        lines.append("## All trials by mean IoU")
        lines.append("")
        lines.append("| Trial | IoU | Dice | Boundary F1 | Runtime s |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in sorted_iou:
            lines.append(
                f"| {r['label']} | {r['iou_mean']:.4f} | "
                f"{r['dice_mean']:.4f} | "
                f"{r['boundary_f1_mean']:.4f} | "
                f"{r['runtime_s']:.1f} |"
            )
        lines.append("")

    if failed:
        lines.append("## Failed trials")
        lines.append("")
        for r in failed:
            err = "; ".join(r.get("error") or ["(no detail)"])
            lines.append(f"- **{r['label']}** — {err}")
        lines.append("")

    SUMMARY_MD.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N trials (for smoke-testing)")
    args = parser.parse_args()

    if not VRIFA.exists():
        print(f"vrifa binary not found at {VRIFA}", file=sys.stderr)
        return 1
    if not LABELS.exists():
        print(f"labels file not found at {LABELS}", file=sys.stderr)
        return 1

    TRIAL_ROOT.mkdir(parents=True, exist_ok=True)
    trials = trial_grid()
    if args.limit:
        trials = trials[: args.limit]

    print(f"running {len(trials)} trials")
    results: list[dict[str, Any]] = []
    for i, (label, flags) in enumerate(trials, 1):
        print(f"[{i:3d}/{len(trials)}] {label} ...", flush=True, end=" ")
        r = run_trial(label, flags)
        results.append(r)
        # Persist after every trial so partial progress is recoverable.
        RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_JSON.write_text(json.dumps(results, indent=2))
        write_summary(results)
        if r.get("ok"):
            print(f"IoU={r['iou_mean']:.4f}  ({r['runtime_s']:.1f}s)")
        else:
            err = "; ".join(r.get("error") or ["?"])
            print(f"FAILED: {err}")

    print(f"\nresults: {RESULTS_JSON}")
    print(f"summary: {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
