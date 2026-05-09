#!/usr/bin/env python3
"""Phase 3 fine-grained ablation around the input_1 optimum.

Phase 2 found the optimum at peak=off, lock=0, threshold_offset=-35,
min_area=800, IoU 0.8333 — but only sampled threshold_offset every 10
units, never crossed it with morph_kernel or blur_kernel, and didn't
explore the peak=off, lock=0 regime at min_area<100 or >800.

Phase 3 plan (everything at peak=off, lock=0 unless noted):

  A. threshold_offset axis at finer resolution: -42 to -18 by 2,
     min_area in {400, 800, 1200}.    -> 13 * 3 = 39 trials.

  B. morph_kernel x threshold_offset cross at the optimum strip:
     morph_kernel in {7, 11, 13, 15}, threshold_offset in {-32, -28, -24},
     min_area=800.                    -> 4 * 3 = 12 trials.

  C. blur_kernel x threshold_offset cross at the optimum:
     blur_kernel in {7, 9, 11, 13}, threshold_offset in {-32, -28},
     min_area=800.                    -> 4 * 2 = 8 trials.

  D. ROI margin scan at the optimum (input_1 has YouTube borders):
     roi_margin in {0.0, 0.02, 0.05, 0.08, 0.12, 0.15}.   -> 6 trials.

  Total: 65 trials.

Results merge into the existing labels file at paper/data/ablation_results.json
and the markdown summary at paper/data/ablation_summary.md.
"""

from __future__ import annotations

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


def trial_grid_phase3() -> list[tuple[str, list[str], dict]]:
    """Each entry: (label, vrifa_extra_flags, metadata_for_summary)."""
    trials: list[tuple[str, list[str], dict]] = []

    BASE = ["--no-peak-reference", "--lock-frames", "0"]

    # A. Fine threshold-offset axis at three min_area values.
    for ma in (400, 800, 1200):
        for to in range(-42, -16, 2):  # -42, -40, ..., -18
            label = f"p3a__to{to:+d}__ma{ma}"
            flags = BASE + [f"--threshold-offset={to}", "--min-area", str(ma)]
            trials.append((label, flags, {"phase": "A", "to": to, "ma": ma}))

    # B. morph_kernel x threshold_offset cross.
    for mk in (7, 11, 13, 15):
        for to in (-32, -28, -24):
            label = f"p3b__mk{mk}__to{to:+d}"
            flags = BASE + [
                "--min-area", "800",
                f"--threshold-offset={to}",
                "--morph-kernel", str(mk),
            ]
            trials.append((label, flags, {"phase": "B", "mk": mk, "to": to}))

    # C. blur_kernel x threshold_offset cross.
    for bk in (7, 9, 11, 13):
        for to in (-32, -28):
            label = f"p3c__bk{bk}__to{to:+d}"
            flags = BASE + [
                "--min-area", "800",
                f"--threshold-offset={to}",
                "--blur-kernel", str(bk),
            ]
            trials.append((label, flags, {"phase": "C", "bk": bk, "to": to}))

    # D. ROI margin scan at the optimum.
    for rm in (0.0, 0.02, 0.05, 0.08, 0.12, 0.15):
        label = f"p3d__roi{rm:.2f}"
        flags = BASE + [
            "--min-area", "800",
            "--threshold-offset=-30",
        ]
        trials.append((label, flags, {"phase": "D", "roi": rm}))

    return trials


def run_trial(label: str, extra_flags: list[str], roi_margin: str = "0.0") -> dict[str, Any]:
    trial_dir = TRIAL_ROOT / label
    sample_dir = trial_dir / "input_1"
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(VRIFA),
        "--video-path", str(VIDEO),
        "--output-dir", str(sample_dir),
        "--roi-margin", roi_margin,
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
    sorted_iou = sorted(ok, key=lambda r: r["iou_mean"] or 0, reverse=True)

    lines: list[str] = [
        "# input_1 ablation summary (phases 1-3)",
        "",
        f"Trials: {len(results)} run, {len(ok)} ok, {len(failed)} failed",
        "",
        "## Top-20 by mean IoU",
        "",
        "| Rank | Trial | IoU | 95% CI | Dice | F1_b | px | s |",
        "|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(sorted_iou[:20], 1):
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
    lines.append("| Trial | IoU | Dice | F1_b | s |")
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
    if not VRIFA.exists():
        print(f"vrifa binary not found at {VRIFA}", file=sys.stderr)
        return 1
    if not LABELS.exists():
        print(f"labels file not found at {LABELS}", file=sys.stderr)
        return 1

    TRIAL_ROOT.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else []
    print(f"starting with {len(results)} prior results")

    trials = trial_grid_phase3()
    print(f"running {len(trials)} phase-3 trials")
    for i, (label, flags, meta) in enumerate(trials, 1):
        roi = "0.0"
        if meta.get("phase") == "D":
            roi = f"{meta['roi']:.3f}"
        print(f"[{i:3d}/{len(trials)}] {label} ...", flush=True, end=" ")
        r = run_trial(label, flags, roi_margin=roi)
        results.append(r)
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
