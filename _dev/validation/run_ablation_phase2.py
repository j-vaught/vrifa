#!/usr/bin/env python3
"""Phase 2 ablation: re-run the failed threshold-offset trials with the
correct CLI syntax, then run a focused joint-optimum sweep around the
strongest single-knob wins from phase 1.

Phase 1 winners on input_1:
  - no_peak_reference: IoU 0.675
  - lock_frames_0:      IoU 0.640

Phase 2 plan:
  - Threshold-offset axis with `=` syntax: 9 values from -50 to -10.
  - Joint sweep: {peak on, peak off} x {lock_frames in 0, 3} x
    {threshold_offset in -45, -35, -25, -15} x {min_area in 100, 400, 800}.
    72 trials. Results merge into the same json/md as phase 1.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
VRIFA = REPO_ROOT / "vrifa-rs" / "target" / "release" / "vrifa"
LABELS = REPO_ROOT / "paper" / "data" / "labels_55.json"
VIDEO = REPO_ROOT / "data" / "input_1.mp4"
RESULTS_JSON = REPO_ROOT / "paper" / "data" / "ablation_results.json"
SUMMARY_MD = REPO_ROOT / "paper" / "data" / "ablation_summary.md"
TRIAL_ROOT = Path("/tmp/vrifa_ablation")


def trial_grid_phase2() -> list[tuple[str, list[str]]]:
    trials: list[tuple[str, list[str]]] = []

    # Threshold-offset axis with `=` syntax so clap doesn't parse -50 as a flag.
    for v in (-50, -45, -40, -35, -30, -25, -20, -15, -10):
        trials.append((f"threshold_offset_{v:+d}", [f"--threshold-offset={v}"]))

    # Joint-optimum sweep around the phase-1 winners.
    for peak in (True, False):
        for lock in (0, 3):
            for to in (-45, -35, -25, -15):
                for ma in (100, 400, 800):
                    label_parts = [
                        f"peak={'on' if peak else 'off'}",
                        f"lock={lock}",
                        f"to={to:+d}",
                        f"ma={ma}",
                    ]
                    flags = [
                        f"--threshold-offset={to}",
                        "--lock-frames", str(lock),
                        "--min-area", str(ma),
                    ]
                    if not peak:
                        flags.append("--no-peak-reference")
                    label = "joint__" + "__".join(label_parts).replace("=", "")
                    trials.append((label, flags))

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
    lines.append("# input_1 ablation summary (phase 1 + phase 2)")
    lines.append("")
    lines.append(f"Trials run: {len(results)}  ·  succeeded: {len(ok)}  ·  failed: {len(failed)}")
    lines.append("")

    if ok:
        sorted_iou = sorted(ok, key=lambda r: r["iou_mean"] or 0, reverse=True)
        lines.append("## Top-15 by mean IoU")
        lines.append("")
        lines.append("| Rank | Trial | IoU | 95% CI | Dice | Boundary F1 | Mean px | s |")
        lines.append("|---:|---|---:|---|---:|---:|---:|---:|")
        for i, r in enumerate(sorted_iou[:15], 1):
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
        lines.append("| Trial | IoU | Dice | Boundary F1 | s |")
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

    if RESULTS_JSON.exists():
        results = json.loads(RESULTS_JSON.read_text())
        # Drop the failed phase-1 trials so they don't pollute the merged report.
        results = [r for r in results if r.get("ok") or not r["label"].startswith("threshold_offset")]
    else:
        results = []
    print(f"starting with {len(results)} prior results")

    trials = trial_grid_phase2()
    print(f"running {len(trials)} phase-2 trials")
    for i, (label, flags) in enumerate(trials, 1):
        print(f"[{i:3d}/{len(trials)}] {label} ...", flush=True, end=" ")
        r = run_trial(label, flags)
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
