"""Run a single ablation trial: invoke vrifa on one sample with a
config dict, run agreement against the ground truth, return metrics."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import agreement


@dataclass
class TrialConfig:
    """Self-contained trial spec. trial_id is the durable identity used
    for state, file paths, and CSV rows."""

    trial_id: str
    phase: str
    sample: str
    video_path: Path
    flags: dict[str, str | int | float | bool] = field(default_factory=dict)


@dataclass
class TrialResult:
    trial_id: str
    phase: str
    sample: str
    ok: bool
    iou_mean: float | None = None
    dice_mean: float | None = None
    boundary_f1_mean: float | None = None
    box_iou_mean: float | None = None
    runtime_s: float | None = None
    error: str | None = None
    flags: dict[str, Any] = field(default_factory=dict)


def flags_to_argv(flags: dict[str, Any]) -> list[str]:
    """Convert the trial's flag dict to a vrifa CLI argv list.

    Uses --flag=value form for value-bearing flags so clap does not
    misparse negative numbers (e.g. --threshold-offset=-30) as
    additional flags.
    """
    argv: list[str] = []
    for name, value in flags.items():
        # Private bookkeeping keys (e.g. __top5_kernels__ stashed on a
        # phase-7a winner) are not vrifa CLI flags. Skip them.
        if name.startswith("__"):
            continue
        cli_name = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if name in {"darken_only", "peak_reference"}:
                if value:
                    argv.append(cli_name)
                else:
                    argv.append(f"--no-{name.replace('_', '-')}")
            else:
                if value:
                    argv.append(cli_name)
        elif isinstance(value, (tuple, list)):
            # ref_mode is a multi-arg flag: ("first",) or ("prev", "10").
            # JSON-roundtripping a tuple gives a list, so accept both.
            # Multi-arg flags don't accept = syntax, so emit positionally
            # and rely on clap's num_args bound.
            argv.append(cli_name)
            argv.extend(str(v) for v in value)
        else:
            argv.append(f"{cli_name}={value}")
    return argv


def run_trial(
    trial: TrialConfig,
    binary: Path,
    labels: dict[str, Any],
    runs_root: Path,
    log_root: Path,
    timeout_s: float,
) -> TrialResult:
    """Execute one trial end-to-end. Always cleans up the run dir."""
    trial_dir = runs_root / trial.trial_id
    sample_dir = trial_dir / trial.sample
    masks_dir = sample_dir / "masks"
    if trial_dir.exists():
        shutil.rmtree(trial_dir, ignore_errors=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # --backend is a fast-vrifa-only flag. When running the CPU vrifa-rs
    # binary (e.g. for phases that exercise blur or threshold modes the
    # CUDA fast path doesn't support), strip it so vrifa-rs doesn't error.
    flags = dict(trial.flags)
    if Path(binary).name != "fast-vrifa" and "backend" in flags:
        flags.pop("backend")

    argv = [
        str(binary),
        "--video-path", str(trial.video_path),
        "--output-dir", str(sample_dir),
    ] + flags_to_argv(flags)

    start = time.monotonic()
    proc_result: dict[str, Any] = {"timeout": False, "rc": None,
                                    "stdout": "", "stderr": ""}
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        proc_result["rc"] = completed.returncode
        proc_result["stdout"] = completed.stdout or ""
        proc_result["stderr"] = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        proc_result["timeout"] = True
        proc_result["stdout"] = (exc.stdout or b"").decode("utf-8", errors="replace") \
            if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        proc_result["stderr"] = (exc.stderr or b"").decode("utf-8", errors="replace") \
            if isinstance(exc.stderr, bytes) else (exc.stderr or "")

    runtime_s = time.monotonic() - start

    if proc_result["timeout"] or proc_result["rc"] != 0:
        log_failure(log_root, trial, argv, proc_result)
        shutil.rmtree(trial_dir, ignore_errors=True)
        err_summary = "timeout" if proc_result["timeout"] else f"rc={proc_result['rc']}"
        return TrialResult(
            trial_id=trial.trial_id,
            phase=trial.phase,
            sample=trial.sample,
            ok=False,
            runtime_s=runtime_s,
            error=err_summary,
            flags=dict(trial.flags),
        )

    try:
        metrics = agreement.evaluate_sample(masks_dir, labels, trial.sample)
    except Exception as exc:  # noqa: BLE001 -- we want to capture anything
        log_failure(log_root, trial, argv, proc_result, agreement_error=str(exc))
        shutil.rmtree(trial_dir, ignore_errors=True)
        return TrialResult(
            trial_id=trial.trial_id,
            phase=trial.phase,
            sample=trial.sample,
            ok=False,
            runtime_s=runtime_s,
            error=f"agreement: {exc!s}",
            flags=dict(trial.flags),
        )
    finally:
        # Mask PNGs always get cleaned, success or fail.
        shutil.rmtree(trial_dir, ignore_errors=True)

    agg = metrics["aggregate"]
    return TrialResult(
        trial_id=trial.trial_id,
        phase=trial.phase,
        sample=trial.sample,
        ok=True,
        iou_mean=agg.get("iou_mean"),
        dice_mean=agg.get("dice_mean"),
        boundary_f1_mean=agg.get("boundary_f1_mean"),
        box_iou_mean=agg.get("box_iou_mean"),
        runtime_s=runtime_s,
        flags=dict(trial.flags),
    )


def log_failure(
    log_root: Path,
    trial: TrialConfig,
    argv: list[str],
    proc_result: dict[str, Any],
    agreement_error: str | None = None,
) -> None:
    """Drop the failure's full diagnostic into log_root/<trial_id>/."""
    fail_dir = log_root / trial.trial_id
    fail_dir.mkdir(parents=True, exist_ok=True)
    (fail_dir / "cli.txt").write_text(" ".join(argv) + "\n")
    (fail_dir / "stdout.txt").write_text(proc_result.get("stdout", "") or "")
    (fail_dir / "stderr.txt").write_text(proc_result.get("stderr", "") or "")
    info = {
        "trial_id": trial.trial_id,
        "phase": trial.phase,
        "sample": trial.sample,
        "rc": proc_result.get("rc"),
        "timeout": proc_result.get("timeout", False),
        "agreement_error": agreement_error,
        "flags": dict(trial.flags),
    }
    (fail_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    env_lines = [f"{k}={v}" for k, v in sorted(os.environ.items())
                 if k in {"PATH", "LD_LIBRARY_PATH", "OPENCV_VERSION", "RAYON_NUM_THREADS"}]
    (fail_dir / "env.txt").write_text("\n".join(env_lines) + "\n")
