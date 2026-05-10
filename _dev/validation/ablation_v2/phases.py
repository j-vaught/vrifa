"""Phase definitions: each phase emits a list of TrialConfigs given the
prior-stage per-sample winners. Phase order matches BENCHMARK.md."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Callable

from .config import (
    INTEGRATED_DEFAULTS,
    OUTPUT_FLAGS,
    ROI_MARGIN_BY_SAMPLE,
    SAMPLE_STABILIZATION,
    SAMPLES_FULL,
)
from .trial import TrialConfig


def _format_value(value: Any) -> str:
    """Compact value rendering for trial_id."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, tuple):
        return "-".join(_format_value(v) for v in value)
    s = str(value)
    return s.replace(":", "-").replace(",", "_").replace("/", "-").replace(" ", "")


def make_trial_id(phase: str, sample: str, overrides: dict[str, Any]) -> str:
    parts = [f"phase{phase}", sample]
    for key in sorted(overrides.keys()):
        parts.append(f"{key}={_format_value(overrides[key])}")
    return "__".join(parts)


def base_config_for(sample: str, prior_winner: dict[str, Any] | None) -> dict[str, Any]:
    """Return the trial-baseline flags for a sample.

    For Phase 1 the prior_winner is None and we use INTEGRATED_DEFAULTS.
    For later phases the per-sample prior winner is the chained best.
    The ROI margin is video-specific (input_1 keeps default margin,
    pre-cropped videos use 0).
    """
    base = dict(INTEGRATED_DEFAULTS) if prior_winner is None else dict(prior_winner)
    base.update(OUTPUT_FLAGS)
    base["roi_margin"] = ROI_MARGIN_BY_SAMPLE.get(sample, 0.15)
    return base


def make_trials(
    phase: str,
    sample: str,
    video_path: Path,
    prior_winner: dict[str, Any] | None,
    overrides_iter: list[dict[str, Any]],
) -> list[TrialConfig]:
    """Build TrialConfigs by overlaying overrides on the per-sample base."""
    trials: list[TrialConfig] = []
    base = base_config_for(sample, prior_winner)
    for overrides in overrides_iter:
        flags = dict(base)
        flags.update(overrides)
        trial_id = make_trial_id(phase, sample, overrides)
        trials.append(TrialConfig(
            trial_id=trial_id,
            phase=phase,
            sample=sample,
            video_path=video_path,
            flags=flags,
        ))
    return trials


# --- Phase 1: rough per-video baseline -------------------------------------

def phase1_overrides() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for to, ma, mk, lf in product(
        [-50, -30, -10],
        [100, 400, 1600],
        [7, 13, 21],
        [0, 3, 10],
    ):
        grid.append({
            "threshold_offset": float(to),
            "min_area": ma,
            "morph_kernel": mk,
            "lock_frames": lf,
        })
    return grid


# --- Phase 2: colorspace + channel weights ---------------------------------

PHASE2_WEIGHTS = {
    "CIELAB": [
        "1,0,0", "0,1,0", "0,0,1",
        "1,0.5,0", "1,0,0.5", "1,0.5,0.5",
        "1,1,0", "1,0,1", "0,1,1",
        "1,1,1", "0.5,1,1",
        "0.7,0.7,0.3", "0.3,0.7,0.7", "0.7,0.3,0.7",
        "0.5,0.7,0.3", "0.3,0.5,0.7", "0.6,0.2,0.2",
    ],
    "RGB": [
        "1,1,1", "1,0,0", "0,1,0", "0,0,1",
        "0.299,0.587,0.114",
        "0.5,0.5,0", "0.5,0,0.5", "0,0.5,0.5",
        "0.7,0.2,0.1", "0.2,0.7,0.1", "0.1,0.2,0.7",
        "0.4,0.4,0.2", "0.6,0.3,0.1", "0.3,0.6,0.1",
        "0.1,0.3,0.6", "0.33,0.34,0.33", "0.5,0.3,0.2",
    ],
    "HSV": [
        "0,0,1", "0,1,0", "1,0,0",
        "0,1,1", "1,0,1", "1,1,0",
        "1,1,1", "0.3,0.3,0.4", "0.2,0.4,0.4",
    ],
    "GRAYSCALE": ["1"],
}


def phase2_overrides() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for cs, weights_list in PHASE2_WEIGHTS.items():
        for w in weights_list:
            for darken in (True, False):
                grid.append({
                    "colorspace": cs,
                    "channel_weights": w,
                    "darken_only": darken,
                })
    return grid


# --- Phase 3: reference selection ------------------------------------------

def phase3_overrides() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    grid.append({"ref_mode": ("first",)})
    for alpha in (0.01, 0.02, 0.05, 0.1, 0.2):
        grid.append({"ref_mode": ("running",), "ref_running_alpha": alpha})
    for k in (1, 3, 10, 30):
        grid.append({"ref_mode": ("prev", k)})
    for tf, ls in product((0.1, 0.2, 0.3, 0.5), (0.5, 1.0, 2.0)):
        grid.append({
            "ref_mode": ("dynamic",),
            "dynamic_target_fraction": tf,
            "dynamic_lag_scale": ls,
        })
    return grid


# --- Phase 4: threshold mode -----------------------------------------------

def phase4_overrides() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for off in (-60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10):
        grid.append({"threshold": "otsu", "threshold_offset": float(off)})
    for off in (-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30):
        grid.append({"threshold": "triangle", "threshold_offset": float(off)})
    for v in (20, 40, 60, 80, 100, 120, 140, 160, 180):
        grid.append({"threshold": f"manual:{v}"})
    for p in (40, 50, 60, 70, 80, 85, 90, 95, 98):
        grid.append({"threshold": f"percentile:{p}"})
    for b, c in product((7, 11, 21, 31, 51, 71), (-10, -5, 0, 5, 10, 15)):
        grid.append({"threshold": f"adaptive-mean:{b}:{c}"})
    for b, c in product((7, 11, 21, 31, 51, 71), (-10, -5, 0, 5, 10, 15)):
        grid.append({"threshold": f"adaptive-gaussian:{b}:{c}"})
    return grid


# --- Phase 5: pre-delta blur -----------------------------------------------

def phase5_overrides() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = [{"pre_delta_blur": "none"}]
    sizes_basic = (3, 5, 7, 9, 11, 13, 15, 17, 21)
    for kind in ("flat", "gaussian", "triangle"):
        for s in sizes_basic:
            grid.append({"pre_delta_blur": f"{kind}:{s}"})
    for s in (3, 5):
        grid.append({"pre_delta_blur": f"median:{s}"})
    for s in (5, 9, 15):
        grid.append({"pre_delta_blur": f"bilateral:{s}"})
    return grid


# --- Phase 6: post-delta blur ----------------------------------------------

def phase6_overrides() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = [{"blur": "none"}]
    sizes_basic = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25)
    for kind in ("flat", "gaussian", "triangle"):
        for s in sizes_basic:
            grid.append({"blur": f"{kind}:{s}"})
    for s in (3, 5):
        grid.append({"blur": f"median:{s}"})
    for s in (5, 9, 15):
        grid.append({"blur": f"bilateral:{s}"})
    return grid


# --- Phase 7a: morph kernel x shape, iters at (1, 1) ----------------------

PHASE7A_KERNELS = (3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 31, 41, 51, 71, 101, 151)
PHASE7A_SHAPES = ("ellipse", "rect", "cross")


def phase7a_overrides() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for kernel, shape in product(PHASE7A_KERNELS, PHASE7A_SHAPES):
        grid.append({
            "morph_kernel": kernel,
            "morph_shape": shape,
            "morph_close_iterations": 1,
            "morph_open_iterations": 1,
        })
    return grid


# --- Phase 7b: iters x top-5 kernel from 7a per sample --------------------

def phase7b_overrides_for_sample(top_kernels: list[int], shape: str) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for kernel in top_kernels:
        for ci, oi in product((1, 2, 3), (1, 2, 3)):
            grid.append({
                "morph_kernel": kernel,
                "morph_shape": shape,
                "morph_close_iterations": ci,
                "morph_open_iterations": oi,
            })
    return grid


# --- Phase 8: lock fine sweep ----------------------------------------------

PHASE8_LOCKS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 60, 90)


def phase8_overrides() -> list[dict[str, Any]]:
    return [{"lock_frames": lf} for lf in PHASE8_LOCKS]


# --- Phase 10: 5^3 joint perturbation around chained best -----------------

PHASE10_THRESHOLD_OFFSET_STEPS = (-10.0, -5.0, 0.0, 5.0, 10.0)
PHASE10_MORPH_KERNEL_STEP_INDICES = (-2, -1, 0, 1, 2)
PHASE10_LOCK_FRAMES_STEP_INDICES = (-2, -1, 0, 1, 2)
# Dense ladder of valid morph kernel sizes; ±1 step jumps to the next or
# previous odd kernel in the ladder.
PHASE10_KERNEL_LADDER = (3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 31, 41, 51, 71, 101, 151)


def _ladder_step(value: int, ladder: tuple[int, ...], step: int) -> int:
    if value not in ladder:
        # Snap to nearest, then step.
        nearest = min(ladder, key=lambda v: abs(v - value))
    else:
        nearest = value
    idx = ladder.index(nearest) + step
    idx = max(0, min(len(ladder) - 1, idx))
    return ladder[idx]


def phase10_overrides_for_sample(prior_winner: dict[str, Any]) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    base_offset = float(prior_winner.get("threshold_offset", -30.0))
    base_kernel = int(prior_winner.get("morph_kernel", 13))
    base_lock = int(prior_winner.get("lock_frames", 3))

    for off_step, kern_step, lock_step in product(
        PHASE10_THRESHOLD_OFFSET_STEPS,
        PHASE10_MORPH_KERNEL_STEP_INDICES,
        PHASE10_LOCK_FRAMES_STEP_INDICES,
    ):
        new_offset = base_offset + off_step
        new_kernel = _ladder_step(base_kernel, PHASE10_KERNEL_LADDER, kern_step)
        new_lock = max(0, base_lock + lock_step)
        grid.append({
            "threshold_offset": new_offset,
            "morph_kernel": new_kernel,
            "lock_frames": new_lock,
        })
    return grid


# --- Phase 9: stabilization (input_1 only) ---------------------------------

def phase9_overrides() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = [{"camera_stable": False}]
    for model, per_frame, cumulative in product(
        ("translation", "affine"),
        (0.5, 1.0, 1.5, 2.5, 5.0),
        (1.0, 2.0, 3.0, 5.0, 10.0),
    ):
        grid.append({
            "camera_stable": True,
            "motion_model": model,
            "motion_per_frame_threshold": per_frame,
            "cumulative_motion_threshold": cumulative,
        })
    return grid


# --- Phase entry table -----------------------------------------------------

PHASE_SAMPLES = {
    "1": SAMPLES_FULL, "2": SAMPLES_FULL, "3": SAMPLES_FULL, "4": SAMPLES_FULL,
    "5": SAMPLES_FULL, "6": SAMPLES_FULL, "7a": SAMPLES_FULL, "7b": SAMPLES_FULL,
    "8": SAMPLES_FULL, "10": SAMPLES_FULL,
    "9": [SAMPLE_STABILIZATION],
}

PHASE_OVERRIDE_FNS: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "1": phase1_overrides,
    "2": phase2_overrides,
    "3": phase3_overrides,
    "4": phase4_overrides,
    "5": phase5_overrides,
    "6": phase6_overrides,
    "7a": phase7a_overrides,
    "8": phase8_overrides,
    "9": phase9_overrides,
}


def expected_trial_count_estimate() -> int:
    """Rough expectation; phase 7b and 10 depend on prior winners and are
    not included in this estimate (they each add a fixed per-video count
    documented in BENCHMARK.md)."""
    n_static_per_sample = sum(len(fn()) for fn in (
        phase1_overrides, phase2_overrides, phase3_overrides,
        phase4_overrides, phase5_overrides, phase6_overrides,
        phase7a_overrides, phase8_overrides,
    ))
    # 11 samples for the static phases except phase 9 which is input_1 only.
    static_total = n_static_per_sample * len(SAMPLES_FULL)
    phase9_total = len(phase9_overrides())
    # Phase 7b: 9 iters x 5 top-kernels per sample = 45.
    phase7b_total = 45 * len(SAMPLES_FULL)
    # Phase 10: 5^3 = 125 per sample.
    phase10_total = 125 * len(SAMPLES_FULL)
    return static_total + phase9_total + phase7b_total + phase10_total
