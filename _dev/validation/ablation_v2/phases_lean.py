"""Lean override grids for emergency-budget ablation runs.

Cuts the Option-C grids back to a minimum sufficient set so the run
finishes in ~2 hours after Phase 1 instead of ~30 hours. Drops every
sweep value whose marginal information value is low: extreme morph
kernels, dense threshold-offset sweeps, esoteric blur kinds, dense
joint perturbations. Keeps every NAMED MODE (so the per-mode comparison
in the paper still shows up) but with a sparse value grid.

Usage from run.py:
  if args.lean:
    from . import phases_lean as P
  else:
    from . import phases as P
"""

from __future__ import annotations

from itertools import product
from typing import Any


# Phase 1 is unchanged; it's already running.

# Phase 2 lean: 4 weight combos per colorspace, single darken setting.
PHASE2_WEIGHTS_LEAN = {
    "CIELAB": ["1,0,0", "1,1,1", "0,1,1", "1,0.5,0.5"],
    "RGB": ["1,1,1", "0.299,0.587,0.114", "1,0,0", "0,0,1"],
    "HSV": ["0,0,1", "0,1,1", "1,1,1"],
    "GRAYSCALE": ["1"],
}


def phase2_overrides() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for cs, weights_list in PHASE2_WEIGHTS_LEAN.items():
        for w in weights_list:
            grid.append({
                "colorspace": cs,
                "channel_weights": w,
                "darken_only": True,
            })
    return grid  # 4+4+3+1 = 12 per video × 11 = 132 trials


# Phase 3 lean: just the named modes at default per-mode params.
def phase3_overrides() -> list[dict[str, Any]]:
    return [
        {"ref_mode": ("first",)},
        {"ref_mode": ("running",), "ref_running_alpha": 0.05},
        {"ref_mode": ("prev", 3)},
        {"ref_mode": ("prev", 10)},
        {"ref_mode": ("dynamic",), "dynamic_target_fraction": 0.2,
         "dynamic_lag_scale": 1.0},
    ]  # 5 per video × 11 = 55 trials


# Phase 4 lean: each named mode at a representative offset/value.
def phase4_overrides() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for off in (-50, -30, -10, 10):
        grid.append({"threshold": "otsu", "threshold_offset": float(off)})
    for off in (-20, 0, 20):
        grid.append({"threshold": "triangle", "threshold_offset": float(off)})
    for v in (50, 90, 130):
        grid.append({"threshold": f"manual:{v}"})
    for p in (60, 80, 95):
        grid.append({"threshold": f"percentile:{p}"})
    for b, c in product((11, 31), (-5, 5)):
        grid.append({"threshold": f"adaptive-mean:{b}:{c}"})
    for b, c in product((11, 31), (-5, 5)):
        grid.append({"threshold": f"adaptive-gaussian:{b}:{c}"})
    return grid  # 4+3+3+3+4+4 = 21 per video × 11 = 231 trials


# Phase 5 lean: none + 3 sizes per kind, all 5 kinds.
def phase5_overrides() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = [{"pre_delta_blur": "none"}]
    for kind in ("flat", "gaussian", "triangle"):
        for s in (3, 7, 11):
            grid.append({"pre_delta_blur": f"{kind}:{s}"})
    grid.append({"pre_delta_blur": "median:5"})
    grid.append({"pre_delta_blur": "bilateral:9"})
    return grid  # 1 + 9 + 1 + 1 = 12 per video × 11 = 132 trials


# Phase 6 lean: same shape as 5, slightly more sizes for the post-delta blur
# since that's the integrated configuration's primary blur knob.
def phase6_overrides() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = [{"blur": "none"}]
    for kind in ("flat", "gaussian", "triangle"):
        for s in (3, 7, 11, 15):
            grid.append({"blur": f"{kind}:{s}"})
    grid.append({"blur": "median:5"})
    grid.append({"blur": "bilateral:9"})
    return grid  # 1 + 12 + 1 + 1 = 15 per video × 11 = 165 trials


# Phase 7a lean: kernel × shape but tighter.
PHASE7A_KERNELS_LEAN = (5, 9, 13, 21, 31, 51)
PHASE7A_SHAPES_LEAN = ("ellipse", "rect", "cross")


def phase7a_overrides() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for kernel, shape in product(PHASE7A_KERNELS_LEAN, PHASE7A_SHAPES_LEAN):
        grid.append({
            "morph_kernel": kernel,
            "morph_shape": shape,
            "morph_close_iterations": 1,
            "morph_open_iterations": 1,
        })
    return grid  # 6 × 3 = 18 per video × 11 = 198 trials


# Phase 7b lean: smaller iter grid × fewer kernels.
def phase7b_overrides_for_sample(top_kernels: list[int], shape: str) -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    # Cap to top 3 kernels and only iterations 1, 2, 3 on close+open separately.
    for kernel in top_kernels[:3]:
        for ci in (1, 2):
            for oi in (1, 2):
                grid.append({
                    "morph_kernel": kernel,
                    "morph_shape": shape,
                    "morph_close_iterations": ci,
                    "morph_open_iterations": oi,
                })
    return grid  # 3 × 4 = 12 per video × 11 = 132 trials


# Phase 8 lean: coarser lock sweep.
PHASE8_LOCKS_LEAN = (0, 1, 3, 5, 10, 30)


def phase8_overrides() -> list[dict[str, Any]]:
    return [{"lock_frames": lf} for lf in PHASE8_LOCKS_LEAN]
    # 6 per video × 11 = 66 trials


# Phase 10 lean: 3^3 instead of 5^3 around chained best.
def phase10_overrides_for_sample(prior_winner: dict[str, Any]) -> list[dict[str, Any]]:
    from . import phases as P
    base_offset = float(prior_winner.get("threshold_offset", -30.0))
    base_kernel = int(prior_winner.get("morph_kernel", 13))
    base_lock = int(prior_winner.get("lock_frames", 3))
    grid: list[dict[str, Any]] = []
    for off_step, kern_step, lock_step in product(
        (-10.0, 0.0, 10.0),
        (-1, 0, 1),
        (-1, 0, 1),
    ):
        new_offset = base_offset + off_step
        new_kernel = P._ladder_step(base_kernel, P.PHASE10_KERNEL_LADDER, kern_step)
        new_lock = max(0, base_lock + lock_step)
        grid.append({
            "threshold_offset": new_offset,
            "morph_kernel": new_kernel,
            "lock_frames": new_lock,
        })
    return grid  # 3^3 = 27 per video × 11 = 297 trials


# Phase 9 lean: smaller stabilization grid (input_1 only).
def phase9_overrides() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = [{"camera_stable": False}]
    for model, per_frame, cumulative in product(
        ("translation", "affine"),
        (1.0, 2.5),
        (1.0, 3.0, 10.0),
    ):
        grid.append({
            "camera_stable": True,
            "motion_model": model,
            "motion_per_frame_threshold": per_frame,
            "cumulative_motion_threshold": cumulative,
        })
    return grid  # 1 + 2*2*3 = 13 trials, input_1 only


# Phase 1 unchanged; if --start-phase 2 is used the existing Phase 1
# state is reused. We still re-export the old phase 1 fn as a no-op fallback.
def phase1_overrides() -> list[dict[str, Any]]:
    from . import phases as P
    return P.phase1_overrides()
