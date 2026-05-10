"""Constants and integrated defaults for the ablation v2 harness.

The integrated configuration mirrors what the vrifa binary ships with --
held fixed across every experiment in the paper unless a phase explicitly
varies a knob. Phase 1's per-trial config starts from this dict and
overrides only the four knobs Phase 1 sweeps; later phases start from
their video's prior-stage winner.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VIDEOS_DIR = REPO_ROOT / "data" / "ablation_data"
DEFAULT_LABELS = DEFAULT_VIDEOS_DIR / "labels.json"
DEFAULT_BINARY = REPO_ROOT / "vrifa-rs" / "target" / "release" / "vrifa"
DEFAULT_STATE_DIR = Path("/tmp/vrifa_ablation")
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "ablation"
DEFAULT_ROI_MASKS_DIR = REPO_ROOT / "data" / "roi_masks"

PHASE_ORDER = ["1", "2", "3", "4", "5", "6", "7a", "7b", "8", "10", "9"]
SAMPLES_FULL = [f"input_{i}" for i in range(1, 12)]
SAMPLE_STABILIZATION = "input_1"

# Per-sample ROI handling.
#
# input_1 uses a polygon-shaped ROI imported from the labeled boundary
# frame (data/roi_masks/input_1.png), passed via --roi-mask. The
# imported mask is mutually exclusive with --roi-margin*.
#
# input_2..input_11 are pre-cropped to roughly the laminate region in
# their source videos, so the ablation runs them with --roi-margin 0
# (full frame).
SAMPLES_WITH_ROI_MASK = {"input_1"}
ROI_MARGIN_BY_SAMPLE = {sample: 0.0 for sample in SAMPLES_FULL}


def roi_mask_path_for(sample: str, masks_dir: Path = DEFAULT_ROI_MASKS_DIR) -> Path | None:
    """Return the per-sample ROI mask PNG path, or None if the sample
    uses the rectangular --roi-margin form."""
    if sample not in SAMPLES_WITH_ROI_MASK:
        return None
    return masks_dir / f"{sample}.png"

# Integrated configuration defaults. These are what `vrifa --help`
# reports as the binary's defaults; the ablation reports what changing
# each row to a per-video best does to IoU.
INTEGRATED_DEFAULTS = {
    "colorspace": "CIELAB",
    "channel_weights": "1,0,0",  # CIELAB default = lightness-only
    "darken_only": True,
    "peak_reference": True,
    "ref_mode": ("first",),
    "ref_running_alpha": 0.05,
    "dynamic_target_fraction": 0.2,
    "dynamic_lag_scale": 1.0,
    "pre_delta_blur": "none",
    "blur": "gaussian:9",
    "threshold": "otsu",
    "threshold_offset": -30.0,
    "morph_kernel": 13,
    "morph_shape": "ellipse",
    "morph_close_iterations": 1,
    "morph_open_iterations": 1,
    "min_area": 400,
    "lock_frames": 3,
    "camera_stable": False,
    "motion_model": "affine",
    "motion_per_frame_threshold": 1.5,
    "cumulative_motion_threshold": 3.0,
    "peak_on_shift": "reset",
}

# Output flags shared by every trial: mask PNGs only, no overlays /
# heatmaps / videos / annotations. The mask PNGs are the only artifact
# agreement.py reads; everything else is wasted IO for the ablation.
OUTPUT_FLAGS = {
    "write_mask_pngs": "true",
    "write_overlay_pngs": "false",
    "write_heatmap_pngs": "false",
    # fast-vrifa default backend is "delegate" which forwards to vrifa-rs
    # CPU. Force CUDA when running through fast-vrifa. vrifa-rs ignores
    # unknown flags? -- it doesn't. So this is fast-vrifa-only.
    "backend": "cuda",
}
