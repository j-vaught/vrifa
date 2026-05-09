#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


STAGES = [
    "frame_converted",
    "delta",
    "delta_blur",
    "delta_norm",
    "binary",
    "mask",
    "overlay",
    "heatmap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("python_dump", type=Path)
    parser.add_argument("rust_dump", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Report the first stage whose max-abs-diff exceeds this threshold.",
    )
    return parser.parse_args()


def max_abs_diff(py_array: np.ndarray, rs_array: np.ndarray) -> float:
    if py_array.shape != rs_array.shape:
        raise ValueError(f"shape mismatch: python={py_array.shape} rust={rs_array.shape}")
    return float(np.max(np.abs(py_array.astype(np.float64) - rs_array.astype(np.float64))))


def main() -> int:
    args = parse_args()
    frame_dirs = sorted(path for path in args.python_dump.iterdir() if path.is_dir())
    if not frame_dirs:
        raise RuntimeError(f"no frame dumps found in {args.python_dump}")

    for py_frame_dir in frame_dirs:
        rs_frame_dir = args.rust_dump / py_frame_dir.name
        if not rs_frame_dir.exists():
            raise RuntimeError(f"missing Rust frame dump {rs_frame_dir}")
        print(py_frame_dir.name)
        first_divergence = None
        for stage in STAGES:
            py_array = np.load(py_frame_dir / f"{stage}.npy")
            rs_array = np.load(rs_frame_dir / f"{stage}.npy")
            diff = max_abs_diff(py_array, rs_array)
            print(f"  {stage:16s} max_abs_diff={diff:g}")
            if first_divergence is None and diff > args.threshold:
                first_divergence = (stage, diff)
        if first_divergence is None:
            print(f"  first_divergence  none (all <= {args.threshold:g})")
        else:
            stage, diff = first_divergence
            print(f"  first_divergence  {stage} ({diff:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
