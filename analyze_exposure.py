#!/usr/bin/env python3
"""
Automated Exposure Compensation Correction for VARTM Process Monitoring

Two-step correction method:
1. LAB L*-channel correction to eliminate brightness discontinuities
2. RGB ratio normalization to preserve original color relationships
"""

import cv2
import numpy as np
from pathlib import Path


def find_stable_region(video_path, region_size=100):
    """Find the image region with minimum temporal variance."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    h, w = frames[0].shape
    frames = np.array(frames)

    # Test corner regions
    regions = {
        'top_left': (0, 0),
        'top_right': (0, w - region_size),
        'bottom_left': (h - region_size, 0),
        'bottom_right': (h - region_size, w - region_size),
    }

    min_var = float('inf')
    best_region = None

    for name, (y, x) in regions.items():
        region_means = [f[y:y+region_size, x:x+region_size].mean() for f in frames]
        var = np.var(region_means)
        if var < min_var:
            min_var = var
            best_region = (y, x, region_size)

    return best_region


def extract_luminosity(video_path, region):
    """Extract mean luminosity from stable region for each frame."""
    y, x, size = region
    cap = cv2.VideoCapture(str(video_path))
    luminosity = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L = lab[y:y+size, x:x+size, 0]
        luminosity.append(L.mean())

    cap.release()
    return np.array(luminosity)


def detect_exposure_events(luminosity, threshold_std=3.0, min_jump=2.0):
    """Detect exposure compensation events using derivative analysis."""
    # Compute frame-to-frame derivative
    derivative = np.diff(luminosity)

    # Detect positive jumps (exposure increases)
    mean_d = np.mean(derivative)
    std_d = np.std(derivative)

    # Use both statistical threshold and absolute minimum jump
    threshold = max(mean_d + threshold_std * std_d, min_jump)

    events = []
    i = 0
    while i < len(derivative):
        if derivative[i] > threshold:
            # Find the frame with maximum jump in this cluster
            cluster_start = i
            while i < len(derivative) and derivative[i] > min_jump / 2:
                i += 1
            cluster_end = i
            max_idx = cluster_start + np.argmax(derivative[cluster_start:cluster_end])
            events.append(max_idx + 1)  # +1 because derivative shifts index
        else:
            i += 1

    return events


def compute_correction_factors(luminosity, events):
    """Compute multiplicative correction factors for each event."""
    factors = {}
    for event in events:
        L_before = luminosity[event - 1]
        L_after = luminosity[event]
        gamma = L_before / L_after
        factors[event] = gamma
    return factors


def apply_two_step_correction(video_path, output_path, events, factors):
    """Apply two-step correction: LAB L* scaling + RGB ratio normalization."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Read all frames first to get original color ratios
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    original_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        original_frames.append(frame.astype(np.float32))

    total_frames = len(original_frames)

    # Process each frame
    for frame_idx in range(total_frames):
        frame = original_frames[frame_idx].copy()

        # Compute cumulative correction factor
        cumulative_gamma = 1.0
        for event in events:
            if frame_idx >= event:
                cumulative_gamma *= factors[event]

        if cumulative_gamma != 1.0:
            # Step 1: LAB L* correction
            lab = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            lab[:, :, 0] = np.clip(lab[:, :, 0] * cumulative_gamma, 0, 255)
            corrected = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

            # Step 2: RGB ratio normalization
            # Get original R/G and B/G ratios
            orig_R = frame[:, :, 2]
            orig_G = frame[:, :, 1]
            orig_B = frame[:, :, 0]

            # Compute mean ratios in stable region (avoid division issues)
            mask = orig_G > 10  # Avoid dark pixels
            if mask.sum() > 0:
                orig_rg = np.mean(orig_R[mask] / orig_G[mask])
                orig_bg = np.mean(orig_B[mask] / orig_G[mask])

                # Get corrected ratios
                corr_R = corrected[:, :, 2]
                corr_G = corrected[:, :, 1]
                corr_B = corrected[:, :, 0]

                corr_rg = np.mean(corr_R[mask] / corr_G[mask])
                corr_bg = np.mean(corr_B[mask] / corr_G[mask])

                # Compute normalization factors
                alpha_r = orig_rg / corr_rg if corr_rg > 0 else 1.0
                alpha_b = orig_bg / corr_bg if corr_bg > 0 else 1.0

                # Apply normalization
                corrected[:, :, 2] = np.clip(corr_R * alpha_r, 0, 255)
                corrected[:, :, 0] = np.clip(corr_B * alpha_b, 0, 255)

            frame = corrected

        out.write(frame.astype(np.uint8))

        if (frame_idx + 1) % 20 == 0:
            print(f"Processed frame {frame_idx + 1}/{total_frames}")

    cap.release()
    out.release()
    print(f"Output saved to {output_path}")


def main():
    video_path = Path("/Users/jvaught/Downloads/Code/vrifa/input_4.mp4")
    output_path = Path("/Users/jvaught/Downloads/Code/vrifa/input_4_corrected_final.mp4")

    print("Finding stable region...")
    region = find_stable_region(video_path)
    print(f"Selected region: y={region[0]}, x={region[1]}, size={region[2]}")

    print("Extracting luminosity...")
    luminosity = extract_luminosity(video_path, region)
    print(f"Extracted {len(luminosity)} frames")

    print("Detecting exposure events...")
    events = detect_exposure_events(luminosity, threshold_std=3.0)
    print(f"Detected events at frames: {events}")

    print("Computing correction factors...")
    factors = compute_correction_factors(luminosity, events)
    for event, gamma in factors.items():
        print(f"  Frame {event}: gamma = {gamma:.4f}")

    cumulative = 1.0
    for event in sorted(events):
        cumulative *= factors[event]
    print(f"Cumulative correction factor: {cumulative:.4f}")

    print("Applying two-step correction...")
    apply_two_step_correction(video_path, output_path, events, factors)

    print("Done!")


if __name__ == "__main__":
    main()
