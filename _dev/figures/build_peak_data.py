"""Compute per-pixel L* traces and running peak P for the peak-brightness
figure (fig:peak).

Picks four pixels inside the ROI of input_1.mp4 with front-arrival times
spread across the run (early / mid-early / mid-late / late) and writes
their full 706-frame traces and metadata to CSV. The Typst figure then
reads these and plots all four traces on a shared axis so the reader
sees the peak-brightness logic working on pixels that wet at different
points.

Outputs:
  paper/typst/figures/peak_data/traces.csv   long-format trace rows
  paper/typst/figures/peak_data/pixels.csv   pixel metadata
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

REPO = Path("/Users/user/Downloads/vrifa")
VIDEO = REPO / "data" / "input_1.mp4"
ROI_MASK_PATH = REPO / "data" / "roi_masks" / "input_1.png"
OUT_DIR = REPO / "paper" / "typst" / "figures" / "peak_data"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Sample pixels on a coarse grid to keep the candidate set manageable.
GRID = 50

# Front-arrival threshold: drop > this many uint8 L* units below peak.
ARRIVAL_DROP = 30

# Number of representative pixels to track in the figure.
N_PIXELS = 4


def load_roi(h, w):
    raw = cv2.imread(str(ROI_MASK_PATH), cv2.IMREAD_GRAYSCALE)
    if raw.shape != (h, w):
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    return raw > 127


def main():
    cap = cv2.VideoCapture(str(VIDEO))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {n_frames} frames, {width}x{height}")

    roi = load_roi(height, width)

    # Build the candidate-pixel grid inside the ROI.
    ys = np.arange(GRID, height - GRID, GRID)
    xs = np.arange(GRID, width  - GRID, GRID)
    candidates = [(y, x) for y in ys for x in xs if roi[y, x]]
    print(f"{len(candidates)} candidate pixels inside ROI grid")

    # Extract L* values at every candidate pixel, every frame.
    traces = np.zeros((n_frames, len(candidates)), dtype=np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for t in range(n_frames):
        ok, f = cap.read()
        if not ok:
            print(f"warning: frame {t} failed to read")
            n_frames = t
            traces = traces[:t]
            break
        lstar = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[..., 0]
        for i, (y, x) in enumerate(candidates):
            traces[t, i] = lstar[y, x]
    cap.release()
    print(f"trace array shape: {traces.shape}")

    # Per-pixel running peak and arrival frame.
    traces_i = traces.astype(np.int16)
    peaks = np.maximum.accumulate(traces_i, axis=0)
    drop_mask = traces_i < peaks - ARRIVAL_DROP
    # Arrival frame is the first row that drops below peak; pixels that
    # never wet are given arrival = n_frames so they sort to the end.
    arrival = np.where(
        drop_mask.any(axis=0),
        drop_mask.argmax(axis=0),
        n_frames,
    )

    # Pick pixels whose arrival frames are spread across the run rather
    # than concentrated at quantile boundaries. With many pixels wetting
    # by the 50% fill point, naive quantile sampling clusters in the
    # first half; explicit target frames give one early, one mid-early,
    # one mid-late, and one late-wetting pixel.
    wetted_idx = np.flatnonzero(arrival < n_frames)
    print(f"{len(wetted_idx)} of {len(candidates)} candidate pixels wetted")
    target_frames = [80, 230, 400, 600]
    selected = []
    for tf in target_frames:
        i = int(np.argmin(np.abs(arrival[wetted_idx] - tf)))
        selected.append(int(wetted_idx[i]))

    print("selected pixels (x, y, arrival frame):")
    for s in selected:
        y, x = candidates[s]
        print(f"  ({x:4d}, {y:4d})  arrival frame {int(arrival[s])}")

    # Write traces.csv: frame, L1, P1, L2, P2, L3, P3, L4, P4
    header = ["frame"]
    for i in range(N_PIXELS):
        header += [f"L{i+1}", f"P{i+1}"]
    rows = [",".join(header)]
    for t in range(n_frames):
        cells = [str(t)]
        for s in selected:
            cells.append(str(int(traces[t, s])))
            cells.append(str(int(peaks[t, s])))
        rows.append(",".join(cells))
    (OUT_DIR / "traces.csv").write_text("\n".join(rows) + "\n")

    # Write pixels.csv: pixel index, x, y, arrival_frame, peak_value
    meta = ["pixel,x,y,arrival_frame,peak_value"]
    for i, s in enumerate(selected, start=1):
        y, x = candidates[s]
        meta.append(f"{i},{x},{y},{int(arrival[s])},{int(peaks[-1, s])}")
    (OUT_DIR / "pixels.csv").write_text("\n".join(meta) + "\n")

    print(f"wrote {OUT_DIR}/traces.csv and pixels.csv")


if __name__ == "__main__":
    main()
