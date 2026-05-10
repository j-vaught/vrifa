#!/usr/bin/env bash
# Single max-intensity run per (video, binary) on this host.
# 11 videos x 3 binaries = 33 runs.
# Max-intensity outputs that all 3 binaries support:
#   --write-mask-pngs --write-overlay-pngs --write-heatmap-pngs
#   --write-videos (mask+overlay+heatmap MP4s)
#   --annotation-formats coco,yolov5,darknet
# Stabilization and pre-delta-blur are NOT enabled (fast-vrifa does not
# implement them yet; default values match across all 3 binaries).
#
# Output: a 33-row CSV at $RESULTS_CSV plus a markdown table on stdout.

set -uo pipefail
REPO_ROOT="${BENCH_REPO_ROOT:-$HOME/bench_vrifa}"
VIDEO_DIR="$REPO_ROOT/data"
OUT_BASE="/tmp/vrifa_3way_bench"
RESULTS_CSV="${RESULTS_CSV:-/tmp/vrifa_3way_results.csv}"

PY="$REPO_ROOT/_dev/reference_impl/vrifa.py"
RS="$REPO_ROOT/vrifa-rs/target/release/vrifa"
FAST="$REPO_ROOT/fast-vrifa-rs/target/release/fast-vrifa"

VIDEOS=(input_1 input_2 input_3 input_4 input_5 input_6 input_7 input_8 input_9 input_10 input_11)

# Common flags supported by all 3 binaries
COMMON_FLAGS="--write-mask-pngs true --write-overlay-pngs true --write-heatmap-pngs true --write-videos --annotation-formats coco,yolov5,darknet"

frame_count() {
    ffprobe -v 0 -count_frames -select_streams v -show_entries stream=nb_read_frames "$1" 2>/dev/null \
        | sed -n 's/^nb_read_frames=//p' | head -1
}

run_one() {
    local binary="$1" sample="$2" cmd="$3"
    local outdir="$OUT_BASE/${binary}/${sample}"
    rm -rf "$outdir"
    mkdir -p "$outdir"
    local start end elapsed
    start=$(date +%s.%N)
    eval "$cmd --output-dir $outdir" >/dev/null 2>&1
    local rc=$?
    end=$(date +%s.%N)
    elapsed=$(python3 -c "print(f'{$end - $start:.2f}')")
    rm -rf "$outdir"
    echo "$rc:$elapsed"
}

mkdir -p "$OUT_BASE"
echo "binary,sample,frames,elapsed_s,fps,exit_code" > "$RESULTS_CSV"

for sample in "${VIDEOS[@]}"; do
    video="$VIDEO_DIR/${sample}.mp4"
    if [[ ! -f "$video" ]]; then
        echo "SKIP $sample (no video)"
        continue
    fi
    frames=$(frame_count "$video")
    echo ""
    echo "=== $sample ($frames frames) ==="

    # python
    res=$(run_one "python" "$sample" "python3 $PY --video-path $video $COMMON_FLAGS")
    rc=${res%%:*}; t=${res##*:}
    fps=$(python3 -c "print(f'{$frames / $t:.1f}' if $t > 0 else '0.0')")
    echo "  python:     ${t}s ($fps fps) rc=$rc"
    echo "python,$sample,$frames,$t,$fps,$rc" >> "$RESULTS_CSV"

    # vrifa-rs
    res=$(run_one "vrifa_rs" "$sample" "$RS --video-path $video $COMMON_FLAGS")
    rc=${res%%:*}; t=${res##*:}
    fps=$(python3 -c "print(f'{$frames / $t:.1f}' if $t > 0 else '0.0')")
    echo "  vrifa-rs:   ${t}s ($fps fps) rc=$rc"
    echo "vrifa_rs,$sample,$frames,$t,$fps,$rc" >> "$RESULTS_CSV"

    # fast-vrifa
    res=$(run_one "fast_vrifa" "$sample" "$FAST --video-path $video $COMMON_FLAGS")
    rc=${res%%:*}; t=${res##*:}
    fps=$(python3 -c "print(f'{$frames / $t:.1f}' if $t > 0 else '0.0')")
    echo "  fast-vrifa: ${t}s ($fps fps) rc=$rc"
    echo "fast_vrifa,$sample,$frames,$t,$fps,$rc" >> "$RESULTS_CSV"
done

echo ""
echo "=== summary table ==="
python3 - "$RESULTS_CSV" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
samples = sorted({r["sample"] for r in rows}, key=lambda s: int(s.split("_")[1]))
binaries = ["python", "vrifa_rs", "fast_vrifa"]

print()
print("| Sample | Frames | Python (s) | vrifa-rs (s) | fast-vrifa (s) | Rust speedup | GPU speedup |")
print("|---|---:|---:|---:|---:|---:|---:|")
totals = {b: 0.0 for b in binaries}
total_frames = 0
for s in samples:
    by_bin = {r["binary"]: r for r in rows if r["sample"] == s}
    if not all(b in by_bin for b in binaries):
        continue
    py = float(by_bin["python"]["elapsed_s"])
    rs = float(by_bin["vrifa_rs"]["elapsed_s"])
    fa = float(by_bin["fast_vrifa"]["elapsed_s"])
    fr = int(by_bin["python"]["frames"])
    rs_x = py / rs if rs > 0 else 0
    fa_x = py / fa if fa > 0 else 0
    print(f"| {s} | {fr} | {py:.2f} | {rs:.2f} | {fa:.2f} | {rs_x:.1f}x | {fa_x:.1f}x |")
    totals["python"] += py
    totals["vrifa_rs"] += rs
    totals["fast_vrifa"] += fa
    total_frames += fr

py = totals["python"]; rs = totals["vrifa_rs"]; fa = totals["fast_vrifa"]
print(f"| **TOTAL** | **{total_frames}** | **{py:.1f}** | **{rs:.1f}** | **{fa:.1f}** | **{py/rs:.1f}x** | **{py/fa:.1f}x** |")
print()
print(f"Aggregate fps:  python {total_frames/py:.1f},  vrifa-rs {total_frames/rs:.1f},  fast-vrifa {total_frames/fa:.1f}")
PYEOF
