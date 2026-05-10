#!/usr/bin/env bash
# 3-way wall-clock benchmark on input_1.mp4: Python reference vs vrifa-rs (CPU) vs fast-vrifa (CUDA/wgpu).
# 3 runs each, mask PNGs only, defaults otherwise. Prints a markdown table and writes JSON to stdout's last block.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VIDEO="$REPO_ROOT/data/input_1.mp4"
OUT_BASE="/tmp/vrifa_bench"

PY="$REPO_ROOT/_dev/reference_impl/vrifa.py"
RS="$REPO_ROOT/vrifa-rs/target/release/vrifa"
FAST="$REPO_ROOT/fast-vrifa-rs/target/release/fast-vrifa"

run_one() {
    local label="$1" cmd="$2" outdir="$3"
    rm -rf "$outdir"
    mkdir -p "$outdir"
    local start end elapsed
    start=$(python3 -c "import time; print(time.monotonic())")
    eval "$cmd" >/dev/null 2>&1
    end=$(python3 -c "import time; print(time.monotonic())")
    elapsed=$(python3 -c "print(f'{$end - $start:.2f}')")
    echo "  $label: ${elapsed}s"
    rm -rf "$outdir"
    echo "$elapsed"
}

bench() {
    local label="$1" cmd_template="$2"
    local outdir="$OUT_BASE/$label"
    local times=()
    for i in 1 2 3; do
        local cmd
        cmd=$(printf "$cmd_template" "$outdir")
        local t
        t=$(run_one "${label}_run${i}" "$cmd" "$outdir" | tail -1)
        times+=("$t")
    done
    python3 -c "import statistics; ts=[float(x) for x in '${times[*]}'.split()]; print(f'{statistics.median(ts):.2f}')"
}

echo "=== Python reference ==="
PY_MEDIAN=$(bench "python" "python3 $PY --video-path $VIDEO --output-dir %s --write-mask-pngs true --write-overlay-pngs false --write-heatmap-pngs false --no-write-mask-video --no-write-overlay-video --no-write-heatmap-video 2>/dev/null")

echo ""
echo "=== vrifa-rs (CPU Rust) ==="
RS_MEDIAN=$(bench "vrifa_rs" "$RS --video-path $VIDEO --output-dir %s --write-mask-pngs true --write-overlay-pngs false --write-heatmap-pngs false")

echo ""
echo "=== fast-vrifa (GPU) ==="
FAST_MEDIAN=$(bench "fast_vrifa" "$FAST --video-path $VIDEO --output-dir %s --write-mask-pngs true --write-overlay-pngs false --write-heatmap-pngs false")

echo ""
echo "=== summary (median of 3 runs, input_1.mp4 = 706 frames) ==="
echo ""
echo "| Binary | Median wall-clock (s) | fps | Speedup vs Python |"
echo "|---|---:|---:|---:|"
python3 - <<EOF
py = $PY_MEDIAN
rs = $RS_MEDIAN
fa = $FAST_MEDIAN
fr = 706
print(f"| Python reference | {py:.2f} | {fr/py:.1f} | 1.00x |")
print(f"| vrifa-rs (CPU)   | {rs:.2f} | {fr/rs:.1f} | {py/rs:.2f}x |")
print(f"| fast-vrifa (GPU) | {fa:.2f} | {fr/fa:.1f} | {py/fa:.2f}x |")
EOF
