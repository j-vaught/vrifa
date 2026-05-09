#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"

if [[ -z "${PKG_CONFIG_PATH:-}" && -d /opt/homebrew/opt/opencv/lib/pkgconfig ]]; then
  export PKG_CONFIG_PATH="/opt/homebrew/opt/opencv/lib/pkgconfig"
fi

cd "$repo_root/vrifa-rs"
cargo build --release -p vrifa-cli

cd "$repo_root/fast-vrifa-rs"
cargo build --release -p fast-vrifa-cli --features wgpu

run_case() {
  local label="$1"
  local video_path="$2"
  local roi_margin="$3"

  local cpu_out="/tmp/fast_vrifa_${label}_cpu"
  local delegate_out="/tmp/fast_vrifa_${label}_delegate"
  local wgpu_out="/tmp/fast_vrifa_${label}_wgpu"
  rm -rf "$cpu_out" "$delegate_out" "$wgpu_out"

  "$repo_root/vrifa-rs/target/release/vrifa" \
    --video-path "$video_path" \
    --output-dir "$cpu_out" \
    --write-videos \
    --write-mask-pngs true \
    --write-overlay-pngs true \
    --write-heatmap-pngs true \
    --roi-margin "$roi_margin" \
    --annotation-formats coco

  "$repo_root/fast-vrifa-rs/target/release/fast-vrifa" \
    --video-path "$video_path" \
    --output-dir "$delegate_out" \
    --write-videos \
    --write-mask-pngs true \
    --write-overlay-pngs true \
    --write-heatmap-pngs true \
    --roi-margin "$roi_margin" \
    --annotation-formats coco

  "$repo_root/_dev/validation/compare_runs.py" "$cpu_out" "$delegate_out"

  "$repo_root/fast-vrifa-rs/target/release/fast-vrifa" \
    --backend wgpu \
    --video-path "$video_path" \
    --output-dir "$wgpu_out" \
    --write-videos \
    --write-mask-pngs true \
    --write-overlay-pngs true \
    --write-heatmap-pngs true \
    --roi-margin "$roi_margin" \
    --annotation-formats coco

  "$repo_root/_dev/validation/compare_runs.py" "$cpu_out" "$wgpu_out"
}

run_case "input_1" "$repo_root/data/input_1.mp4" "0.15"
run_case "input_2" "$repo_root/data/input_2.mp4" "0.0"
