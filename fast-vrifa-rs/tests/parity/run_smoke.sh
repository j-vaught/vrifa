#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../../.." && pwd)"

if [[ -z "${PKG_CONFIG_PATH:-}" && -d /opt/homebrew/opt/opencv/lib/pkgconfig ]]; then
  export PKG_CONFIG_PATH="/opt/homebrew/opt/opencv/lib/pkgconfig"
fi

cd "$repo_root/vrifa-rs"
cargo build --release -p vrifa-cli

cd "$repo_root/fast-vrifa-rs"
cargo build --release -p fast-vrifa-cli

cpu_out="/tmp/fast_vrifa_cpu_smoke"
gpu_out="/tmp/fast_vrifa_delegate_smoke"
rm -rf "$cpu_out" "$gpu_out"

"$repo_root/vrifa-rs/target/release/vrifa" \
  --video-path "$repo_root/data/input_2.mp4" \
  --output-dir "$cpu_out" \
  --write-videos \
  --write-mask-pngs true \
  --write-overlay-pngs true \
  --write-heatmap-pngs true \
  --roi-margin 0.0 \
  --annotation-formats coco

"$repo_root/fast-vrifa-rs/target/release/fast-vrifa" \
  --video-path "$repo_root/data/input_2.mp4" \
  --output-dir "$gpu_out" \
  --write-videos \
  --write-mask-pngs true \
  --write-overlay-pngs true \
  --write-heatmap-pngs true \
  --roi-margin 0.0 \
  --annotation-formats coco

"$repo_root/_dev/validation/compare_runs.py" "$cpu_out" "$gpu_out"
