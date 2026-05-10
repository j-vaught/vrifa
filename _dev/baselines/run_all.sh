#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FRAME_SELECTION="${FRAME_SELECTION:-all}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"

run_baseline() {
  local script_name="$1"
  local output_root="$2"

  for i in $(seq 1 11); do
    local sample="input_${i}"
    "${PYTHON_BIN}" "${ROOT_DIR}/_dev/baselines/${script_name}" \
      --video "${DATA_DIR}/${sample}.mp4" \
      --out "${ROOT_DIR}/${output_root}/${sample}" \
      --frame-selection "${FRAME_SELECTION}"
  done
}

run_baseline "lekanidis_vosniakos_2020.py" "outputs_baseline_lekanidis_vosniakos_2020"
"${PYTHON_BIN}" "${ROOT_DIR}/_dev/validation/agreement.py" \
  --runs-dir "${ROOT_DIR}/outputs_baseline_lekanidis_vosniakos_2020" \
  --output "${ROOT_DIR}/data/agreement_metrics_lekanidis.json"

run_baseline "almazan_lazaro_2022.py" "outputs_baseline_almazan_lazaro_2022"
"${PYTHON_BIN}" "${ROOT_DIR}/_dev/validation/agreement.py" \
  --runs-dir "${ROOT_DIR}/outputs_baseline_almazan_lazaro_2022" \
  --output "${ROOT_DIR}/data/agreement_metrics_almazan.json"
