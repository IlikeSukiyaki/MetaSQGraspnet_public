#!/usr/bin/env bash
# Collect the verified demo dataset (000, 001, 004, 011) for SQ-v0-Data.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

OUTPUT_DIR="${1:-datasets/supersq_grasp_data/demo}"
TOPDOWN_HEIGHT="${TOPDOWN_HEIGHT:-2.1}"

run_collect() {
  local obj_id="$1"
  python -u run.py --task SQ-v0-Data --num_envs 1 --enable_cameras --headless \
    --save_camera_data \
    --export_profile lite \
    --obj_ids "${obj_id}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_episodes 1 \
    --topdown_height "${TOPDOWN_HEIGHT}"
}

for OBJ in 000 001 004 011; do
  echo "=== Collecting demo episode for object ${OBJ} ==="
  run_collect "${OBJ}"
done

echo "Demo dataset written to ${OUTPUT_DIR}"
