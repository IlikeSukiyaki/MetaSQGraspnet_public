#!/usr/bin/env bash
# Convert MetaGraspNet IFL objects from textured.urdf -> orbit_obj.usd (physics-ready USD).
#
# Output is written next to each URDF:
#   <models_root>/<id>/orbit_obj.usd
#
# MetaSQGrasp/models/models_ifl is a symlink to DSQNet/models_ifl, so converting
# the corpus once is enough for RL / classical / SQ tasks.
#
# Note: urdf_converter.py was written for Isaac Lab 1.4 + Isaac Sim 4.2. If conversion
# fails on newer stacks, activate the env_isaaclab conda env recommended in README.
#
# Usage:
#   ./tools/convert_models_ifl_to_usd.sh
#   ./tools/convert_models_ifl_to_usd.sh --force
#   ./tools/convert_models_ifl_to_usd.sh --ids 001,004,011
#   ./tools/convert_models_ifl_to_usd.sh --batch          # one Isaac Sim session for all
#   MODELS_ROOT=/path/to/models_ifl ./tools/convert_models_ifl_to_usd.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODELS_ROOT="${MODELS_ROOT:-/home/sustech/桌面/DSQNet/models_ifl}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
FORCE=0
BATCH=0
IDS=""

usage() {
    sed -n '2,18p' "$0" | tail -n +2
    echo
    echo "Options:"
    echo "  --force     Re-convert even if orbit_obj.usd already exists"
    echo "  --batch     Convert all pending URDFs in one urdf_converter.py run"
    echo "  --ids LIST  Comma-separated object ids (e.g. 001,004,011)"
    echo "  -h, --help  Show this help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --batch) BATCH=1; shift ;;
        --ids) IDS="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ ! -d "${MODELS_ROOT}" ]]; then
    echo "ERROR: models root not found: ${MODELS_ROOT}" >&2
    exit 1
fi

if [[ -n "${IDS}" ]]; then
    mapfile -t URDF_PATHS < <(
        IFS=',' read -ra _ids <<< "${IDS}"
        for id in "${_ids[@]}"; do
            id="$(echo "${id}" | xargs)"
            [[ -z "${id}" ]] && continue
            printf '%s\n' "${MODELS_ROOT}/${id}/textured.urdf"
        done
    )
else
    mapfile -t URDF_PATHS < <(find "${MODELS_ROOT}" -mindepth 2 -maxdepth 2 -name 'textured.urdf' | sort)
fi

if [[ ${#URDF_PATHS[@]} -eq 0 ]]; then
    echo "ERROR: no textured.urdf files found under ${MODELS_ROOT}" >&2
    exit 1
fi

PENDING=()
SKIPPED=0
for urdf in "${URDF_PATHS[@]}"; do
    if [[ ! -f "${urdf}" ]]; then
        echo "WARN: missing URDF, skipping: ${urdf}" >&2
        continue
    fi
    usd="${urdf%/textured.urdf}/orbit_obj.usd"
    if [[ ${FORCE} -eq 0 && -f "${usd}" ]]; then
        echo "[skip] $(basename "$(dirname "${urdf}")") -> orbit_obj.usd exists"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    PENDING+=("${urdf}")
done

echo "Models root : ${MODELS_ROOT}"
echo "Project root: ${PROJECT_ROOT}"
echo "Conda env   : ${CONDA_ENV}"
echo "URDF total  : ${#URDF_PATHS[@]}"
echo "Skipped     : ${SKIPPED}"
echo "Pending     : ${#PENDING[@]}"

if [[ ${#PENDING[@]} -eq 0 ]]; then
    echo "Nothing to convert."
    exit 0
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/logs/urdf_conversion/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "${LOG_DIR}"

convert_one() {
    local urdf="$1"
    local obj_id
    obj_id="$(basename "$(dirname "${urdf}")")"
    local log_file="${LOG_DIR}/${obj_id}.log"

    echo "[convert] ${obj_id}"
    if python urdf_converter.py --input "${urdf}" --headless >"${log_file}" 2>&1; then
        if [[ -f "${urdf%/textured.urdf}/orbit_obj.usd" ]]; then
            echo "[ok] ${obj_id}"
            return 0
        fi
        echo "[fail] ${obj_id}: converter exited 0 but orbit_obj.usd missing (see ${log_file})" >&2
        return 1
    fi
    echo "[fail] ${obj_id}: see ${log_file}" >&2
    return 1
}

OK=0
FAIL=0

if [[ ${BATCH} -eq 1 ]]; then
    BATCH_GLOB="${MODELS_ROOT}/{" 
    first=1
    for urdf in "${PENDING[@]}"; do
        obj_id="$(basename "$(dirname "${urdf}")")"
        if [[ ${first} -eq 1 ]]; then
            BATCH_GLOB+="${obj_id}"
            first=0
        else
            BATCH_GLOB+=",${obj_id}"
        fi
    done
    BATCH_GLOB+="}/textured.urdf"

    echo "[batch] ${BATCH_GLOB}"
    BATCH_LOG="${LOG_DIR}/batch.log"
    if python urdf_converter.py --input "${BATCH_GLOB}" --headless | tee "${BATCH_LOG}"; then
        for urdf in "${PENDING[@]}"; do
            if [[ -f "${urdf%/textured.urdf}/orbit_obj.usd" ]]; then
                OK=$((OK + 1))
            else
                echo "[fail] $(basename "$(dirname "${urdf}")") missing after batch run" >&2
                FAIL=$((FAIL + 1))
            fi
        done
    else
        echo "Batch conversion failed. See ${BATCH_LOG}" >&2
        exit 1
    fi
else
    for urdf in "${PENDING[@]}"; do
        if convert_one "${urdf}"; then
            OK=$((OK + 1))
        else
            FAIL=$((FAIL + 1))
        fi
    done
fi

echo
echo "Done."
echo "  converted : ${OK}"
echo "  failed    : ${FAIL}"
echo "  skipped   : ${SKIPPED}"
echo "  logs      : ${LOG_DIR}"

[[ ${FAIL} -eq 0 ]]
