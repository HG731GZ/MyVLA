#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LIBERO_PRO_ROOT="${LIBERO_PRO_ROOT:-${PROJECT_ROOT}/third_party/LIBERO-PRO}"
LIBERO_PRO_DATA_ROOT="${LIBERO_PRO_DATA_ROOT:-${PROJECT_ROOT}/third_party/LIBERO-PRO-data}"
LIBERO_PRO_REPO="${LIBERO_PRO_REPO:-https://github.com/Zxy-MLlab/LIBERO-PRO.git}"
LIBERO_PRO_DATASET="${LIBERO_PRO_DATASET:-zhouxueyang/LIBERO-Pro}"

usage() {
    cat <<EOF
Usage: bash ${SCRIPT_DIR}/setup_libero_pro.sh [options]

Options:
  --libero_pro_root PATH       LIBERO-PRO source checkout
  --libero_pro_data_root PATH  Downloaded BDDL/init data
  -h, --help                   Show this help

The same paths can be set with LIBERO_PRO_ROOT and LIBERO_PRO_DATA_ROOT.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --libero_pro_root)
            if [[ $# -lt 2 ]]; then
                echo "--libero_pro_root requires a path." >&2
                exit 2
            fi
            LIBERO_PRO_ROOT="$2"
            shift 2
            ;;
        --libero_pro_data_root|--data_root)
            if [[ $# -lt 2 ]]; then
                echo "$1 requires a path." >&2
                exit 2
            fi
            LIBERO_PRO_DATA_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

mkdir -p "$(dirname "${LIBERO_PRO_ROOT}")" "$(dirname "${LIBERO_PRO_DATA_ROOT}")"

if [[ -e "${LIBERO_PRO_ROOT}" ]]; then
    if [[ ! -f "${LIBERO_PRO_ROOT}/libero/libero/benchmark/__init__.py" ]]; then
        echo "Existing path is not a LIBERO-PRO checkout: ${LIBERO_PRO_ROOT}" >&2
        exit 1
    fi
    echo "Using existing LIBERO-PRO source: ${LIBERO_PRO_ROOT}"
else
    echo "Cloning official LIBERO-PRO source into ${LIBERO_PRO_ROOT}"
    git clone --depth 1 "${LIBERO_PRO_REPO}" "${LIBERO_PRO_ROOT}"
fi

if [[ -d "${LIBERO_PRO_DATA_ROOT}/bddl_files/libero_goal_lan" &&
      -d "${LIBERO_PRO_DATA_ROOT}/init_files/libero_goal_lan" ]]; then
    echo "Using existing LIBERO-PRO data: ${LIBERO_PRO_DATA_ROOT}"
else
    mkdir -p "${LIBERO_PRO_DATA_ROOT}"
    echo "Downloading official LIBERO-PRO data into ${LIBERO_PRO_DATA_ROOT}"
    if command -v hf >/dev/null 2>&1; then
        hf download "${LIBERO_PRO_DATASET}" \
            --repo-type dataset \
            --local-dir "${LIBERO_PRO_DATA_ROOT}"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "${LIBERO_PRO_DATASET}" \
            --repo-type dataset \
            --local-dir "${LIBERO_PRO_DATA_ROOT}"
    else
        echo "Neither 'hf' nor 'huggingface-cli' is installed." >&2
        echo "Install it with: pip install -U huggingface_hub" >&2
        exit 1
    fi
fi

for suite in libero_goal libero_spatial libero_10 libero_object; do
    for suffix in object swap lan task; do
        if [[ ! -d "${LIBERO_PRO_DATA_ROOT}/bddl_files/${suite}_${suffix}" ||
              ! -d "${LIBERO_PRO_DATA_ROOT}/init_files/${suite}_${suffix}" ]]; then
            echo "Incomplete LIBERO-PRO data: ${suite}_${suffix}" >&2
            exit 1
        fi
    done
done

echo
echo "LIBERO-PRO source and pre-generated data are ready."
echo "Source: ${LIBERO_PRO_ROOT}"
echo "Data:   ${LIBERO_PRO_DATA_ROOT}"
echo
echo "Environment perturbation data is generated separately:"
echo "  bash ${SCRIPT_DIR}/prepare_environment_suite.sh libero_object"
