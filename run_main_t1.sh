#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SCRIPT_START_EPOCH="$(date +%s)"
SCRIPT_START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"

print_runtime_summary() {
  local script_end_epoch script_end_time elapsed hours minutes seconds
  script_end_epoch="$(date +%s)"
  script_end_time="$(date '+%Y-%m-%d %H:%M:%S')"
  elapsed=$((script_end_epoch - SCRIPT_START_EPOCH))
  hours=$((elapsed / 3600))
  minutes=$(((elapsed % 3600) / 60))
  seconds=$((elapsed % 60))
  printf 'Program start time: %s | Program end time: %s | Total elapsed time: %02d:%02d:%02d\n' \
    "${SCRIPT_START_TIME}" "${script_end_time}" "${hours}" "${minutes}" "${seconds}"
}

trap print_runtime_summary EXIT

DCT_N="${DCT_N:-20}"
ADD_NAME="${ADD_NAME:-}"
SEED="${SEED:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dct_n|--dct-n)
      DCT_N="${2:?missing value for $1}"
      shift 2
      ;;
    --dct_n=*|--dct-n=*)
      DCT_N="${1#*=}"
      shift
      ;;
    --add_name|--add-name)
      ADD_NAME="${2:?missing value for $1}"
      shift 2
      ;;
    --add_name=*|--add-name=*)
      ADD_NAME="${1#*=}"
      shift
      ;;
    --seed)
      SEED="${2:?missing value for $1}"
      shift 2
      ;;
    --seed=*)
      SEED="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

COMMON_DCT_ARGS=()
if [[ -n "${DCT_N}" ]]; then
  COMMON_DCT_ARGS+=(
    --dct_n="${DCT_N}"
  )
fi

echo "===== Running stage1 train ====="
python main.py \
  --exp_name=pvcp_video_t1 \
  --add_name="${ADD_NAME}" \
  --is_train=1 \
  --seed="${SEED}" \
  "${COMMON_DCT_ARGS[@]}"
