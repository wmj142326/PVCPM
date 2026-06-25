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

STAGE2_SEMANTIC_FEATURE_FILE="${STAGE2_SEMANTIC_FEATURE_FILE:-dataset/pvcp/data_3d_pvcp_video_m35_Qwen3.5-9B_ped_pose_ego_vehicle_hist10_d512.rfeat.npz}"

STAGE2_USE_VIDEO="${STAGE2_USE_VIDEO:-1}"
STAGE2_VIDEO_IN_CONDITION="${STAGE2_VIDEO_IN_CONDITION:-0}"
STAGE2_VIDEO_CONDITION_MODE="${STAGE2_VIDEO_CONDITION_MODE:-film}"
STAGE2_VIDEO_IN_GAUSSIAN="${STAGE2_VIDEO_IN_GAUSSIAN:-1}"

DCT_N="${DCT_N:-20}" ##
ADD_NAME="${ADD_NAME:-seed44_dct20}" ##
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-pvcp_video_t2}"
SEED="${SEED:-44}"


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
    --stage2_semantic_feature_file|--stage2-semantic-feature-file)
      STAGE2_SEMANTIC_FEATURE_FILE="${2:?missing value for $1}"
      shift 2
      ;;
    --stage2_semantic_feature_file=*|--stage2-semantic-feature-file=*)
      STAGE2_SEMANTIC_FEATURE_FILE="${1#*=}"
      shift
      ;;
    --stage2_video_condition_mode|--stage2-video-condition-mode)
      STAGE2_VIDEO_CONDITION_MODE="${2:?missing value for $1}"
      shift 2
      ;;
    --stage2_video_condition_mode=*|--stage2-video-condition-mode=*)
      STAGE2_VIDEO_CONDITION_MODE="${1#*=}"
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

STAGE2_VIDEO_CONDITION_MODE="${STAGE2_VIDEO_CONDITION_MODE,,}"
if [[ "${STAGE2_VIDEO_CONDITION_MODE}" != "cat" && "${STAGE2_VIDEO_CONDITION_MODE}" != "film" ]]; then
  echo "Unsupported stage2 video condition mode: ${STAGE2_VIDEO_CONDITION_MODE}. Expected cat or film." >&2
  exit 2
fi

NORMALIZED_ADD_NAME="${ADD_NAME}"
if [[ -n "${NORMALIZED_ADD_NAME}" && "${NORMALIZED_ADD_NAME}" != _* && "${NORMALIZED_ADD_NAME}" != -* ]]; then
  NORMALIZED_ADD_NAME="_${NORMALIZED_ADD_NAME}"
fi
STAGE1_CKPT_NAME="pvcp_video_t1${NORMALIZED_ADD_NAME}"
STAGE2_CKPT_NAME="${STAGE2_EXP_NAME}${NORMALIZED_ADD_NAME}"

COMMON_VIDEO_ARGS=()
if [[ "${STAGE2_USE_VIDEO}" == "1" ]]; then
  COMMON_VIDEO_ARGS+=(
    --stage2_semantic_feature_file="${STAGE2_SEMANTIC_FEATURE_FILE}"
  )
fi

COMMON_DCT_ARGS=()
if [[ -n "${DCT_N}" ]]; then
  COMMON_DCT_ARGS+=(
    --dct_n="${DCT_N}"
  )
fi

# stage1 train
echo "===== Running stage1 train ====="
python main.py \
  --exp_name=pvcp_video_t1 \
  --add_name="${ADD_NAME}" \
  --is_train=1 \
  --seed="${SEED}" \
  "${COMMON_DCT_ARGS[@]}" \
  "${COMMON_VIDEO_ARGS[@]}"

# stage2 train
echo "===== Running stage2 train ====="
python main.py \
  --exp_name="${STAGE2_EXP_NAME}" \
  --add_name="${ADD_NAME}" \
  --is_train=1 \
  --seed="${SEED}" \
  --stage2_use_video="${STAGE2_USE_VIDEO}" \
  --stage2_video_in_condition="${STAGE2_VIDEO_IN_CONDITION}" \
  --stage2_video_condition_mode="${STAGE2_VIDEO_CONDITION_MODE}" \
  --stage2_video_in_gaussian="${STAGE2_VIDEO_IN_GAUSSIAN}" \
  --model_path_t1="ckpt/${STAGE1_CKPT_NAME}/models/pvcp_video_t1_best.pth" \
  "${COMMON_DCT_ARGS[@]}" \
  "${COMMON_VIDEO_ARGS[@]}"

# stage2 eval/load
echo "===== Running stage2 eval/load ====="
python main.py \
  --exp_name="${STAGE2_EXP_NAME}" \
  --add_name="${ADD_NAME}" \
  --is_load=1 \
  --seed="${SEED}" \
  --stage2_use_video="${STAGE2_USE_VIDEO}" \
  --stage2_video_in_condition="${STAGE2_VIDEO_IN_CONDITION}" \
  --stage2_video_condition_mode="${STAGE2_VIDEO_CONDITION_MODE}" \
  --stage2_video_in_gaussian="${STAGE2_VIDEO_IN_GAUSSIAN}" \
  --model_path_t1="ckpt/${STAGE1_CKPT_NAME}/models/pvcp_video_t1_best.pth" \
  --model_path="ckpt/${STAGE2_CKPT_NAME}/models/${STAGE2_EXP_NAME}_best.pth" \
  "${COMMON_DCT_ARGS[@]}" \
  "${COMMON_VIDEO_ARGS[@]}"
  
