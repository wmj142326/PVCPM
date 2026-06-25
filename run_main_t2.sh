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

normalize_add_name() {
  local suffix="$1"
  if [[ -n "${suffix}" && "${suffix}" != _* && "${suffix}" != -* ]]; then
    suffix="_${suffix}"
  fi
  printf '%s' "${suffix}"
}

trap print_runtime_summary EXIT

STAGE2_SEMANTIC_FEATURE_FILE="${STAGE2_SEMANTIC_FEATURE_FILE:-dataset/pvcp/data_3d_pvcp_video_m35_Qwen3.5-9B_ped_pose_ego_vehicle_hist10_d512.rfeat.npz}"
STAGE2_USE_VIDEO="${STAGE2_USE_VIDEO:-1}"

STAGE2_VIDEO_IN_CONDITION="${STAGE2_VIDEO_IN_CONDITION:-0}"
STAGE2_VIDEO_IN_GAUSSIAN="${STAGE2_VIDEO_IN_GAUSSIAN:-1}"
STAGE2_VIDEO_CONDITION_MODE="${STAGE2_VIDEO_CONDITION_MODE:-film}"

DCT_N="${DCT_N:-20}"

STAGE1_CKPT_NAME="${STAGE1_CKPT_NAME:-pvcp_video_t1}"
STAGE1_MODEL_FILE="${STAGE1_MODEL_FILE:-pvcp_video_t1_best.pth}"
MODEL_PATH_T1="${MODEL_PATH_T1:-}"
ADD_NAME="${ADD_NAME:-}"
STAGE2_EXP_NAME="${STAGE2_EXP_NAME:-pvcp_video_t2}"
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
    --stage2_semantic_feature_file|--stage2-semantic-feature-file)
      STAGE2_SEMANTIC_FEATURE_FILE="${2:?missing value for $1}"
      shift 2
      ;;
    --stage2_semantic_feature_file=*|--stage2-semantic-feature-file=*)
      STAGE2_SEMANTIC_FEATURE_FILE="${1#*=}"
      shift
      ;;
    --stage2_use_video|--stage2-use-video)
      STAGE2_USE_VIDEO="${2:?missing value for $1}"
      shift 2
      ;;
    --stage2_use_video=*|--stage2-use-video=*)
      STAGE2_USE_VIDEO="${1#*=}"
      shift
      ;;
    --stage2_video_in_condition|--stage2-video-in-condition)
      STAGE2_VIDEO_IN_CONDITION="${2:?missing value for $1}"
      shift 2
      ;;
    --stage2_video_in_condition=*|--stage2-video-in-condition=*)
      STAGE2_VIDEO_IN_CONDITION="${1#*=}"
      shift
      ;;
    --stage2_video_in_gaussian|--stage2-video-in-gaussian)
      STAGE2_VIDEO_IN_GAUSSIAN="${2:?missing value for $1}"
      shift 2
      ;;
    --stage2_video_in_gaussian=*|--stage2-video-in-gaussian=*)
      STAGE2_VIDEO_IN_GAUSSIAN="${1#*=}"
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
    --stage1_ckpt_name|--stage1-ckpt-name)
      STAGE1_CKPT_NAME="${2:?missing value for $1}"
      shift 2
      ;;
    --stage1_ckpt_name=*|--stage1-ckpt-name=*)
      STAGE1_CKPT_NAME="${1#*=}"
      shift
      ;;
    --stage1_model_file|--stage1-model-file)
      STAGE1_MODEL_FILE="${2:?missing value for $1}"
      shift 2
      ;;
    --stage1_model_file=*|--stage1-model-file=*)
      STAGE1_MODEL_FILE="${1#*=}"
      shift
      ;;
    --model_path_t1|--model-path-t1)
      MODEL_PATH_T1="${2:?missing value for $1}"
      shift 2
      ;;
    --model_path_t1=*|--model-path-t1=*)
      MODEL_PATH_T1="${1#*=}"
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

NORMALIZED_ADD_NAME="$(normalize_add_name "${ADD_NAME}")"
STAGE2_CKPT_NAME="${STAGE2_EXP_NAME}${NORMALIZED_ADD_NAME}"
if [[ -z "${MODEL_PATH_T1}" ]]; then
  MODEL_PATH_T1="ckpt/${STAGE1_CKPT_NAME}/models/${STAGE1_MODEL_FILE}"
fi

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
  --model_path_t1="${MODEL_PATH_T1}" \
  "${COMMON_DCT_ARGS[@]}" \
  "${COMMON_VIDEO_ARGS[@]}"

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
  --model_path_t1="${MODEL_PATH_T1}" \
  --model_path="ckpt/${STAGE2_CKPT_NAME}/models/${STAGE2_EXP_NAME}_best.pth" \
  "${COMMON_DCT_ARGS[@]}" \
  "${COMMON_VIDEO_ARGS[@]}"
