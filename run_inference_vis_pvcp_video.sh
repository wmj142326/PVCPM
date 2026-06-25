#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PVCP_VIDEO_DIR="$ROOT_DIR/pvcp_video"
AMASS_DIR="$(cd "$ROOT_DIR/../AMASS-skeleton-to-SMPL" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH_T1="${MODEL_PATH_T1:-$ROOT_DIR/ckpt/pvcp_video_t1/models/pvcp_video_t1_best.pth}"
MODEL_PATH_T2="${MODEL_PATH_T2:-$ROOT_DIR/ckpt/pvcp_video_t2/models/pvcp_video_t2_best.pth}"
INFER_DEVICE="${INFER_DEVICE:-cuda:0}"
INFER_SPLIT="${INFER_SPLIT:-test}"
SEED="${SEED:-88}"
SMPL_DEVICE="${SMPL_DEVICE:-cuda:0}"
SMPLIFY_ITERS="${SMPLIFY_ITERS:-50}"
SMPL_REDO="${SMPL_REDO:-1}"
SMPL_MESH_YAW_FIX_DEG="${SMPL_MESH_YAW_FIX_DEG:-180}"
DEFAULT_SAMPLE_COUNT="${DEFAULT_SAMPLE_COUNT:-5}"
SORT_PREDICTIONS_BY_GT="${SORT_PREDICTIONS_BY_GT:-1}"
GIF_FPS="${GIF_FPS:-25}"
SMPL_OVERVIEW_STRIDE="${SMPL_OVERVIEW_STRIDE:-2}"
FRAME_DATA_DIR="${FRAME_DATA_DIR:-$ROOT_DIR/dataset/pvcp/frame}"
VIDEO_META_JSON="${VIDEO_META_JSON:-$ROOT_DIR/dataset/pvcp/data_3d_pvcp_video_m35.json}"
SAVE_BBOX_FRAMES="${SAVE_BBOX_FRAMES:-1}"
BBOX_CROP_MARGIN="${BBOX_CROP_MARGIN:-0.15}"
BBOX_CROP_ASPECT="${BBOX_CROP_ASPECT:-0.55}"
BBOX_CROP_HEIGHT="${BBOX_CROP_HEIGHT:-512}"
BBOX_CROP_WIDTH="${BBOX_CROP_WIDTH:-0}"
POSE_IMAGE_YAW_DEG="${POSE_IMAGE_YAW_DEG:-20}"
VIS_X_ROT=0
VIS_Y_ROT=0
VIS_Z_ROT=0
VIS_VIEW_MODE="diverse_sampling"
VIS_YAW_DEG="$POSE_IMAGE_YAW_DEG"
SMOOTH_HEAD_MOTION="${SMOOTH_HEAD_MOTION:-0}"
FIX_HEAD_PITCH="${FIX_HEAD_PITCH:-1}"
HEAD_PITCH_AXIS="${HEAD_PITCH_AXIS:-0}"
STAGE2_SEMANTIC_FEATURE_FILE="${STAGE2_SEMANTIC_FEATURE_FILE:-$ROOT_DIR/dataset/pvcp/data_3d_pvcp_video_m35_Qwen3.5-9B_ped_pose_ego_vehicle_hist10_d512.rfeat.npz}"
STAGE2_USE_VIDEO="${STAGE2_USE_VIDEO:-1}"
STAGE2_VIDEO_IN_CONDITION="${STAGE2_VIDEO_IN_CONDITION:-0}"
STAGE2_VIDEO_IN_GAUSSIAN="${STAGE2_VIDEO_IN_GAUSSIAN:-1}"
DCT_N="${DCT_N:-20}"


die() {
  echo "Error: $*" >&2
  exit 1
}


print_command() {
  printf '\n$'
  printf ' %q' "$@"
  printf '\n'
}


require_file() {
  local path="$1"
  [[ -e "$path" ]] || die "Required path not found: $path"
}


run_inference() {
  (
    cd "$PVCP_VIDEO_DIR"
    "$PYTHON_BIN" inference.py "$@"
  )
}


run_skeleton2mesh() {
  (
    cd "$AMASS_DIR"
    "$PYTHON_BIN" skeleton2mesh.py "$@"
  )
}


run_visualize() {
  (
    cd "$AMASS_DIR"
    ./setup_headless.bash "$PYTHON_BIN" visualize_mesh_seq.py "$@"
  )
}


extract_max_start() {
  local scene_listing="$1"
  local segment_name="$2"
  awk -v seg="$segment_name" '
    $1 == seg ":" {
      if (match($0, /valid_window_start=\[0, [0-9]+\]/)) {
        value = substr($0, RSTART, RLENGTH)
        gsub(/[^0-9]/, " ", value)
        n = split(value, parts, /[[:space:]]+/)
        for (i = 1; i <= n; i++) {
          if (parts[i] != "") {
            last = parts[i]
          }
        }
        if (last != "") {
          print last
          exit
        }
      }
    }
  ' <<<"$scene_listing"
}


prompt_nonempty() {
  local prompt_text="$1"
  local value=""
  while true; do
    read -r -p "$prompt_text" value
    if [[ -n "$value" ]]; then
      printf '%s' "$value"
      return
    fi
    echo "Input cannot be empty."
  done
}


prompt_positive_int() {
  local prompt_text="$1"
  local default_value="${2:-}"
  local value=""
  while true; do
    read -r -p "$prompt_text" value
    if [[ -z "$value" && -n "$default_value" ]]; then
      value="$default_value"
    fi
    if [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
      printf '%s' "$value"
      return
    fi
    echo "Please enter a positive integer."
  done
}


prompt_window_start() {
  local max_start="$1"
  local value=""
  while true; do
    read -r -p "Choose window_start [0-${max_start}]: " value
    if [[ "$value" =~ ^[0-9]+$ ]] && (( value >= 0 && value <= max_start )); then
      printf '%s' "$value"
      return
    fi
    echo "Please enter an integer in [0, ${max_start}]."
  done
}


prompt_segment_name() {
  local scene_listing="$1"
  local segment_scene="$2"
  local value="" selected="" selected_max_start="" idx=""
  local segment_options=()

  mapfile -t segment_options < <(
    awk '
      /^[[:space:]]*[^[:space:]]+:/ {
        name = $1
        sub(/:$/, "", name)
        print name
      }
    ' <<<"$scene_listing"
  )

  if (( ${#segment_options[@]} == 0 )); then
    die "No valid segment was listed for scene [$segment_scene]."
  fi

  if (( ${#segment_options[@]} == 1 )); then
    selected="${segment_options[0]}"
    selected_max_start="$(extract_max_start "$scene_listing" "$selected")"
    segment_name="$selected"
    max_start="$selected_max_start"
    return
  fi

  while true; do
    echo "Available sub-sequences under scene [$segment_scene]:"
    for idx in "${!segment_options[@]}"; do
      printf '  %d: %s\n' "$((idx + 1))" "${segment_options[$idx]}"
    done
    read -r -p "Choose sub-sequence [1-${#segment_options[@]}]: " value

    selected=""
    if [[ "$value" =~ ^[0-9]+$ ]] && (( value >= 1 && value <= ${#segment_options[@]} )); then
      selected="${segment_options[$((value - 1))]}"
    else
      for idx in "${!segment_options[@]}"; do
        if [[ "$value" == "${segment_options[$idx]}" ]]; then
          selected="$value"
          break
        fi
      done
    fi

    if [[ -n "$selected" ]]; then
      selected_max_start="$(extract_max_start "$scene_listing" "$selected")"
      segment_name="$selected"
      max_start="$selected_max_start"
      return
    fi
    echo "sub-sequence [$value] is not valid for scene [$segment_scene]. Please choose one listed above."
  done
}


copy_rgb_window_frames() {
  "$PYTHON_BIN" - "$FRAME_DATA_DIR" "$VIDEO_META_JSON" "$scene_name" "$segment_name" "$window_start" "$output_dir" "$SAVE_BBOX_FRAMES" "$BBOX_CROP_MARGIN" "$BBOX_CROP_ASPECT" "$BBOX_CROP_HEIGHT" "$BBOX_CROP_WIDTH" <<'PY'
import json
import shutil
import sys
from pathlib import Path

from PIL import Image

frame_data_dir = Path(sys.argv[1])
video_meta_json = Path(sys.argv[2])
scene_name = sys.argv[3]
segment_name = sys.argv[4]
window_start = int(sys.argv[5])
output_dir = Path(sys.argv[6])
save_bbox_frames = sys.argv[7] == "1"
bbox_crop_margin = max(0.0, float(sys.argv[8]))
bbox_crop_aspect = max(1e-6, float(sys.argv[9]))
bbox_crop_height = int(sys.argv[10])
bbox_crop_width = int(sys.argv[11])
if bbox_crop_height <= 0:
    bbox_crop_height = 512
if bbox_crop_width <= 0:
    bbox_crop_width = max(1, int(round(bbox_crop_height * bbox_crop_aspect)))

try:
    resize_filter = Image.Resampling.LANCZOS
except AttributeError:
    resize_filter = Image.LANCZOS

history_len = 10
prediction_len = 25
total_len = history_len + prediction_len

with video_meta_json.open("r", encoding="utf-8") as f:
    loaded = json.load(f)
video_meta = loaded.get("video_meta", {})
try:
    segment_meta = video_meta[scene_name][segment_name]
except KeyError as exc:
    raise KeyError(f"Missing video_meta for {scene_name}/{segment_name}") from exc

frame_keys = list(segment_meta.get("frame_keys", []))
image_names = list(segment_meta.get("image_names", []))
bboxes = list(segment_meta.get("bbox", []))
bbox_valid = list(segment_meta.get("bbox_valid", []))
if len(frame_keys) < window_start + total_len or len(image_names) < window_start + total_len:
    raise ValueError(
        f"Window [{window_start}, {window_start + total_len}) exceeds available RGB metadata "
        f"for {scene_name}/{segment_name}: frame_keys={len(frame_keys)}, image_names={len(image_names)}"
    )
if save_bbox_frames and (len(bboxes) < window_start + total_len or len(bbox_valid) < window_start + total_len):
    raise ValueError(
        f"Window [{window_start}, {window_start + total_len}) exceeds available bbox metadata "
        f"for {scene_name}/{segment_name}: bbox={len(bboxes)}, bbox_valid={len(bbox_valid)}"
    )


def resolve_frame_path(image_name, frame_key):
    candidates = [
        frame_data_dir / image_name,
        frame_data_dir / scene_name / image_name,
        frame_data_dir / frame_key,
        frame_data_dir / scene_name / frame_key,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find RGB frame for {scene_name}/{segment_name}: "
        f"image_name={image_name}, frame_key={frame_key}. Checked: {[str(p) for p in candidates]}"
    )


def adjusted_bbox_crop_box(bbox, image_width, image_height, margin_ratio, target_aspect):
    x1, y1, box_w, box_h = [float(value) for value in bbox]
    x2 = x1 + max(1.0, box_w)
    y2 = y1 + max(1.0, box_h)

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    width *= 1.0 + 2.0 * margin_ratio
    height *= 1.0 + 2.0 * margin_ratio

    current_aspect = width / height
    if current_aspect < target_aspect:
        width = height * target_aspect
    else:
        height = width / target_aspect

    max_width = float(image_width)
    max_height = float(image_height)
    if width > max_width:
        width = max_width
        height = width / target_aspect
    if height > max_height:
        height = max_height
        width = height * target_aspect
    width = min(width, max_width)
    height = min(height, max_height)

    left = cx - 0.5 * width
    top = cy - 0.5 * height
    right = cx + 0.5 * width
    bottom = cy + 0.5 * height

    if left < 0:
        right -= left
        left = 0.0
    if right > image_width:
        left -= right - image_width
        right = float(image_width)
    if top < 0:
        bottom -= top
        top = 0.0
    if bottom > image_height:
        top -= bottom - image_height
        bottom = float(image_height)

    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(image_width), right)
    bottom = min(float(image_height), bottom)

    return (
        int(round(left)),
        int(round(top)),
        max(int(round(right)), int(round(left)) + 1),
        max(int(round(bottom)), int(round(top)) + 1),
    )


frame_dir = output_dir / "frame"
frame_dir.mkdir(parents=True, exist_ok=True)
for stale_file in frame_dir.iterdir():
    if stale_file.is_file() and stale_file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        stale_file.unlink()

bbox_dir = output_dir / "frame_bbox"
if save_bbox_frames:
    bbox_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in bbox_dir.iterdir():
        if stale_file.is_file() and stale_file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            stale_file.unlink()

for local_idx in range(total_len):
    meta_idx = window_start + local_idx
    src = resolve_frame_path(str(image_names[meta_idx]), str(frame_keys[meta_idx]))
    frame_id = window_start + local_idx
    dst = frame_dir / f"frame_{local_idx:03d}_{src.stem}_frame_id{frame_id}{src.suffix}"
    shutil.copy2(src, dst)

    if save_bbox_frames and bool(bbox_valid[meta_idx]):
        with Image.open(src) as image:
            image = image.convert("RGB")
            crop_box = adjusted_bbox_crop_box(
                bbox=bboxes[meta_idx],
                image_width=image.width,
                image_height=image.height,
                margin_ratio=bbox_crop_margin,
                target_aspect=bbox_crop_aspect,
            )
            cropped = image.crop(crop_box).resize((bbox_crop_width, bbox_crop_height), resize_filter)
            bbox_dst = bbox_dir / f"frame_{local_idx:03d}_{src.stem}_frame_id{frame_id}_bbox{src.suffix}"
            cropped.save(bbox_dst)

print(f"Saved RGB frames: {total_len} frames -> {frame_dir}")
if save_bbox_frames:
    print(
        f"Saved bbox RGB crops: {total_len} frames -> {bbox_dir} "
        f"(aspect={bbox_crop_aspect:g}, margin={bbox_crop_margin:g}, size={bbox_crop_width}x{bbox_crop_height})"
    )
PY
}


prepend_bbox_row_to_pose_overview() {
  "$PYTHON_BIN" - "$output_dir" "$window_start" "$SMPL_OVERVIEW_STRIDE" "$POSE_IMAGE_YAW_DEG" <<'PY'
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

output_dir = Path(sys.argv[1])
window_start = int(sys.argv[2])
stride = max(1, int(sys.argv[3]))
draw_yaw_deg = float(sys.argv[4])
history_len = 10
prediction_len = 25
total_len = history_len + prediction_len

image_dir = output_dir.parent / "images"
image_dir.mkdir(parents=True, exist_ok=True)
bbox_dir = output_dir / "frame_bbox"
pose_svg = image_dir / f"window_start_{window_start}.svg"
pose_pdf = image_dir / f"window_start_{window_start}.pdf"
for stale_path in (
    image_dir / f"window_start_{window_start}.png",
    image_dir / f"window_start_{window_start}_pose_only.png",
):
    if stale_path.is_file():
        stale_path.unlink()

obs_path = output_dir / "obs.npy"
gt_path = output_dir / "pred_far_gt.npy"
if not obs_path.is_file() or not gt_path.is_file() or not bbox_dir.is_dir():
    print(f"Skip pose overview rebuild: missing {obs_path}, {gt_path}, or {bbox_dir}")
    raise SystemExit(0)


def bbox_frame_png(local_idx: int) -> Path | None:
    matches = sorted(bbox_dir.glob(f"frame_{local_idx:03d}_*_bbox.*"))
    return matches[0] if matches else None


def pred_index(path: Path) -> int:
    match = re.search(r"pred_far_(\d+)\.npy$", path.name)
    return int(match.group(1)) if match else 10**9


def project_sequence(seq: np.ndarray, yaw_deg: float) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    x = seq[:, :, 0, :]
    y = seq[:, :, 1, :]
    z = seq[:, :, 2, :]
    view_x = np.cos(angle) * x - np.sin(angle) * z
    return np.stack((view_x, y), axis=2)


def draw_skeleton(ax, pose: np.ndarray, color_pair):
    for bone_idx, (start, end) in enumerate(zip(I_plot, J_plot)):
        color = color_pair[0] if LR_plot[bone_idx] else color_pair[1]
        xs = [pose[start, 0], pose[end, 0]]
        ys = [pose[start, 1], pose[end, 1]]
        ax.plot(xs, ys, lw=1.0, color=color, solid_capstyle="round")


I_plot = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19]
J_plot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
LR_plot = [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]
history_colors = ("#0B0B0B", "#B4B4B4")
gt_future_colors = ("#008000", "#66BB6A")
pred_future_colors = ("#0000CD", "#6495ED")

obs = np.load(obs_path)
gt_future = np.load(gt_path)
pred_paths = sorted(
    [path for path in output_dir.glob("pred_far_*.npy") if path.name != "pred_far_gt.npy"],
    key=pred_index,
)

rows = [("gt", np.concatenate((obs, gt_future), axis=0))]
rows.extend((path.stem, np.concatenate((obs, np.load(path)), axis=0)) for path in pred_paths)

seqs = np.stack([seq for _, seq in rows], axis=0).transpose(0, 2, 3, 1) * 1000.0
projected = project_sequence(seqs, draw_yaw_deg)

x_min = float(projected[:, :, 0, :].min())
x_max = float(projected[:, :, 0, :].max())
y_min = float(projected[:, :, 1, :].min())
y_max = float(projected[:, :, 1, :].max())
x_margin = max(150.0, 0.1 * (x_max - x_min))
y_margin = max(150.0, 0.1 * (y_max - y_min))
x_period = (x_min - x_margin, x_max + x_margin)
y_period = (y_min - y_margin, y_max + y_margin)

frame_indices = list(range(0, total_len, stride))
num_rows = len(rows) + 1
num_cols = len(frame_indices)
fig_w = max(8.0, 0.62 * num_cols + 0.9)
fig_h = max(3.0, 1.02 * num_rows + 0.45)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_w, fig_h), dpi=180)
if num_rows == 1:
    axes = np.expand_dims(axes, axis=0)
if num_cols == 1:
    axes = np.expand_dims(axes, axis=1)

for col_idx, local_idx in enumerate(frame_indices):
    ax = axes[0, col_idx]
    ax.axis("off")
    src = bbox_frame_png(local_idx)
    if src is None:
        continue
    with Image.open(src) as image:
        ax.imshow(image.convert("RGB"))

for row_idx, (row_label, _) in enumerate(rows, start=1):
    for col_idx, local_idx in enumerate(frame_indices):
        ax = axes[row_idx, col_idx]
        ax.axis("off")
        ax.set_xlim(*x_period)
        ax.set_ylim(*y_period)
        ax.set_aspect("equal", adjustable="box")
        color_pair = history_colors if local_idx < history_len else (gt_future_colors if row_idx == 1 else pred_future_colors)
        draw_skeleton(ax, projected[row_idx - 1, :, :, local_idx], color_pair)

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.0, hspace=0.02)
fig.savefig(pose_svg)
fig.savefig(pose_pdf)
plt.close(fig)
print(f"Saved pose overview with bbox row: {pose_svg}, {pose_pdf}")
PY
}


compose_smpl_overview_image() {
  "$PYTHON_BIN" - "$output_dir" "$window_start" "$SMPL_OVERVIEW_STRIDE" <<'PY'
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

output_dir = Path(sys.argv[1])
window_start = int(sys.argv[2])
stride = max(1, int(sys.argv[3]))
history_len = 10
total_len = 35

image_dir = output_dir.parent / "images"
image_dir.mkdir(parents=True, exist_ok=True)
svg_path = image_dir / f"window_start_{window_start}_smpl.svg"
pdf_path = image_dir / f"window_start_{window_start}_smpl.pdf"
for stale_path in (
    image_dir / f"window_start_{window_start}_smpl.png",
):
    if stale_path.is_file():
        stale_path.unlink()
bbox_dir = output_dir / "frame_bbox"


def frame_png(seq_dir: Path, frame_idx: int) -> Path | None:
    matches = sorted(seq_dir.glob(f"frame{frame_idx:03d}_frame_id*.png"))
    if matches:
        return matches[0]
    matches = sorted(seq_dir.glob(f"frame{frame_idx:03d}*.png"))
    return matches[0] if matches else None


def bbox_frame_png(local_idx: int) -> Path | None:
    matches = sorted(bbox_dir.glob(f"frame_{local_idx:03d}_*_bbox.*"))
    return matches[0] if matches else None


def pred_index(path: Path) -> int:
    match = re.search(r"pred_far_(\d+)_obj$", path.name)
    return int(match.group(1)) if match else 10**9


def crop_visible_region(image: Image.Image, margin: int = 8) -> Image.Image:
    image = image.convert("RGBA")
    alpha_bbox = image.getchannel("A").getbbox()
    if alpha_bbox is None:
        rgb = image.convert("RGB")
        bg = Image.new("RGB", rgb.size, "white")
        diff = Image.eval(ImageChops.difference(rgb, bg).convert("L"), lambda p: 255 if p > 8 else 0)
        alpha_bbox = diff.getbbox()
    if alpha_bbox is None:
        return image
    left, top, right, bottom = alpha_bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(image.width, right + margin)
    bottom = min(image.height, bottom + margin)
    return image.crop((left, top, right, bottom))


obs_dir = output_dir / "obs_obj"
gt_dir = output_dir / "pred_far_gt_obj"
pred_dirs = sorted(
    [path for path in output_dir.glob("pred_far_*_obj") if path.name != "pred_far_gt_obj"],
    key=pred_index,
)

rows = [("gt", gt_dir)] + [(path.name.removesuffix("_obj"), path) for path in pred_dirs]
frame_indices = list(range(0, total_len, stride))

num_rows = len(rows) + 1
num_cols = len(frame_indices)
fig_w = max(8.0, 0.62 * num_cols + 0.9)
fig_h = max(3.0, 1.02 * num_rows + 0.45)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_w, fig_h), dpi=180)
if num_rows == 1:
    axes = np.expand_dims(axes, axis=0)
if num_cols == 1:
    axes = np.expand_dims(axes, axis=1)

for col_idx, local_idx in enumerate(frame_indices):
    ax = axes[0, col_idx]
    ax.axis("off")
    src = bbox_frame_png(local_idx)
    if src is None:
        continue
    with Image.open(src) as image:
        ax.imshow(image.convert("RGB"))

for row_idx, (row_label, future_dir) in enumerate(rows, start=1):
    for col_idx, local_idx in enumerate(frame_indices):
        ax = axes[row_idx, col_idx]
        ax.axis("off")

        src = frame_png(obs_dir, local_idx) if local_idx < history_len else frame_png(future_dir, local_idx - history_len)
        if src is None:
            continue

        with Image.open(src) as image:
            image = crop_visible_region(image)
            ax.imshow(image)

plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=0.0, hspace=0.02)
fig.savefig(svg_path)
fig.savefig(pdf_path)
plt.close(fig)
print(f"Saved SMPL overview image: {svg_path}, {pdf_path}")
PY
}


require_file "$PVCP_VIDEO_DIR/inference.py"
require_file "$AMASS_DIR/skeleton2mesh.py"
require_file "$AMASS_DIR/visualize_mesh_seq.py"
require_file "$AMASS_DIR/setup_headless.bash"
require_file "$MODEL_PATH_T2"
require_file "$MODEL_PATH_T1"
require_file "$VIDEO_META_JSON"
if [[ "$STAGE2_USE_VIDEO" == "1" ]]; then
  require_file "$STAGE2_SEMANTIC_FEATURE_FILE"
fi

INFER_COMMON_ARGS=(
  --device "$INFER_DEVICE"
  --split "$INFER_SPLIT"
  --seed "$SEED"
  --sort_predictions_by_gt "$SORT_PREDICTIONS_BY_GT"
  --stage2_use_video "$STAGE2_USE_VIDEO"
  --stage2_video_in_condition "$STAGE2_VIDEO_IN_CONDITION"
  --stage2_video_in_gaussian "$STAGE2_VIDEO_IN_GAUSSIAN"
  --draw_yaw_deg "$POSE_IMAGE_YAW_DEG"
)
if [[ -n "$DCT_N" ]]; then
  INFER_COMMON_ARGS+=(--dct_n "$DCT_N")
fi
if [[ "$STAGE2_USE_VIDEO" == "1" ]]; then
  INFER_COMMON_ARGS+=(--stage2_semantic_feature_file "$STAGE2_SEMANTIC_FEATURE_FILE")
fi
SMPL_REDO_ARGS=()
if [[ "$SMPL_REDO" == "1" ]]; then
  SMPL_REDO_ARGS+=(--redo)
fi

echo "Model checkpoints:"
echo "  Stage 2: $MODEL_PATH_T2"
echo "  Stage 1: $MODEL_PATH_T1"
echo "Devices:"
echo "  Inference: $INFER_DEVICE"
echo "  Split:     $INFER_SPLIT"
echo "  Seed:      $SEED"
echo "  SMPLify:   $SMPL_DEVICE"
echo "  SMPL redo: $SMPL_REDO"
echo "  Mesh yaw fix: $SMPL_MESH_YAW_FIX_DEG"
echo "RGB frames:"
echo "  Frame dir: $FRAME_DATA_DIR"
echo "  Metadata:  $VIDEO_META_JSON"
echo "Semantic video:"
echo "  Stage 2 use video: $STAGE2_USE_VIDEO"
echo "  Stage 2 branches:  C=$STAGE2_VIDEO_IN_CONDITION G=$STAGE2_VIDEO_IN_GAUSSIAN"
if [[ "$STAGE2_USE_VIDEO" == "1" ]]; then
  echo "  Stage 2 cache:     $STAGE2_SEMANTIC_FEATURE_FILE"
fi
echo "Visualization:"
echo "  GIF fps:           $GIF_FPS"
echo "  Sort pred by GT:   $SORT_PREDICTIONS_BY_GT"
echo "  SMPL overview step:$SMPL_OVERVIEW_STRIDE"
echo "  Mesh rotation:     x=$VIS_X_ROT y=$VIS_Y_ROT z=$VIS_Z_ROT"
echo "  View mode/yaw:     $VIS_VIEW_MODE / $VIS_YAW_DEG"
echo "  Pose image yaw:    $POSE_IMAGE_YAW_DEG"
echo "  Smooth head:       $SMOOTH_HEAD_MOTION"
echo "  Fix head pitch:    $FIX_HEAD_PITCH axis=$HEAD_PITCH_AXIS"
if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" && -z "${CUDA_VISIBLE_DEVICES}" && "$INFER_DEVICE" == cuda* ]]; then
  echo "Notice: CUDA_VISIBLE_DEVICES is set to an empty string."
  echo "        inference.py will temporarily expose the requested GPU for this run."
  echo "        If you still see CPU fallback, try: unset CUDA_VISIBLE_DEVICES"
fi

scene_name=""
scene_listing=""
while true; do
  print_command "$PYTHON_BIN" inference.py "${INFER_COMMON_ARGS[@]}" --list_test_scenes
  run_inference "${INFER_COMMON_ARGS[@]}" --list_test_scenes

  echo
  scene_name="$(prompt_nonempty 'Choose scene_name: ')"

  print_command "$PYTHON_BIN" inference.py "${INFER_COMMON_ARGS[@]}" --scene_name "$scene_name"
  set +e
  scene_listing="$(run_inference "${INFER_COMMON_ARGS[@]}" --scene_name "$scene_name" 2>&1)"
  status=$?
  set -e
  printf '%s\n' "$scene_listing"

  if (( status == 0 )); then
    break
  fi
  echo "scene_name [$scene_name] is not valid. Please choose again."
done

segment_name=""
max_start=""
prompt_segment_name "$scene_listing" "$scene_name"

echo
echo "Using segment_name=$segment_name with valid window_start range [0, $max_start]."

echo
window_start="$(prompt_window_start "$max_start")"
sample_count="$(prompt_positive_int "Choose sample_count [default: ${DEFAULT_SAMPLE_COUNT}]: " "$DEFAULT_SAMPLE_COUNT")"

output_run_name="${scene_name}_start${window_start}_sample${sample_count}_seed${SEED}"
output_root="$ROOT_DIR/output/$output_run_name"
output_dir="$output_root/$scene_name/$segment_name/window_start_$window_start"

echo
echo "Selected run:"
echo "  scene_name=$scene_name"
echo "  segment_name=$segment_name"
echo "  window_start=$window_start"
echo "  sample_count=$sample_count"
echo "  output_root=$output_root"
echo "  output_dir=$output_dir"

print_command \
  "$PYTHON_BIN" inference.py \
  "${INFER_COMMON_ARGS[@]}" \
  --model_path_t2 "$MODEL_PATH_T2" \
  --model_path_t1 "$MODEL_PATH_T1" \
  --scene_name "$scene_name" \
  --segment_name "$segment_name" \
  --window_start "$window_start" \
  --output_root "$output_root" \
  --sample_count "$sample_count"
run_inference \
  "${INFER_COMMON_ARGS[@]}" \
  --model_path_t2 "$MODEL_PATH_T2" \
  --model_path_t1 "$MODEL_PATH_T1" \
  --scene_name "$scene_name" \
  --segment_name "$segment_name" \
  --window_start "$window_start" \
  --output_root "$output_root" \
  --sample_count "$sample_count"

[[ -d "$output_dir" ]] || die "Inference finished but output directory was not created: $output_dir"
copy_rgb_window_frames
prepend_bbox_row_to_pose_overview

print_command \
  "$PYTHON_BIN" skeleton2mesh.py \
  --directory "$output_dir" \
  --device "$SMPL_DEVICE" \
  --num_smplify_iters "$SMPLIFY_ITERS" \
  --smooth_head_motion "$SMOOTH_HEAD_MOTION" \
  --fix_head_pitch "$FIX_HEAD_PITCH" \
  --head_pitch_axis "$HEAD_PITCH_AXIS" \
  --mesh_yaw_fix_deg "$SMPL_MESH_YAW_FIX_DEG" \
  "${SMPL_REDO_ARGS[@]}"
run_skeleton2mesh \
  --directory "$output_dir" \
  --device "$SMPL_DEVICE" \
  --num_smplify_iters "$SMPLIFY_ITERS" \
  --smooth_head_motion "$SMOOTH_HEAD_MOTION" \
  --fix_head_pitch "$FIX_HEAD_PITCH" \
  --head_pitch_axis "$HEAD_PITCH_AXIS" \
  --mesh_yaw_fix_deg "$SMPL_MESH_YAW_FIX_DEG" \
  "${SMPL_REDO_ARGS[@]}"

print_command \
  ./setup_headless.bash \
  "$PYTHON_BIN" visualize_mesh_seq.py \
  -p "$output_dir" \
  --x_rot "$VIS_X_ROT" \
  --y_rot "$VIS_Y_ROT" \
  --z_rot "$VIS_Z_ROT" \
  --view_mode "$VIS_VIEW_MODE" \
  --yaw_deg "$VIS_YAW_DEG" \
  --gif_fps "$GIF_FPS" \
  --save_split_frames 0 \
  --history_frames 10 \
  --prediction_frames 25 \
  --if_concat_obs_in_gif \
  --redo
run_visualize \
  -p "$output_dir" \
  --x_rot "$VIS_X_ROT" \
  --y_rot "$VIS_Y_ROT" \
  --z_rot "$VIS_Z_ROT" \
  --view_mode "$VIS_VIEW_MODE" \
  --yaw_deg "$VIS_YAW_DEG" \
  --gif_fps "$GIF_FPS" \
  --save_split_frames 0 \
  --history_frames 10 \
  --prediction_frames 25 \
  --if_concat_obs_in_gif \
  --redo

compose_smpl_overview_image

echo
echo "All steps finished."
echo "Rendered meshes are under: $output_dir"
