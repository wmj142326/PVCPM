#!/usr/bin/env python
# encoding: utf-8

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

SCRIPT_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _extract_raw_device_arg(argv: list[str]) -> str:
    for index, arg in enumerate(argv):
        if arg == "--device" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--device="):
            return arg.split("=", 1)[1]
    return "cuda:0"


def _requested_cuda_visible_index(device_text: str) -> str | None:
    normalized = str(device_text).strip().lower()
    if normalized in {"", "auto", "cuda"}:
        return "0"
    if normalized.startswith("cuda:"):
        index_text = normalized.split(":", 1)[1]
        if index_text.isdigit():
            return index_text
    return None


_RAW_DEVICE_ARG = _extract_raw_device_arg(sys.argv[1:])
_ISOLATED_VISIBLE_CUDA_DEVICE = None

# Some IDE terminals export CUDA_VISIBLE_DEVICES='' which hides all GPUs.
# If the user asked for CUDA, expose the requested physical GPU before
# importing torch so CUDA initialization can succeed.
if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
    _requested_visible_index = _requested_cuda_visible_index(_RAW_DEVICE_ARG)
    if _requested_visible_index is not None:
        _ISOLATED_VISIBLE_CUDA_DEVICE = _requested_visible_index
        os.environ["CUDA_VISIBLE_DEVICES"] = _requested_visible_index

try:
    import torch
    _TORCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local runtime
    torch = None
    _TORCH_IMPORT_ERROR = exc


ConfigDiverseSampling = None
apply_video_checkpoint_settings = None
format_video_setting_updates = None
validate_dct_checkpoint_settings = None
MaoweiGSPS_Dynamic_Seq_PVCP_Video = None
draw_multi_seqs_2d = None
dct_transform_torch = None
get_dct_matrix = None
reverse_dct_torch = None
CVAE = None
DiverseSampling = None
_PROJECT_MODULES_LOADED = False


def load_project_modules():
    global ConfigDiverseSampling
    global apply_video_checkpoint_settings
    global format_video_setting_updates
    global validate_dct_checkpoint_settings
    global MaoweiGSPS_Dynamic_Seq_PVCP_Video
    global draw_multi_seqs_2d
    global dct_transform_torch
    global get_dct_matrix
    global reverse_dct_torch
    global CVAE
    global DiverseSampling
    global _PROJECT_MODULES_LOADED

    if _PROJECT_MODULES_LOADED:
        return

    try:
        if __package__ is None or __package__ == "":
            if SCRIPT_REPO_ROOT not in sys.path:
                sys.path.insert(0, SCRIPT_REPO_ROOT)

            from pvcp_video.configs import ConfigDiverseSampling as _ConfigDiverseSampling
            from pvcp_video.configs.common import (
                apply_video_checkpoint_settings as _apply_video_checkpoint_settings,
                format_video_setting_updates as _format_video_setting_updates,
                validate_dct_checkpoint_settings as _validate_dct_checkpoint_settings,
            )
            from pvcp_video.datas import (
                MaoweiGSPS_Dynamic_Seq_PVCP_Video as _MaoweiGSPS_Dynamic_Seq_PVCP_Video,
                dct_transform_torch as _dct_transform_torch,
                draw_multi_seqs_2d as _draw_multi_seqs_2d,
                get_dct_matrix as _get_dct_matrix,
                reverse_dct_torch as _reverse_dct_torch,
            )
            from pvcp_video.nets import (
                CVAE as _CVAE,
                DiverseSampling as _DiverseSampling,
            )
        else:
            from .configs import ConfigDiverseSampling as _ConfigDiverseSampling
            from .configs.common import (
                apply_video_checkpoint_settings as _apply_video_checkpoint_settings,
                format_video_setting_updates as _format_video_setting_updates,
                validate_dct_checkpoint_settings as _validate_dct_checkpoint_settings,
            )
            from .datas import (
                MaoweiGSPS_Dynamic_Seq_PVCP_Video as _MaoweiGSPS_Dynamic_Seq_PVCP_Video,
                dct_transform_torch as _dct_transform_torch,
                draw_multi_seqs_2d as _draw_multi_seqs_2d,
                get_dct_matrix as _get_dct_matrix,
                reverse_dct_torch as _reverse_dct_torch,
            )
            from .nets import (
                CVAE as _CVAE,
                DiverseSampling as _DiverseSampling,
            )
    except Exception as exc:  # pragma: no cover - depends on local runtime
        raise RuntimeError(
            "Failed to import PVCP-video runtime modules. "
            "Please use a Python environment that can import torch and this project package."
        ) from exc

    ConfigDiverseSampling = _ConfigDiverseSampling
    apply_video_checkpoint_settings = _apply_video_checkpoint_settings
    format_video_setting_updates = _format_video_setting_updates
    validate_dct_checkpoint_settings = _validate_dct_checkpoint_settings
    MaoweiGSPS_Dynamic_Seq_PVCP_Video = _MaoweiGSPS_Dynamic_Seq_PVCP_Video
    draw_multi_seqs_2d = _draw_multi_seqs_2d
    dct_transform_torch = _dct_transform_torch
    get_dct_matrix = _get_dct_matrix
    reverse_dct_torch = _reverse_dct_torch
    CVAE = _CVAE
    DiverseSampling = _DiverseSampling
    _PROJECT_MODULES_LOADED = True


@dataclass(frozen=True)
class RuntimeDevice:
    requested_text: str
    device_text: str
    torch_device: "torch.device"
    is_cuda: bool
    fallback_reason: str = ""


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value: {value}")


def collect_explicit_cli_dests(parser, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    option_to_dest = {}
    for action in parser._actions:
        for option_string in action.option_strings:
            option_to_dest[option_string] = action.dest

    explicit_dests = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        option_token = token.split("=", 1)[0]
        dest = option_to_dest.get(option_token)
        if dest:
            explicit_dests.add(dest)
    return explicit_dests


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run PVCP-video diverse-sampling inference on one custom test window and export "
            "obs.npy / pred_far_gt.npy / pred_far_x.npy under output/{scene_name}/{segment_name}/window_start_x "
            "and save draw_multi_seqs_2d images under output/{scene_name}/{segment_name}/images."
        )
    )
    parser.add_argument("--exp_name", type=str, default="pvcp_video_t2", help="Only pvcp_video_t2 is supported.")
    parser.add_argument(
        "--model_path_t2",
        type=str,
        default="",
        help="Stage-2 checkpoint path. Required when running inference.",
    )
    parser.add_argument("--model_path_t1", type=str, default="", help="Stage-1 checkpoint path.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device, e.g. cuda:0 or cpu. Defaults to cuda:0 and falls back to cpu if CUDA is unavailable.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed used for sampling.")
    parser.add_argument("--sample_count", type=int, default=4, help="How many diverse futures to save.")
    parser.add_argument(
        "--sort_predictions_by_gt",
        type=str2bool,
        default=False,
        help="Sort sampled predictions by ADE to the future GT before saving and visualization.",
    )

    parser.add_argument("--scene_name", type=str, default="", help="Test scene name, e.g. S005.")
    parser.add_argument("--segment_name", type=str, default="", help="Sub-scene / segment name inside the scene.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train", "all"],
        help="Dataset split used for scene selection and visualization.",
    )
    parser.add_argument(
        "--window_start",
        type=int,
        default=None,
        help="Start frame index of the selected test window. Valid range depends on the segment length.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="",
        help="Output root. Defaults to {repo_root}/output if omitted.",
    )
    parser.add_argument(
        "--list_test_scenes",
        action="store_true",
        help="Print all scenes in the selected split and exit.",
    )

    parser.add_argument("--data_file", type=str, default="", help="Optional PVCP-video dataset npz filename.")
    parser.add_argument("--dct_n", type=int, default=None, help="DCT coefficient count.")
    parser.add_argument(
        "--draw_yaw_deg",
        type=float,
        default=None,
        help="Yaw angle used by the saved 2D skeleton visualization.",
    )
    parser.add_argument("--semantic_feature_file", type=str, default="", help="Optional backward-compatible semantic feature cache path.")
    parser.add_argument("--stage2_semantic_feature_file", type=str, default="", help="Stage-2 semantic feature cache path.")
    parser.add_argument("--semantic_input_dim", type=int, default=512, help="Window-level semantic feature dimension.")
    parser.add_argument("--stage2_use_video", type=str2bool, default=False, help="Whether Stage 2 uses semantic video.")
    parser.add_argument(
        "--stage2_video_in_condition",
        type=str2bool,
        default=False,
        help="Whether Stage 2 injects semantic video features into the condition branch.",
    )
    parser.add_argument(
        "--stage2_video_condition_mode",
        type=str,
        default="",
        choices=["", "cat", "film"],
        help="How Stage 2 uses semantic video in the condition branch: cat or film.",
    )
    parser.add_argument(
        "--stage2_video_in_gaussian",
        type=str2bool,
        default=False,
        help="Whether Stage 2 uses semantic video features to predict basis-selection logits W.",
    )
    parser.add_argument("--stage2_video_feat_dim", type=int, default=None, help="Stage-2 semantic projection dimension.")
    args = parser.parse_args(args=argv)
    args._explicit_cli_dests = collect_explicit_cli_dests(parser, argv)
    return args


def ensure_runtime_dependencies():
    if torch is None:
        raise RuntimeError(
            "PyTorch could not be imported in the current Python environment. "
            "Please run this script in the training/inference environment that has a working torch installation."
        ) from _TORCH_IMPORT_ERROR
    load_project_modules()


def resolve_repo_relative_path(path_text, repo_root=SCRIPT_REPO_ROOT):
    if not path_text:
        return ""
    if os.path.isabs(path_text):
        return path_text
    return os.path.abspath(os.path.join(repo_root, path_text))


def canonicalize_device_text(device):
    device_text = str(device).strip().lower()
    if device_text in {"", "auto", "cuda"}:
        return "cuda:0"
    if device_text.startswith("cuda:"):
        device_index_text = device_text.split(":", 1)[1]
        try:
            int(device_index_text)
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device string: {device!r}") from exc
    return device_text or "cpu"


def summarize_exception(exc):
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    if not lines:
        return exc.__class__.__name__
    summary = lines[0]
    if exc.__class__.__name__ not in summary:
        summary = f"{exc.__class__.__name__}: {summary}"
    return summary


def make_cpu_runtime(requested_text, reason=""):
    return RuntimeDevice(
        requested_text=requested_text,
        device_text="cpu",
        torch_device=torch.device("cpu"),
        is_cuda=False,
        fallback_reason=reason,
    )


def resolve_runtime_device(requested_device):
    requested_text = canonicalize_device_text(requested_device)
    if requested_text == "cpu":
        return make_cpu_runtime(requested_text)

    if not requested_text.startswith("cuda"):
        try:
            return RuntimeDevice(
                requested_text=requested_text,
                device_text=requested_text,
                torch_device=torch.device(requested_text),
                is_cuda=False,
                fallback_reason="",
            )
        except Exception as exc:
            raise ValueError(f"Unsupported torch device: {requested_device!r}") from exc

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
    except Exception as exc:
        return make_cpu_runtime(
            requested_text,
            reason=(
                f"CUDA probe failed for requested device {requested_text!r}: {summarize_exception(exc)}. "
                f"CUDA_VISIBLE_DEVICES={visible_devices!r}. Falling back to CPU."
            ),
        )

    if not cuda_available or device_count <= 0:
        return make_cpu_runtime(
            requested_text,
            reason=(
                f"Requested --device {requested_text}, but torch cannot see any usable CUDA device. "
                f"CUDA_VISIBLE_DEVICES={visible_devices!r}. Falling back to CPU."
            ),
        )

    device_index = int(requested_text.split(":", 1)[1]) if ":" in requested_text else 0
    fallback_reason = ""
    if device_index < 0 or device_index >= device_count:
        fallback_reason = (
            f"Requested --device {requested_text}, but this process only sees {device_count} CUDA device(s). "
            f"Using cuda:0 instead. CUDA_VISIBLE_DEVICES={visible_devices!r}."
        )
        device_index = 0

    resolved_text = f"cuda:{device_index}"
    resolved_device = torch.device(resolved_text)
    try:
        probe_tensor = torch.empty(1, device=resolved_device)
        del probe_tensor
    except Exception as exc:
        return make_cpu_runtime(
            requested_text,
            reason=(
                f"CUDA initialization failed for {resolved_text}: {summarize_exception(exc)}. "
                f"CUDA_VISIBLE_DEVICES={visible_devices!r}. Falling back to CPU."
            ),
        )

    return RuntimeDevice(
        requested_text=requested_text,
        device_text=resolved_text,
        torch_device=resolved_device,
        is_cuda=True,
        fallback_reason=fallback_reason,
    )


def announce_runtime_device(runtime):
    if runtime.fallback_reason:
        print(runtime.fallback_reason)
    if runtime.requested_text == runtime.device_text:
        print(f"Using runtime device: {runtime.device_text}")
    else:
        print(f"Using runtime device: {runtime.device_text} (requested {runtime.requested_text})")


def seed_everything(seed, runtime):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if runtime.is_cuda:
        try:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception as exc:
            print(f"CUDA seeding failed on {runtime.device_text}: {exc}. Continuing without CUDA-specific seeds.")

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_checkpoint_state_cpu(checkpoint_path):
    checkpoint_path = resolve_repo_relative_path(checkpoint_path)
    if not checkpoint_path:
        raise FileNotFoundError("Checkpoint path is empty.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(state).__name__}: {checkpoint_path}")
    return checkpoint_path, state


def load_checkpoint_cfg_snapshot(checkpoint_path):
    checkpoint_path, state = load_checkpoint_state_cpu(checkpoint_path)
    checkpoint_cfg = state.get("cfg_snapshot", {})
    if not isinstance(checkpoint_cfg, dict):
        checkpoint_cfg = {}
    return checkpoint_path, checkpoint_cfg


def maybe_apply_checkpoint_video_settings(args):
    if not args.model_path_t2:
        return

    explicit_cli_dests = getattr(args, "_explicit_cli_dests", set())
    stage1_checkpoint_path = resolve_repo_relative_path(args.model_path_t1) if args.model_path_t1 else ""
    if stage1_checkpoint_path and os.path.exists(stage1_checkpoint_path):
        args.model_path_t1 = stage1_checkpoint_path

    args.model_path_t2 = resolve_repo_relative_path(args.model_path_t2)
    resolved_path_t2, checkpoint_cfg_t2 = load_checkpoint_cfg_snapshot(args.model_path_t2)
    updates_t2 = apply_video_checkpoint_settings(
        args,
        checkpoint_cfg_t2,
        stages=("stage2",),
        include_runtime_settings=True,
        skip_keys=explicit_cli_dests,
        repo_root=SCRIPT_REPO_ROOT,
    )
    if updates_t2:
        print(
            "Auto-applied Stage-2 checkpoint settings from {}: {}".format(
                resolved_path_t2,
                format_video_setting_updates(updates_t2),
            )
        )


def build_cfg(args):
    cfg_args = SimpleNamespace(**vars(args))
    if getattr(cfg_args, "model_path_t1", ""):
        cfg_args.model_path_t1 = resolve_repo_relative_path(cfg_args.model_path_t1)

    cfg = ConfigDiverseSampling(exp_name=args.exp_name, device=args.device, num_works=0, args=cfg_args)
    if cfg.model_path_t1:
        cfg.model_path_t1 = resolve_repo_relative_path(cfg.model_path_t1, cfg.repo_root)
    if args.draw_yaw_deg is not None:
        cfg.draw_yaw_deg = float(args.draw_yaw_deg)
    return cfg


def resolve_output_root(output_root, repo_root):
    if not output_root:
        return os.path.join(repo_root, "output")
    if os.path.isabs(output_root):
        return output_root
    return os.path.abspath(os.path.join(repo_root, output_root))


def build_inference_dataset(cfg, split="test"):
    return MaoweiGSPS_Dynamic_Seq_PVCP_Video(
        data_path=cfg.base_data_dir,
        data_file=cfg.data_file,
        scene_split_path=cfg.scene_split_path,
        test_ratio=cfg.test_ratio,
        t_his=cfg.t_his,
        t_pred=cfg.t_pred,
        similar_cnt=0,
        dynamic_sub_len=cfg.sub_len_train,
        batch_size=1,
        joint_used=cfg.joint_used,
        parents=cfg.parents,
        mode=split,
        multimodal_threshold=cfg.multimodal_threshold,
        sample_step_train=cfg.sample_step_train,
        sample_step_test=cfg.sample_step_test,
        is_debug=False,
        load_video=cfg.stage2_video_active,
        stage2_semantic_feature_file=cfg.stage2_semantic_feature_file,
        semantic_input_dim=cfg.semantic_input_dim,
    )


def build_models(cfg):
    model_t1 = CVAE(
        node_n=cfg.node_n,
        hidden_dim=cfg.hidden_dim,
        z_dim=cfg.z_dim,
        dct_n=cfg.dct_n,
        dropout_rate=cfg.dropout_rate,
    )
    model_t2 = DiverseSampling(
        node_n=cfg.node_n,
        hidden_dim=cfg.hidden_dim,
        base_dim=cfg.base_dim,
        base_num_p1=cfg.base_num_p1,
        z_dim=cfg.z_dim,
        dct_n=cfg.dct_n,
        dropout_rate=cfg.dropout_rate,
        use_video=cfg.stage2_video_active,
        video_feat_dim=cfg.stage2_video_feat_dim,
        semantic_input_dim=cfg.semantic_input_dim,
        video_in_condition=cfg.stage2_video_in_condition,
        video_in_gaussian=cfg.stage2_video_in_gaussian,
        video_condition_mode=cfg.stage2_video_condition_mode,
    )
    return model_t1, model_t2


def validate_stage1_checkpoint(cfg, checkpoint_state):
    checkpoint_cfg = checkpoint_state.get("cfg_snapshot", {})
    if not checkpoint_cfg:
        validate_dct_checkpoint_settings(cfg, checkpoint_cfg, context="Stage-1 checkpoint")
        return
    validate_dct_checkpoint_settings(cfg, checkpoint_cfg, context="Stage-1 checkpoint")

def move_models_to_runtime_device(model_t1, model_t2, runtime):
    if not runtime.is_cuda:
        return runtime

    try:
        model_t1.to(runtime.torch_device)
        model_t2.to(runtime.torch_device)
        return runtime
    except Exception as exc:
        fallback_runtime = make_cpu_runtime(
            runtime.requested_text,
            reason=f"Moving models to {runtime.device_text} failed: {exc}. Falling back to CPU.",
        )
        try:
            model_t1.to(fallback_runtime.torch_device)
            model_t2.to(fallback_runtime.torch_device)
        finally:
            if hasattr(torch.cuda, "empty_cache"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        return fallback_runtime


def load_models(cfg, model_t1, model_t2, stage2_checkpoint_path):
    stage1_checkpoint_path = resolve_repo_relative_path(cfg.model_path_t1, cfg.repo_root)
    if not stage1_checkpoint_path:
        raise FileNotFoundError(
            "Stage-1 checkpoint path is empty. Please pass --model_path_t1 or configure it in the checkpoint."
        )

    stage1_checkpoint_path, stage1_state = load_checkpoint_state_cpu(stage1_checkpoint_path)
    validate_stage1_checkpoint(cfg, stage1_state)
    model_t1.load_state_dict(stage1_state["model"])
    for param in model_t1.parameters():
        param.requires_grad = False
    model_t1.eval()
    print(f"Loaded Stage-1 checkpoint: {stage1_checkpoint_path}")

    stage2_checkpoint_path, stage2_state = load_checkpoint_state_cpu(stage2_checkpoint_path)
    validate_dct_checkpoint_settings(cfg, stage2_state.get("cfg_snapshot", {}), context="Stage-2 checkpoint")
    model_t2.load_state_dict(stage2_state["model"])
    model_t2.eval()
    print(f"Loaded Stage-2 checkpoint: {stage2_checkpoint_path}")


def numpy_to_device_tensor(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def batch_to_device(batch, device):
    pose = numpy_to_device_tensor(batch["pose"], device)
    stage2_semantic_features = batch.get("stage2_semantic_features")
    if stage2_semantic_features is None:
        stage2_semantic_features = batch.get("history_semantic_features")
    if stage2_semantic_features is not None:
        stage2_semantic_features = numpy_to_device_tensor(stage2_semantic_features, device)
    return pose, stage2_semantic_features


def select_video_feature_batch(semantic_features, use_video):
    if not use_video:
        return None
    return semantic_features


def list_split_scenes(dataset, split):
    print(f"Available {split} scenes:")
    for scene_name in dataset.scene_names:
        segment_count = len(dataset.data.get(scene_name, {}))
        print(f"  {scene_name} ({segment_count} segments)")


def list_scene_segments(dataset, scene_name, split):
    scene_segments = dataset.data.get(scene_name)
    if not scene_segments:
        raise ValueError(f"Scene {scene_name} is not available in the {split} split.")

    print(f"Available segments in {split} scene {scene_name}:")
    for segment_name in sorted(scene_segments.keys()):
        seq_len = dataset.segment_lookup[(scene_name, segment_name)].shape[0]
        max_start = seq_len - dataset.t_total
        if max_start < 0:
            continue
        print(f"  {segment_name}: length={seq_len}, valid_window_start=[0, {max_start}]")


def build_selected_batch(dataset, scene_name, segment_name, window_start, split):
    if scene_name not in dataset.scene_names:
        raise ValueError(
            f"Scene {scene_name} is not in the {split} split. "
            f"Available {split} scenes: {', '.join(dataset.scene_names[:20])}"
        )
    if scene_name not in dataset.data or segment_name not in dataset.data[scene_name]:
        raise ValueError(f"Segment {scene_name}/{segment_name} was not found in the {split} split.")

    seq = dataset.segment_lookup[(scene_name, segment_name)]
    max_start = seq.shape[0] - dataset.t_total
    if max_start < 0:
        raise ValueError(
            f"Segment {scene_name}/{segment_name} is too short for t_total={dataset.t_total}: length={seq.shape[0]}"
        )
    if window_start is None:
        raise ValueError(
            f"window_start is required for inference. Valid range for {scene_name}/{segment_name}: [0, {max_start}]"
        )
    if window_start < 0 or window_start > max_start:
        raise ValueError(
            f"Invalid window_start={window_start} for {scene_name}/{segment_name}. Valid range: [0, {max_start}]"
        )

    window = seq[window_start : window_start + dataset.t_total]
    sample_bundle = dataset._build_sample_bundle(scene_name, segment_name, window_start, window)
    batch = dataset._collate_sample_bundles([sample_bundle])
    return batch, max_start


def model_output_to_sequence(data, node_n):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data[None, ...]
    joint_data = data.reshape(data.shape[0], node_n, 3, data.shape[-1])
    root = np.zeros((joint_data.shape[0], 1, 3, joint_data.shape[-1]), dtype=np.float32)
    full_pose = np.concatenate((root, joint_data), axis=1)
    return np.transpose(full_pose, (0, 3, 1, 2))


def add_root_for_draw(data, node_n):
    data = np.asarray(data, dtype=np.float32).reshape(data.shape[0], node_n, 3, data.shape[-1])
    root = np.zeros((data.shape[0], 1, 3, data.shape[-1]), dtype=np.float32)
    return np.concatenate((root, data), axis=1)


def draw_history_count(t_his):
    return max(1, int(math.ceil(t_his / 2.0)))


DRAW_HISTORY_COLORS = ("#666666", "#B4B4B4")
DRAW_GT_FUTURE_COLORS = ("#2E8B57", "#8FBC8F")
DRAW_PRED_FUTURE_COLORS = ("#0000CD", "#6495ED")


def project_oblique_view_for_draw(data_3d, yaw_deg):
    angle = np.deg2rad(yaw_deg)
    x = data_3d[:, :, 0, :]
    y = data_3d[:, :, 1, :]
    z = data_3d[:, :, 2, :]

    view_x = np.cos(angle) * x - np.sin(angle) * z
    projected = np.stack((view_x, y), axis=2)

    x_min = float(projected[:, :, 0, :].min())
    x_max = float(projected[:, :, 0, :].max())
    y_min = float(projected[:, :, 1, :].min())
    y_max = float(projected[:, :, 1, :].max())
    x_margin = max(150.0, 0.1 * (x_max - x_min))
    y_margin = max(150.0, 0.1 * (y_max - y_min))

    x_period = [x_min - x_margin, x_max + x_margin]
    vertical_period = [y_min - y_margin, y_max + y_margin]
    return projected, x_period, vertical_period


def save_draw_figure(cfg, scene_name, segment_name, window_start, full_pose, pred_only, output_root):
    image_dir = os.path.join(output_root, scene_name, segment_name, "images")
    os.makedirs(image_dir, exist_ok=True)

    origin = add_root_for_draw(full_pose, cfg.node_n) * 1000.0
    all_outputs = add_root_for_draw(pred_only, cfg.node_n) * 1000.0
    all_outputs = np.concatenate(
        (np.repeat(origin[:, :, :, : cfg.t_his], repeats=pred_only.shape[0], axis=0), all_outputs),
        axis=-1,
    )

    all_to_draw = np.concatenate((origin, all_outputs), axis=0)
    draw_acc = list(range(0, all_to_draw.shape[-1], 2))
    all_to_draw = all_to_draw[:, :, :, draw_acc]
    frame_ids = [window_start + frame_idx for frame_idx in draw_acc]
    row_labels = ["gt"] + [f"pred_far_{sample_idx}" for sample_idx in range(pred_only.shape[0])]
    draw_yaw_deg = float(getattr(cfg, "draw_yaw_deg", -45.0))
    all_to_draw, x_period, vertical_period = project_oblique_view_for_draw(all_to_draw, draw_yaw_deg)

    svg_path = os.path.join(image_dir, f"window_start_{window_start}.svg")
    for stale_path in (
        os.path.join(image_dir, f"window_start_{window_start}.png"),
        os.path.join(image_dir, f"window_start_{window_start}.pdf"),
    ):
        if os.path.exists(stale_path):
            os.remove(stale_path)
    draw_multi_seqs_2d(
        all_to_draw.copy(),
        gt_cnt=1,
        t_his=draw_history_count(cfg.t_his),
        I=cfg.I_plot,
        J=cfg.J_plot,
        LR=cfg.LR_plot,
        x_period=x_period,
        z_period=vertical_period,
        xlabel="frame id",
        ylabel="diversity id",
        history_colors=DRAW_HISTORY_COLORS,
        gt_future_colors=DRAW_GT_FUTURE_COLORS,
        pred_future_colors=DRAW_PRED_FUTURE_COLORS,
        frame_ids=frame_ids,
        row_labels=row_labels,
        full_path=svg_path,
    )
    return svg_path


def infer_future_sequences(cfg, runtime, model_t1, model_t2, batch, sample_count):
    pose, stage2_semantic_features = batch_to_device(batch, runtime.torch_device)
    stage2_video_features = select_video_feature_batch(
        semantic_features=stage2_semantic_features,
        use_video=cfg.stage2_video_active,
    )

    dct_m, i_dct_m = get_dct_matrix(cfg.t_total)
    dct_m = numpy_to_device_tensor(dct_m, runtime.torch_device)
    i_dct_m = numpy_to_device_tensor(i_dct_m, runtime.torch_device)

    with torch.no_grad():
        padded_inputs = pose[:, :, list(range(cfg.t_his)) + [cfg.t_his - 1] * cfg.t_pred]
        padded_inputs_coeff = dct_transform_torch(padded_inputs, dct_m, dct_n=cfg.dct_n)
        padded_inputs_coeff = padded_inputs_coeff.view(pose.shape[0], -1, 3 * cfg.dct_n)

        repeated_eps = torch.randn((pose.shape[0] * sample_count, cfg.z_dim), device=runtime.torch_device)

        all_z, _, _ = model_t2(
            condition=padded_inputs_coeff,
            repeated_eps=repeated_eps,
            many_weights=None,
            multi_modal_head=sample_count,
            video_features=stage2_video_features,
            temperature=cfg.temperature_p1,
        )
        all_outs_coeff = model_t1.inference(
            condition=torch.repeat_interleave(padded_inputs_coeff, repeats=sample_count, dim=0),
            z=all_z,
        )
        all_outs_coeff = all_outs_coeff.reshape(sample_count, -1, cfg.dct_n)
        outputs = reverse_dct_torch(all_outs_coeff, i_dct_m, cfg.t_total)
        pred_only = outputs[:, :, cfg.t_his :].detach().cpu().numpy()
        obs_only = pose[:, :, : cfg.t_his].detach().cpu().numpy()
        full_pose = pose.detach().cpu().numpy()

    obs_seq = model_output_to_sequence(obs_only, cfg.node_n)[0]
    pred_seq = model_output_to_sequence(pred_only, cfg.node_n)
    return (
        obs_seq.astype(np.float32),
        pred_seq.astype(np.float32),
        full_pose.astype(np.float32),
        pred_only.astype(np.float32),
    )


def compute_prediction_gt_metrics(pred_seq, gt_future_seq):
    errors = np.linalg.norm(pred_seq - gt_future_seq[None, ...], axis=-1)
    ade = errors.mean(axis=(1, 2))
    fde = errors[:, -1, :].mean(axis=1)
    return ade, fde


def sort_predictions_by_gt(pred_seq, pred_only, gt_future_seq):
    ade, fde = compute_prediction_gt_metrics(pred_seq, gt_future_seq)
    order = np.lexsort((fde, ade))
    return pred_seq[order], pred_only[order], order, ade, fde


def save_prediction_order(target_dir, order, ade, fde):
    order_path = os.path.join(target_dir, "prediction_order_by_gt.txt")
    with open(order_path, "w", encoding="utf-8") as f:
        f.write("rank saved_name original_sample_idx ade fde\n")
        for rank, original_idx in enumerate(order):
            f.write(
                f"{rank} pred_far_{rank}.npy {int(original_idx)} "
                f"{float(ade[original_idx]):.8f} {float(fde[original_idx]):.8f}\n"
            )
    return order_path


def save_outputs(
    cfg,
    scene_name,
    segment_name,
    window_start,
    obs_seq,
    pred_seq,
    full_pose,
    pred_only,
    output_root,
    sort_by_gt=False,
):
    target_dir = os.path.join(output_root, scene_name, segment_name, f"window_start_{window_start}")
    os.makedirs(target_dir, exist_ok=True)

    obs_path = os.path.join(target_dir, "obs.npy")
    np.save(obs_path, obs_seq)
    gt_future_seq = model_output_to_sequence(full_pose[:, :, cfg.t_his :], cfg.node_n)[0]
    gt_future_path = os.path.join(target_dir, "pred_far_gt.npy")
    np.save(gt_future_path, gt_future_seq)
    order_path = ""

    if sort_by_gt:
        pred_seq, pred_only, order, ade, fde = sort_predictions_by_gt(pred_seq, pred_only, gt_future_seq)
        order_path = save_prediction_order(target_dir, order, ade, fde)

    for sample_idx, pred in enumerate(pred_seq):
        pred_path = os.path.join(target_dir, f"pred_far_{sample_idx}.npy")
        np.save(pred_path, pred)

    image_path = save_draw_figure(
        cfg=cfg,
        scene_name=scene_name,
        segment_name=segment_name,
        window_start=window_start,
        full_pose=full_pose,
        pred_only=pred_only,
        output_root=output_root,
    )

    print(
        f"Saved future GT plus {pred_seq.shape[0]} predictions for {scene_name}/{segment_name} "
        f"(window_start={window_start}) to {target_dir}"
    )
    print(f"  obs.npy shape: {obs_seq.shape}")
    print(f"  pred_far_gt.npy shape: {gt_future_seq.shape}")
    print(f"  pred_far_0.npy shape: {pred_seq[0].shape}")
    if order_path:
        print(f"  prediction order: {order_path}")
    print(f"  draw image: {image_path}")


def main(argv=None):
    args = parse_args(argv)
    ensure_runtime_dependencies()

    if args.exp_name != "pvcp_video_t2":
        raise ValueError(f"Only pvcp_video_t2 is supported, got {args.exp_name}")
    if args.sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {args.sample_count}")

    maybe_apply_checkpoint_video_settings(args)
    runtime = resolve_runtime_device(args.device)
    announce_runtime_device(runtime)
    args.device = runtime.device_text
    seed_everything(args.seed, runtime)

    cfg = build_cfg(args)
    cfg.device = runtime.device_text
    output_root = resolve_output_root(args.output_root, cfg.repo_root)

    dataset = build_inference_dataset(cfg, split=args.split)

    if args.list_test_scenes:
        list_split_scenes(dataset, args.split)
        return

    if not args.scene_name:
        raise ValueError(f"scene_name is required. Use --list_test_scenes to inspect available {args.split} scenes.")
    if not args.segment_name:
        list_scene_segments(dataset, args.scene_name, args.split)
        return
    if not args.model_path_t2:
        raise ValueError("model_path_t2 is required when running inference.")

    batch, max_start = build_selected_batch(
        dataset=dataset,
        scene_name=args.scene_name,
        segment_name=args.segment_name,
        window_start=args.window_start,
        split=args.split,
    )
    print(
        f"Selected test window: scene={args.scene_name}, segment={args.segment_name}, "
        f"window_start={args.window_start}, max_window_start={max_start}"
    )

    model_t1, model_t2 = build_models(cfg)
    load_models(
        cfg=cfg,
        model_t1=model_t1,
        model_t2=model_t2,
        stage2_checkpoint_path=resolve_repo_relative_path(args.model_path_t2, cfg.repo_root),
    )

    runtime = move_models_to_runtime_device(model_t1, model_t2, runtime)
    if cfg.device != runtime.device_text:
        cfg.device = runtime.device_text
        args.device = runtime.device_text
        announce_runtime_device(runtime)

    obs_seq, pred_seq, full_pose, pred_only = infer_future_sequences(
        cfg=cfg,
        runtime=runtime,
        model_t1=model_t1,
        model_t2=model_t2,
        batch=batch,
        sample_count=args.sample_count,
    )
    save_outputs(
        cfg=cfg,
        scene_name=args.scene_name,
        segment_name=args.segment_name,
        window_start=args.window_start,
        obs_seq=obs_seq,
        pred_seq=pred_seq,
        full_pose=full_pose,
        pred_only=pred_only,
        output_root=output_root,
        sort_by_gt=args.sort_predictions_by_gt,
    )


if __name__ == "__main__":
    main()

"""
Usage examples (run these after `cd /home/guest/wmj/Projects/diverse_sampling/pvcp_video`)

1. List all test scenes:
python inference.py --list_test_scenes

2. List all segments under one test scene, plus each segment's valid window_start range:
python inference.py --scene_name S005

3. Run inference on one selected test window and export obs.npy / pred_far_x.npy plus a draw_multi_seqs_2d image:
python inference.py \
  --model_path_t2 /home/guest/wmj/Projects/diverse_sampling/ckpt/pvcp_video_t2/models/pvcp_video_t2_best.pth \
  --model_path_t1 /home/guest/wmj/Projects/diverse_sampling/ckpt/pvcp_video_t1/models/pvcp_video_t1_best.pth \
  --scene_name S072 \
  --segment_name S072 \
  --window_start 50 \
  --sample_count 5

  CUDA_VISIBLE_DEVICES=1 ./run_inference_vis.sh
"""
