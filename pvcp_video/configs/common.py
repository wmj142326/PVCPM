#!/usr/bin/env python
# encoding: utf-8

from __future__ import annotations

import os
import warnings


_TRUE_TEXT = {"1", "true", "t", "yes", "y", "on"}
_FALSE_TEXT = {"0", "false", "f", "no", "n", "off"}
_VIDEO_STAGE_KEYS = {
    "stage1": (),
    "stage2": (
        "stage2_use_video",
        "stage2_video_feat_dim",
        "stage2_video_in_condition",
        "stage2_video_condition_mode",
        "stage2_video_in_gaussian",
    ),
}
_VIDEO_RUNTIME_KEYS = (
    "data_file",
    "dct_n",
    "semantic_input_dim",
    "semantic_feature_file",
    "stage2_semantic_feature_file",
)
_VIDEO_STAGE2_RUNTIME_KEYS = ("model_path_t1",)
_VIDEO_BOOL_KEYS = {
    "stage2_use_video",
    "stage2_video_in_condition",
    "stage2_video_in_gaussian",
}
_VIDEO_INT_KEYS = {
    "dct_n",
    "semantic_input_dim",
    "stage2_video_feat_dim",
}
_VIDEO_FLOAT_KEYS = set()
_VIDEO_PATH_KEYS = {
    "semantic_feature_file",
    "stage2_semantic_feature_file",
    "model_path_t1",
}


def normalize_add_name(add_name):
    suffix = str(add_name or "").strip()
    if not suffix:
        return ""
    if "/" in suffix or "\\" in suffix:
        raise ValueError(f"add_name must be a folder-name suffix, not a path: {add_name!r}")
    if not suffix.startswith(("_", "-")):
        suffix = "_" + suffix
    return suffix


def build_ckpt_exp_name(exp_name, args=None):
    add_name = getattr(args, "add_name", "") if args is not None else ""
    return f"{exp_name}{normalize_add_name(add_name)}"


def as_bool(value, default=False):
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    value_text = str(value).strip().lower()
    if value_text in _TRUE_TEXT:
        return True
    if value_text in _FALSE_TEXT:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value}")


def _optional_int(value):
    if value is None or value == "":
        return None
    return int(value)


def _normalize_requested_stages(stages):
    normalized = []
    for stage in stages or ():
        if stage not in _VIDEO_STAGE_KEYS:
            raise ValueError(f"Unsupported stage name: {stage}")
        if stage not in normalized:
            normalized.append(stage)
    return tuple(normalized)


def _coerce_checkpoint_setting_value(key, value):
    if key in _VIDEO_BOOL_KEYS:
        return as_bool(value, default=False)
    if key in _VIDEO_INT_KEYS:
        return int(value)
    if key in _VIDEO_FLOAT_KEYS:
        return float(value)
    return value


def _resolve_checkpoint_path_setting(key, value, repo_root=None):
    if not value:
        return value
    value_text = str(value)
    if not os.path.isabs(value_text):
        if repo_root:
            return os.path.abspath(os.path.join(repo_root, value_text))
        return value_text
    if os.path.exists(value_text):
        return value_text

    warnings.warn(
        f"Ignoring checkpoint setting {key} because the absolute path no longer exists: {value_text}"
    )
    return None


def apply_video_checkpoint_settings(
    args_obj,
    checkpoint_cfg,
    stages=("stage1", "stage2"),
    include_runtime_settings=False,
    skip_keys=None,
    repo_root=None,
):
    if args_obj is None or not isinstance(checkpoint_cfg, dict) or not checkpoint_cfg:
        return []

    requested_stages = _normalize_requested_stages(stages)
    skipped_keys = set(skip_keys or ())
    setting_keys = []
    for stage in requested_stages:
        setting_keys.extend(_VIDEO_STAGE_KEYS[stage])

    if include_runtime_settings:
        setting_keys.extend(_VIDEO_RUNTIME_KEYS)
        if "stage2" in requested_stages:
            setting_keys.extend(_VIDEO_STAGE2_RUNTIME_KEYS)

    updates = []
    for key in setting_keys:
        if key in skipped_keys or key not in checkpoint_cfg:
            continue

        new_value = checkpoint_cfg.get(key)
        if key in _VIDEO_PATH_KEYS:
            new_value = _resolve_checkpoint_path_setting(key, new_value, repo_root=repo_root)
            if new_value is None:
                continue
        new_value = _coerce_checkpoint_setting_value(key, new_value)

        old_value = getattr(args_obj, key, None)
        setattr(args_obj, key, new_value)
        if old_value != new_value:
            updates.append({"name": key, "old": old_value, "new": new_value})

    return updates


def format_video_setting_updates(updates):
    if not updates:
        return "no updates"

    formatted = []
    for item in updates:
        name = item["name"]
        old_value = item["old"]
        new_value = item["new"]
        if old_value is None:
            formatted.append(f"{name}={new_value!r}")
        else:
            formatted.append(f"{name}: {old_value!r} -> {new_value!r}")
    return ", ".join(formatted)


def validate_dct_checkpoint_settings(cfg, checkpoint_cfg, context="checkpoint"):
    if not isinstance(checkpoint_cfg, dict) or not checkpoint_cfg:
        return

    if "dct_n" not in checkpoint_cfg:
        raise ValueError(f"{context} does not contain dct_n metadata.")

    checkpoint_n = int(checkpoint_cfg["dct_n"])
    current_n = int(getattr(cfg, "dct_n", 20))
    if checkpoint_n != current_n:
        raise ValueError(
            f"{context} DCT settings do not match the current run configuration. "
            f"dct_n: checkpoint={checkpoint_n}, current={current_n}"
        )


def _normalize_semantic_feature_path(repo_root, base_data_dir, semantic_feature_file):
    if not semantic_feature_file:
        return ""
    if os.path.isabs(semantic_feature_file):
        return semantic_feature_file
    relative_to_base = os.path.join(base_data_dir, semantic_feature_file)
    if os.path.exists(relative_to_base):
        return relative_to_base
    return os.path.abspath(os.path.join(repo_root, semantic_feature_file))


def _find_default_semantic_feature_file(base_data_dir, data_file, preferred_scope=""):
    data_stem = os.path.splitext(os.path.basename(data_file))[0]
    candidates = []
    for name in os.listdir(base_data_dir):
        if not name.endswith(".rfeat.npz"):
            continue
        if not name.startswith(f"{data_stem}_"):
            continue
        if preferred_scope and preferred_scope not in name:
            continue
        candidates.append(os.path.join(base_data_dir, name))

    if preferred_scope and not candidates:
        return _find_default_semantic_feature_file(base_data_dir=base_data_dir, data_file=data_file)

    if not candidates:
        return ""
    if len(candidates) > 1:
        candidates.sort(key=os.path.getmtime, reverse=True)
        warnings.warn(
            "Multiple semantic feature caches match the configured dataset. "
            f"Using the newest cache: {os.path.basename(candidates[0])}"
        )
    else:
        candidates.sort()
    return candidates[0]


def apply_common_video_config(cfg, args=None):
    requested_dct_n = _optional_int(getattr(args, "dct_n", None)) if args is not None else None
    if requested_dct_n is not None:
        cfg.dct_n = requested_dct_n
    cfg.dct_n = int(getattr(cfg, "dct_n", 20))

    cfg.stage2_use_video = as_bool(getattr(args, "stage2_use_video", False), default=False)
    cfg.stage2_video_in_condition = cfg.stage2_use_video and as_bool(
        getattr(args, "stage2_video_in_condition", False),
        default=False,
    )
    cfg.stage2_video_condition_mode = str(
        getattr(args, "stage2_video_condition_mode", None)
        or getattr(cfg, "stage2_video_condition_mode", "cat")
    ).strip().lower()
    if cfg.stage2_video_condition_mode not in {"cat", "film"}:
        raise ValueError(
            f"Unsupported stage2_video_condition_mode={cfg.stage2_video_condition_mode!r}. "
            "Expected 'cat' or 'film'."
        )
    cfg.stage2_video_in_gaussian = cfg.stage2_use_video and as_bool(
        getattr(args, "stage2_video_in_gaussian", False),
        default=False,
    )
    cfg.stage2_video_active = bool(
        cfg.stage2_use_video
        and (cfg.stage2_video_in_condition or cfg.stage2_video_in_gaussian)
    )
    cfg.video_required = cfg.stage2_video_active

    cfg.semantic_input_dim = int(getattr(args, "semantic_input_dim", 512) or 512)
    cfg.stage2_video_feat_dim = _optional_int(getattr(args, "stage2_video_feat_dim", None)) or 128

    configured_data_file = getattr(args, "data_file", "") if args is not None else ""
    cfg.data_file = configured_data_file or f"data_3d_pvcp_video_m{cfg.min_frames}.npz"
    cfg.data_path = os.path.join(cfg.base_data_dir, cfg.data_file)

    configured_semantic_path = getattr(args, "semantic_feature_file", "") if args is not None else ""
    configured_stage2_semantic_path = (
        getattr(args, "stage2_semantic_feature_file", "") if args is not None else ""
    ) or configured_semantic_path
    cfg.semantic_feature_file = _normalize_semantic_feature_path(
        repo_root=cfg.repo_root,
        base_data_dir=cfg.base_data_dir,
        semantic_feature_file=configured_semantic_path,
    )
    cfg.stage2_semantic_feature_file = _normalize_semantic_feature_path(
        repo_root=cfg.repo_root,
        base_data_dir=cfg.base_data_dir,
        semantic_feature_file=configured_stage2_semantic_path,
    )
    if not cfg.stage2_semantic_feature_file and cfg.stage2_video_active:
        cfg.stage2_semantic_feature_file = _find_default_semantic_feature_file(
            base_data_dir=cfg.base_data_dir,
            data_file=cfg.data_file,
            preferred_scope="hist10",
        )
    if not cfg.semantic_feature_file:
        cfg.semantic_feature_file = cfg.stage2_semantic_feature_file

    return cfg


def build_video_settings_summary(cfg):
    return {
        "data_file": cfg.data_file,
        "dct_n": int(cfg.dct_n),
        "stage2_use_video": bool(cfg.stage2_use_video),
        "stage2_video_in_condition": bool(cfg.stage2_video_in_condition),
        "stage2_video_condition_mode": cfg.stage2_video_condition_mode,
        "stage2_video_in_gaussian": bool(cfg.stage2_video_in_gaussian),
        "stage2_video_active": bool(cfg.stage2_video_active),
        "semantic_input_dim": int(cfg.semantic_input_dim),
        "stage2_video_feat_dim": int(cfg.stage2_video_feat_dim),
        "semantic_feature_file": cfg.semantic_feature_file,
        "stage2_semantic_feature_file": cfg.stage2_semantic_feature_file,
    }
