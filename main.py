#!/usr/bin/env python
# encoding: utf-8
'''
@project : pvcpm
@file    : main.py
@author  : wmj
@ide     : PyCharm
@time    : 2026-03-05
'''
# ****************************************************************************************************************
# *********************************************** Environments ***************************************************
# ****************************************************************************************************************

import argparse
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import numpy as np


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

# ****************************************************************************************************************
# *********************************************** Main ***********************************************************
# ****************************************************************************************************************


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value: {value}")


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument(
    '--exp_name',
    type=str,
    default="humaneva_t2",
    help="h36m_t1 / h36m_t2 / humaneva_t1 / humaneva_t2 / pvcp_t1 / pvcp_t2 / pvcp_video_t1 / pvcp_video_t2",
)
parser.add_argument(
    '--add_name',
    type=str,
    default="",
    help="Optional suffix appended to the PVCP-video checkpoint folder, e.g. dct20 -> ckpt/pvcp_video_t1_dct20.",
)
parser.add_argument('--is_train', type=bool, default='', help="")
parser.add_argument('--is_load', type=bool, default='', help="")
parser.add_argument('--is_debug', type=bool, default='', help="")
parser.add_argument('--seed', type=int, default=1, help="Random seed for Python, NumPy, and PyTorch.")

parser.add_argument('--model_path', type=str, default="", help="")
parser.add_argument('--model_path_t1', type=str, default="", help="Stage-1 checkpoint path used by pvcp_video_t2.")

parser.add_argument('--dct_n', type=int, default=None, help="PVCP-video DCT coefficient count.")

parser.add_argument('--stage2_use_video', type=str2bool, default=True, help="Whether Stage 2 uses semantic video features.")
parser.add_argument(
    '--stage2_video_in_condition',
    type=str2bool,
    default=False,
    help="Whether Stage 2 injects semantic video features into the condition branch.",
)
parser.add_argument(
    '--stage2_video_condition_mode',
    type=str,
    default="film",
    choices=["", "cat", "film"],
    help="How Stage 2 uses semantic video in the condition branch: cat or film.",
)
parser.add_argument(
    '--stage2_video_in_gaussian',
    type=str2bool,
    default=True,
    help="Whether Stage 2 uses semantic video features to predict basis-selection logits W.",
)
parser.add_argument('--semantic_feature_file', type=str, default="", help="Optional semantic feature cache (.rfeat.npz).")
parser.add_argument(
    '--stage2_semantic_feature_file',
    type=str,
    default="dataset/pvcp/data_3d_pvcp_video_m35_Qwen3.5-9B_ped_pose_ego_vehicle_hist10_d512.rfeat.npz",
    help="Stage 2 semantic feature cache (.rfeat.npz).",
)
parser.add_argument('--semantic_input_dim', type=int, default=512, help="Window-level semantic feature dimension.")
parser.add_argument('--stage2_video_feat_dim', type=int, default=None, help="Optional Stage 2 semantic projection dimension.")

args = parser.parse_args()
seed_torch(args.seed)


def _use_default_model_path(args_obj, default_name):
    if not args_obj.model_path:
        args_obj.model_path = os.path.join(r"./ckpt/pretrained", default_name)


def _load_checkpoint_cfg_snapshot(checkpoint_path):
    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected checkpoint to be a dict, got {type(state).__name__}: {checkpoint_path}")
    checkpoint_cfg = state.get("cfg_snapshot", {})
    return checkpoint_cfg if isinstance(checkpoint_cfg, dict) else {}


def _auto_apply_video_checkpoint_settings(args_obj, checkpoint_path, stages):
    from pvcp_video.configs.common import apply_video_checkpoint_settings, format_video_setting_updates

    checkpoint_cfg = _load_checkpoint_cfg_snapshot(checkpoint_path)
    updates = apply_video_checkpoint_settings(
        args_obj,
        checkpoint_cfg,
        stages=stages,
        include_runtime_settings=True,
    )
    if updates:
        print(
            "Auto-applied video checkpoint settings from {}: {}".format(
                os.path.abspath(checkpoint_path),
                format_video_setting_updates(updates),
            )
        )


def _format_checkpoint_epoch(epoch):
    return "unknown" if epoch is None else str(epoch)


def _format_loaded_checkpoint_epochs(run_obj):
    checkpoint_infos = []
    for label, path_attr, epoch_attr in (
        ("t1", "stage1_checkpoint_path", "stage1_checkpoint_epoch"),
        ("t2", "stage2_checkpoint_path", "stage2_checkpoint_epoch"),
    ):
        checkpoint_path = getattr(run_obj, path_attr, None)
        if not checkpoint_path:
            continue
        checkpoint_epoch = _format_checkpoint_epoch(getattr(run_obj, epoch_attr, None))
        checkpoint_infos.append(f"{label}: {checkpoint_path} (epoch {checkpoint_epoch})")

    if not checkpoint_infos:
        return ""
    return "Checkpoint epochs --> " + " -- ".join(checkpoint_infos)


def _print_loaded_checkpoint_epochs(run_obj):
    checkpoint_text = _format_loaded_checkpoint_epochs(run_obj)
    if checkpoint_text:
        print(" " + checkpoint_text)


def _save_eval_result_txt(run_obj, result_text, extra_text=""):
    ckpt_dir = getattr(getattr(run_obj, "cfg", None), "ckpt_dir", None)
    if not ckpt_dir:
        return
    os.makedirs(ckpt_dir, exist_ok=True)
    result_path = os.path.join(ckpt_dir, "eval_result.txt")
    save_text = result_text.rstrip() + "\n" + f"Saved eval result to {result_path}"
    if extra_text:
        save_text += "\n" + extra_text.rstrip()
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(save_text.rstrip() + "\n")
    print(f" Saved eval result to {result_path}")


def _format_best_result(run_obj):
    metrics = getattr(run_obj, "best_metrics", None)
    if not metrics:
        return ""
    epoch = metrics.get("epoch", "unknown")
    metric_name = getattr(run_obj, "best_metric_name", "score")
    return (
        "Best --> epo {}: {} {:.6f} | div {:.4f} | ade {:.4f} | fde {:.4f} | "
        "mmade {:.4f} | mmfde {:.4f}".format(
            epoch,
            metric_name,
            float(metrics.get("score", 0.0)),
            float(metrics.get("diversity", 0.0)),
            float(metrics.get("ade", 0.0)),
            float(metrics.get("fde", 0.0)),
            float(metrics.get("mmade", 0.0)),
            float(metrics.get("mmfde", 0.0)),
        )
    )


def _save_best_result_txt(run_obj):
    result_text = _format_best_result(run_obj)
    if not result_text:
        return
    print("\n " + result_text)
    ckpt_dir = getattr(getattr(run_obj, "cfg", None), "ckpt_dir", None)
    if not ckpt_dir:
        return
    os.makedirs(ckpt_dir, exist_ok=True)
    result_path = os.path.join(ckpt_dir, "best_result.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result_text.rstrip() + "\n")
    print(f" Saved best result to {result_path}")


if args.exp_name == "h36m_t1":
    from h36m.runs import RunCVAE as RunCVAEH36m

    _use_default_model_path(args, "h36m_t1.pth")
    r = RunCVAEH36m(exp_name=args.exp_name, is_debug=args.is_debug, args=args, device="cuda:0", num_works=0)

elif args.exp_name == "h36m_t2":
    from h36m.runs import RunDiverseSampling as RunDiverseSamplingH36m

    _use_default_model_path(args, "h36m_t2.pth")
    r = RunDiverseSamplingH36m(exp_name=args.exp_name, is_debug=args.is_debug, args=args, device="cuda:0", num_works=0)

elif args.exp_name == "humaneva_t1":
    from humaneva.runs import RunCVAE as RunCVAEHumaneva

    _use_default_model_path(args, "humaneva_t1.pth")
    r = RunCVAEHumaneva(exp_name=args.exp_name, is_debug=args.is_debug, args=args, device="cuda:0", num_works=0)

elif args.exp_name == "humaneva_t2":
    from humaneva.runs import RunDiverseSampling as RunDiverseSamplingHumaneva

    _use_default_model_path(args, "humaneva_t2.pth")
    r = RunDiverseSamplingHumaneva(exp_name=args.exp_name, is_debug=args.is_debug, args=args, device="cuda:0", num_works=0)

elif args.exp_name == "pvcp_t1":
    from pvcp.runs import RunCVAE as RunCVAEPVCP

    _use_default_model_path(args, "pvcp_t1.pth")
    r = RunCVAEPVCP(exp_name=args.exp_name, is_debug=args.is_debug, args=args, device="cuda:0", num_works=0)

elif args.exp_name == "pvcp_t2":
    from pvcp.runs import RunDiverseSampling as RunDiverseSamplingPVCP

    _use_default_model_path(args, "pvcp_t2.pth")
    r = RunDiverseSamplingPVCP(exp_name=args.exp_name, is_debug=args.is_debug, args=args, device="cuda:0", num_works=0)

elif args.exp_name == "pvcp_video_t1":
    from pvcp_video.runs import RunCVAE as RunCVAEPVCPVideo

    _use_default_model_path(args, "pvcp_video_t1_best.pth")
    r = RunCVAEPVCPVideo(exp_name=args.exp_name, is_debug=args.is_debug, args=args, device="cuda:0", num_works=0)

elif args.exp_name == "pvcp_video_t2":
    from pvcp_video.runs import RunDiverseSampling as RunDiverseSamplingPVCPVideo

    _use_default_model_path(args, "pvcp_video_t2_best.pth")
    if args.is_load:
        _auto_apply_video_checkpoint_settings(args, args.model_path, stages=("stage2",))
    r = RunDiverseSamplingPVCPVideo(exp_name=args.exp_name, is_debug=args.is_debug, args=args, device="cuda:0", num_works=0)

else:
    print("wrong exp_name!")


if args.is_load:
    r.restore(args.model_path)

if args.is_train:
    r.run()
    _save_best_result_txt(r)

else:
    diversity, ade, fde, mmade, mmfde = r.eval(epoch=-1, draw=True)
    if args.exp_name in {"pvcp_t2", "pvcp_video_t2"} and getattr(r, "last_eval_metrics", None):
        short_metrics = r.last_eval_metrics["short_term"]
        long_metrics = r.last_eval_metrics["long_term"]
        short_frames = r.last_eval_metrics["short_term_frames"]
        eval_result_text = (
            "Test --> short-term(first {} pred frames): div {:.4f} -- ade {:.4f} --  fde {:.4f} --  mmade {:.4f} --  mmfde {:.4f}\n"
            "Test --> long-term(all pred frames):       div {:.4f} -- ade {:.4f} --  fde {:.4f} --  mmade {:.4f} --  mmfde {:.4f}\n"
            "Inference time: {:.3f} ms/sample".format(
                short_frames,
                short_metrics["diversity"],
                short_metrics["ade"],
                short_metrics["fde"],
                short_metrics["mmade"],
                short_metrics["mmfde"],
                long_metrics["diversity"],
                long_metrics["ade"],
                long_metrics["fde"],
                long_metrics["mmade"],
                long_metrics["mmfde"],
                float(r.last_eval_metrics.get("inference_time_ms_per_sample", 0.0)),
            )
        )
        print("\n " + eval_result_text)
    else:
        eval_result_text = (
            "Test -->  div {:.4f} -- ade {:.4f} --  fde {:.4f} --  mmade {:.4f} --  mmfde {:.4f}".format(
                diversity,
                ade,
                fde,
                mmade,
                mmfde,
            )
        )
        print("\n " + eval_result_text)
    checkpoint_text = _format_loaded_checkpoint_epochs(r)
    _save_eval_result_txt(r, eval_result_text, extra_text=checkpoint_text)
    _print_loaded_checkpoint_epochs(r)
