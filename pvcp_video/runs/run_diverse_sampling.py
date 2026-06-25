#!/usr/bin/env python
# encoding: utf-8

import glob
import math
import os
import random
import time

import numpy as np
import torch
from torch.optim import Adam, lr_scheduler

try:
    from tensorboardX import SummaryWriter
except ImportError:
    from torch.utils.tensorboard import SummaryWriter

from .json_utils import dump_json_with_inline_scalar_lists, format_json_with_inline_scalar_lists
from .losses import (
    compute_ade,
    compute_diversity,
    compute_fde,
    compute_mmade,
    compute_mmfde,
    loss_acceleration_adelike,
    loss_diversity_hinge_divide,
    loss_kl_normal,
    loss_recons_adelike,
    loss_recons_mmadelike,
    loss_velocity_adelike,
)

from ..configs import ConfigDiverseSampling
from ..configs.common import build_video_settings_summary, validate_dct_checkpoint_settings
from ..datas import (
    MaoweiGSPS_Dynamic_Seq_PVCP_Video,
    dct_transform_torch,
    draw_multi_seqs_2d,
    get_dct_matrix,
    reverse_dct_torch,
)
from ..nets import CVAE, DiverseSampling


class _NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None


class RunDiverseSampling:
    def __init__(self, exp_name="pvcp_video_t2", device="cuda:0", num_works=0, is_debug=False, args=None):
        super().__init__()

        self.is_debug = is_debug
        self.start_epoch = 1
        self.best_accuracy = 1e15
        self.best_metrics = None
        self.best_metric_name = "ade+fde"
        self.last_eval_metrics = None
        self.cfg = ConfigDiverseSampling(exp_name=exp_name, device=device, num_works=num_works, args=args)
        self.draw_interval = 50
        self.stage1_checkpoint_path = None
        self.stage1_checkpoint_epoch = None
        self.stage2_checkpoint_path = None
        self.stage2_checkpoint_epoch = None
        self.dct_m, self.i_dct_m = get_dct_matrix(self.cfg.t_total)

        print("\n================== Arguments =================")
        print(format_json_with_inline_scalar_lists(vars(args) if args is not None else {}))
        print("==========================================\n")

        print("\n================== Configs =================")
        print(format_json_with_inline_scalar_lists(vars(self.cfg)))
        print("==========================================\n")

        print("\n================== Video Settings =================")
        print(format_json_with_inline_scalar_lists(build_video_settings_summary(self.cfg)))
        print("===============================================\n")

        self._prepare_ckpt_dir()
        self._save_config(args)

        self.model_t1 = CVAE(
            node_n=self.cfg.node_n,
            hidden_dim=self.cfg.hidden_dim,
            z_dim=self.cfg.z_dim,
            dct_n=self.cfg.dct_n,
            dropout_rate=self.cfg.dropout_rate,
        )
        self.model = DiverseSampling(
            node_n=self.cfg.node_n,
            hidden_dim=self.cfg.hidden_dim,
            base_dim=self.cfg.base_dim,
            base_num_p1=self.cfg.base_num_p1,
            z_dim=self.cfg.z_dim,
            dct_n=self.cfg.dct_n,
            dropout_rate=self.cfg.dropout_rate,
            use_video=self.cfg.stage2_video_active,
            video_feat_dim=self.cfg.stage2_video_feat_dim,
            semantic_input_dim=self.cfg.semantic_input_dim,
            video_in_condition=self.cfg.stage2_video_in_condition,
            video_in_gaussian=self.cfg.stage2_video_in_gaussian,
            video_condition_mode=self.cfg.stage2_video_condition_mode,
        )
        if self.cfg.device != "cpu":
            self.model_t1.cuda(self.cfg.device)
            self.model.cuda(self.cfg.device)

        print(
            ">>> total params of {}: {:.6f}M\n".format(
                "t1", sum(p.numel() for p in self.model_t1.parameters()) / 1000000.0
            )
        )
        print(
            ">>> total params of {}: {:.6f}M\n".format(
                exp_name, sum(p.numel() for p in self.model.parameters()) / 1000000.0
            )
        )

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.cfg.lr_t2)

        def lambda_rule(epoch):
            return 1.0 - max(0, epoch - self.cfg.epoch_fix_t2) / float(
                self.cfg.epoch_t2 - self.cfg.epoch_fix_t2 + 1
            )

        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)

        model_path_t1 = self._resolve_t1_checkpoint_path()
        model_t1_state = torch.load(model_path_t1, map_location=self.cfg.device)
        self.stage1_checkpoint_path = model_path_t1
        self.stage1_checkpoint_epoch = model_t1_state.get("epoch")
        self._validate_stage1_checkpoint(model_t1_state)
        try:
            self.model_t1.load_state_dict(model_t1_state["model"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load PVCP-video stage-1 checkpoint. "
                f"Please make sure the checkpoint was trained without Stage 1 video conditioning: {model_path_t1}"
            ) from exc
        print(f"model_t1 loaded from {model_path_t1}")
        for p in self.model_t1.parameters():
            p.requires_grad = False
        self.model_t1.eval()

        load_video = self.cfg.stage2_video_active
        self.train_data = MaoweiGSPS_Dynamic_Seq_PVCP_Video(
            data_path=self.cfg.base_data_dir,
            data_file=self.cfg.data_file,
            scene_split_path=self.cfg.scene_split_path,
            test_ratio=self.cfg.test_ratio,
            t_his=self.cfg.t_his,
            t_pred=self.cfg.t_pred,
            similar_cnt=self.cfg.train_similar_cnt,
            dynamic_sub_len=self.cfg.sub_len_train,
            batch_size=self.cfg.train_batch_size,
            joint_used=self.cfg.joint_used,
            parents=self.cfg.parents,
            mode="train",
            multimodal_threshold=self.cfg.multimodal_threshold,
            sample_step_train=self.cfg.sample_step_train,
            sample_step_test=self.cfg.sample_step_test,
            is_debug=self.is_debug,
            load_video=load_video,
            stage2_semantic_feature_file=self.cfg.stage2_semantic_feature_file,
            semantic_input_dim=self.cfg.semantic_input_dim,
        )
        self.test_data = MaoweiGSPS_Dynamic_Seq_PVCP_Video(
            data_path=self.cfg.base_data_dir,
            data_file=self.cfg.data_file,
            scene_split_path=self.cfg.scene_split_path,
            test_ratio=self.cfg.test_ratio,
            t_his=self.cfg.t_his,
            t_pred=self.cfg.t_pred,
            similar_cnt=0,
            dynamic_sub_len=self.cfg.sub_len_train,
            batch_size=self.cfg.test_batch_size,
            joint_used=self.cfg.joint_used,
            parents=self.cfg.parents,
            mode="test",
            multimodal_threshold=self.cfg.multimodal_threshold,
            sample_step_train=self.cfg.sample_step_train,
            sample_step_test=self.cfg.sample_step_test,
            is_debug=self.is_debug,
            load_video=load_video,
            stage2_semantic_feature_file=self.cfg.stage2_semantic_feature_file,
            semantic_input_dim=self.cfg.semantic_input_dim,
        )
        self.test_data.get_test_similat_gt_like_dlow()

        self.dct_m = torch.from_numpy(self.dct_m).float().to(self.cfg.device)
        self.i_dct_m = torch.from_numpy(self.i_dct_m).float().to(self.cfg.device)

        self.summary = self._create_summary_writer()

    def _prepare_ckpt_dir(self):
        for sub_path in ("", "models", "images", os.path.join("images", "sample")):
            try:
                os.makedirs(os.path.join(self.cfg.ckpt_dir, sub_path), exist_ok=True)
            except OSError:
                pass

    def _save_config(self, args):
        try:
            with open(os.path.join(self.cfg.ckpt_dir, "config.json"), "w", encoding="utf-8") as f:
                dump_json_with_inline_scalar_lists(
                    {"args": vars(args) if args is not None else {}, "cfgs": self.cfg.__dict__},
                    f,
                    indent=4,
                )
        except OSError:
            pass

    def _create_summary_writer(self):
        try:
            return SummaryWriter(self.cfg.ckpt_dir)
        except Exception:
            return _NullSummaryWriter()

    def _best_stage_checkpoint_path(self):
        return os.path.join(self.cfg.ckpt_dir, "models", f"{self.cfg.exp_name}_best.pth")

    def _last_stage_checkpoint_path(self):
        return os.path.join(self.cfg.ckpt_dir, "models", f"{self.cfg.exp_name}_last.pth")

    def _add_root_for_draw(self, data):
        data = data.reshape(data.shape[0], self.cfg.node_n, 3, data.shape[-1])
        root = np.zeros((data.shape[0], 1, 3, data.shape[-1]), dtype=data.dtype)
        return np.concatenate((root, data), axis=1)

    def _draw_history_count(self):
        return max(1, int(math.ceil(self.cfg.t_his / 2.0)))

    def _project_oblique_view_for_draw(self, data_3d):
        angle = np.deg2rad(getattr(self.cfg, "draw_yaw_deg", -45.0))
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

    def _batch_to_device(self, batch):
        pose = torch.from_numpy(batch["pose"]).float().to(self.cfg.device)
        stage2_semantic_features = batch.get("stage2_semantic_features")
        if stage2_semantic_features is None:
            stage2_semantic_features = batch.get("history_semantic_features")
        if stage2_semantic_features is not None:
            stage2_semantic_features = torch.from_numpy(stage2_semantic_features).float().to(self.cfg.device)
        similar_pose = batch.get("similar_pose")
        if similar_pose is not None:
            similar_pose = torch.from_numpy(similar_pose).float().to(self.cfg.device)
        return pose, stage2_semantic_features, similar_pose

    def _select_video_feature_batch(self, semantic_features, use_video):
        if not use_video:
            return None
        return semantic_features

    def _validate_stage1_checkpoint(self, checkpoint_state):
        checkpoint_cfg = checkpoint_state.get("cfg_snapshot", {})
        if not checkpoint_cfg:
            validate_dct_checkpoint_settings(self.cfg, checkpoint_cfg, context="Stage 1 checkpoint")
            return
        validate_dct_checkpoint_settings(self.cfg, checkpoint_cfg, context="Stage 1 checkpoint")

    def _resolve_t1_checkpoint_path(self):
        configured_path = self.cfg.model_path_t1
        if configured_path and os.path.exists(configured_path):
            return configured_path

        repo_root = self.cfg.repo_root
        candidate_dirs = [
            os.path.join(repo_root, "ckpt", "pvcp_video_t1", "models"),
            os.path.join(repo_root, "ckpt", "pvcp_video_t1"),
        ]
        candidate_dirs.extend(
            [
                os.path.join(repo_root, "ckpt", "pvcp_t1", "models"),
                os.path.join(repo_root, "ckpt", "pvcp_t1"),
                os.path.join(repo_root, "ckpt", "pretrained"),
            ]
        )

        candidates = []
        for candidate_dir in candidate_dirs:
            if not os.path.isdir(candidate_dir):
                continue
            candidates.extend(glob.glob(os.path.join(candidate_dir, "*.pth")))

        if candidates:
            candidates = sorted(candidates, key=os.path.getmtime, reverse=True)
            resolved_path = candidates[0]
            print(
                "configured stage-1 checkpoint not found, fallback to latest checkpoint under "
                f"candidate dirs: {resolved_path}"
            )
            return resolved_path

        raise FileNotFoundError(
            "PVCP-video stage-2 needs a trained t1 checkpoint, but none was found. "
            f"Checked configured path: {configured_path} and searched under {candidate_dirs}."
        )

    def save(self, checkpoint_path, epoch, curr_err, metrics=None):
        state = {
            "epoch": epoch,
            "lr": self.scheduler.get_last_lr()[0],
            "curr_err": curr_err,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics or {},
            "cfg_snapshot": dict(self.cfg.__dict__),
        }
        torch.save(state, checkpoint_path)
        print(f"saved to {checkpoint_path}")

    def _build_eval_metrics(self, diversity, ade, fde, mmade, mmfde):
        return {
            "diversity": float(diversity),
            "ade": float(ade),
            "fde": float(fde),
            "mmade": float(mmade),
            "mmfde": float(mmfde),
            "score": float(ade + fde),
        }

    def _compute_eval_metrics_for_horizon(self, pred, gt, similars):
        return {
            "diversity": float(compute_diversity(pred=pred).mean().item()),
            "ade": float(compute_ade(pred, gt).item()),
            "fde": float(compute_fde(pred, gt).item()),
            "mmade": float(compute_mmade(pred, gt, similars).item()),
            "mmfde": float(compute_mmfde(pred, gt, similars).item()),
        }

    def _zero_eval_metrics(self):
        return {
            "diversity": 0.0,
            "ade": 0.0,
            "fde": 0.0,
            "mmade": 0.0,
            "mmfde": 0.0,
        }

    def _average_eval_metrics(self, metrics_sum, count):
        if count <= 0:
            return self._zero_eval_metrics()
        return {key: float(value) / float(count) for key, value in metrics_sum.items()}

    def _maybe_save_best(self, epoch, diversity, ade, fde, mmade, mmfde):
        metrics = self._build_eval_metrics(diversity, ade, fde, mmade, mmfde)
        score = metrics["score"]

        if epoch <= 30:
            self.summary.add_scalar("Best/score", self.best_accuracy, epoch)
            return

        if score < self.best_accuracy:
            previous_best = self.best_accuracy
            metrics["epoch"] = int(epoch)
            self.best_accuracy = score
            self.best_metrics = metrics
            self.save(self._best_stage_checkpoint_path(), epoch, score, metrics=metrics)
            previous_best_text = "inf" if math.isinf(previous_best) else f"{previous_best:.6f}"
            print(
                "best model updated at epoch {}: {} {} -> {:.6f} "
                "(div {:.4f} | ade {:.4f} | fde {:.4f} | mmade {:.4f} | mmfde {:.4f})".format(
                    epoch,
                    self.best_metric_name,
                    previous_best_text,
                    score,
                    metrics["diversity"],
                    metrics["ade"],
                    metrics["fde"],
                    metrics["mmade"],
                    metrics["mmfde"],
                )
            )

        self.summary.add_scalar("Best/score", self.best_accuracy, epoch)
        if self.best_metrics is not None:
            self.summary.add_scalar("Best/div", self.best_metrics["diversity"], epoch)
            self.summary.add_scalar("Best/ade", self.best_metrics["ade"], epoch)
            self.summary.add_scalar("Best/fde", self.best_metrics["fde"], epoch)
            self.summary.add_scalar("Best/mmade", self.best_metrics["mmade"], epoch)
            self.summary.add_scalar("Best/mmfde", self.best_metrics["mmfde"], epoch)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=self.cfg.device)
        self.stage2_checkpoint_path = checkpoint_path
        self.stage2_checkpoint_epoch = state.get("epoch")
        validate_dct_checkpoint_settings(self.cfg, state.get("cfg_snapshot", {}), context="Stage 2 checkpoint")
        self.model.load_state_dict(state["model"])
        print(f"load from {checkpoint_path}")

    def train(self, epoch, draw=False):
        self.model.train()
        self.model_t1.eval()
        average_allloss = 0.0
        average_kls_p1 = 0.0
        average_adeerrors = 0.0
        average_hinges = 0.0
        average_similar_loss = 0.0
        average_velocity_loss = 0.0
        average_acceleration_loss = 0.0

        dg = self.train_data.batch_generator()
        generator_len = (
            self.cfg.sub_len_train // self.train_data.batch_size if not self.is_debug else 200 // self.train_data.batch_size
        )
        draw_i = random.randint(0, max(0, generator_len - 1))

        last_i = -1
        for i, batch in enumerate(dg):
            last_i = i
            datas, stage2_semantic_features, similars = self._batch_to_device(batch)
            b, _, _ = datas.shape
            if b == 1:
                continue

            self.global_step = (epoch - 1) * generator_len + i + 1
            stage2_video_features = self._select_video_feature_batch(
                semantic_features=stage2_semantic_features,
                use_video=self.cfg.stage2_video_active,
            )
            eps = torch.randn((b, self.cfg.z_dim), device=self.cfg.device)
            repeated_eps = torch.repeat_interleave(eps, repeats=self.cfg.nk, dim=0)

            with torch.no_grad():
                padded_inputs = datas[:, :, list(range(self.cfg.t_his)) + [self.cfg.t_his - 1] * self.cfg.t_pred]
                padded_inputs_coeff = dct_transform_torch(
                    padded_inputs,
                    self.dct_m,
                    dct_n=self.cfg.dct_n,
                )
                padded_inputs_coeff = padded_inputs_coeff.view(b, -1, 3 * self.cfg.dct_n)

            all_z, all_mean_p1, all_logvar_p1 = self.model(
                condition=padded_inputs_coeff,
                repeated_eps=repeated_eps,
                many_weights=None,
                multi_modal_head=self.cfg.nk,
                video_features=stage2_video_features,
                temperature=self.cfg.temperature_p1,
            )

            all_outs_coeff = self.model_t1.inference(
                condition=torch.repeat_interleave(padded_inputs_coeff, repeats=self.cfg.nk, dim=0),
                z=all_z,
            )
            all_outs_coeff = all_outs_coeff.reshape(b * self.cfg.nk, -1, self.cfg.dct_n)
            outputs = reverse_dct_torch(all_outs_coeff, self.i_dct_m, self.cfg.t_total)
            outputs = outputs.view(b, self.cfg.nk, -1, self.cfg.t_total)

            pred_future = outputs[:, :, :, self.cfg.t_his :]
            gt_future = datas[:, :, self.cfg.t_his :]
            if similars.shape[-1] == datas.shape[-1]:
                similars_future = similars[:, :, :, self.cfg.t_his :]
            else:
                similars_future = similars

            kls_p1 = loss_kl_normal(all_mean_p1, all_logvar_p1)
            adeerrors = loss_recons_adelike(
                gt=gt_future,
                pred=pred_future,
            )
            all_hinges = loss_diversity_hinge_divide(
                pred_future,
                minthreshold=self.cfg.minthreshold,
                seperate_head=self.cfg.seperate_head,
            )
            similar_loss = loss_recons_mmadelike(
                pred=pred_future,
                similars=similars_future,
            )
            velocity_loss = loss_velocity_adelike(
                pred=pred_future,
                gt=gt_future,
            )
            acceleration_loss = loss_acceleration_adelike(
                pred=pred_future,
                gt=gt_future,
            )

            all_loss = (
                kls_p1 * self.cfg.t2_kl_p1_weight
                + adeerrors * self.cfg.t2_ade_weight
                + all_hinges * self.cfg.t2_diversity_weight
                + similar_loss * self.cfg.t2_similar_weight
                + velocity_loss * self.cfg.t2_velocity_weight
                + acceleration_loss * self.cfg.t2_acceleration_weight
            )

            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()

            average_allloss += all_loss.item()
            average_kls_p1 += kls_p1.item()
            average_adeerrors += adeerrors.item()
            average_hinges += all_hinges.item()
            average_similar_loss += similar_loss.item()
            average_velocity_loss += velocity_loss.item()
            average_acceleration_loss += acceleration_loss.item()

            if draw and i == draw_i:
                origin = datas[:1].detach().cpu().numpy()
                origin = self._add_root_for_draw(origin) * 1000.0

                output = outputs[0, : self.cfg.seperate_head].detach().cpu().numpy()
                output = self._add_root_for_draw(output) * 1000.0

                all_to_draw = np.concatenate((origin, output), axis=0)
                draw_acc = [acc for acc in range(0, all_to_draw.shape[-1], 2)]
                all_to_draw = all_to_draw[:, :, :, draw_acc]
                all_to_draw, x_period, vertical_period = self._project_oblique_view_for_draw(all_to_draw)

                draw_multi_seqs_2d(
                    all_to_draw,
                    gt_cnt=1,
                    t_his=self._draw_history_count(),
                    I=self.cfg.I_plot,
                    J=self.cfg.J_plot,
                    LR=self.cfg.LR_plot,
                    x_period=x_period,
                    z_period=vertical_period,
                    xlabel=f"view_x (yaw={self.cfg.draw_yaw_deg:.0f} deg)",
                    ylabel="y",
                    full_path=os.path.join(self.cfg.ckpt_dir, "images", f"train_epo{epoch}idx{draw_i}.png"),
                )

        denom = max(1, last_i + 1)
        average_allloss /= denom
        average_kls_p1 /= denom
        average_adeerrors /= denom
        average_hinges /= denom
        average_similar_loss /= denom
        average_velocity_loss /= denom
        average_acceleration_loss /= denom

        self.summary.add_scalar("loss/average_all", average_allloss, epoch)
        self.summary.add_scalar("loss/average_kls_p1", average_kls_p1, epoch)
        self.summary.add_scalar("loss/average_ades", average_adeerrors, epoch)
        self.summary.add_scalar("loss/average_hinges", average_hinges, epoch)
        self.summary.add_scalar("loss/average_similar", average_similar_loss, epoch)
        self.summary.add_scalar("loss/average_velocity", average_velocity_loss, epoch)
        self.summary.add_scalar("loss/average_acceleration", average_acceleration_loss, epoch)

        return (
            average_allloss,
            average_adeerrors,
            average_hinges,
            average_kls_p1,
            average_similar_loss,
            average_velocity_loss,
            average_acceleration_loss,
        )

    def eval(self, epoch=-1, draw=False):
        self.model.eval()
        self.model_t1.eval()

        long_term_sum = self._zero_eval_metrics()
        short_term_sum = self._zero_eval_metrics()
        valid_count = 0
        short_term_frames = min(10, self.cfg.t_pred)
        inference_time_total = 0.0
        inference_sample_count = 0
        sync_cuda_timing = self.cfg.device != "cpu" and torch.cuda.is_available()

        try:
            os.makedirs(os.path.join(self.cfg.ckpt_dir, "images", "sample"), exist_ok=True)
        except OSError:
            pass

        dg = self.test_data.onebyone_generator()
        generator_len = len(self.test_data.similat_gt_like_dlow) if not self.is_debug else min(90, len(self.test_data.similat_gt_like_dlow))
        if generator_len <= 0:
            self.last_eval_metrics = {
                "short_term_frames": short_term_frames,
                "short_term": self._zero_eval_metrics(),
                "long_term": self._zero_eval_metrics(),
                "inference_time_ms_per_sample": 0.0,
            }
            return 0, 0, 0, 0, 0

        draw_i = random.randint(0, max(0, generator_len - 1))

        for i, batch in enumerate(dg):
            similars = self.test_data.similat_gt_like_dlow[i]
            if similars.shape[0] == 0:
                continue

            datas, stage2_semantic_features, _ = self._batch_to_device(batch)
            stage2_video_features = self._select_video_feature_batch(
                semantic_features=stage2_semantic_features,
                use_video=self.cfg.stage2_video_active,
            )
            similars = torch.from_numpy(similars).float().to(self.cfg.device)

            if sync_cuda_timing:
                torch.cuda.synchronize()
            inference_start = time.perf_counter()
            with torch.no_grad():
                padded_inputs = datas[:, :, list(range(self.cfg.t_his)) + [self.cfg.t_his - 1] * self.cfg.t_pred]
                padded_inputs_coeff = dct_transform_torch(
                    padded_inputs,
                    self.dct_m,
                    dct_n=self.cfg.dct_n,
                )
                padded_inputs_coeff = padded_inputs_coeff.view(datas.shape[0], -1, 3 * self.cfg.dct_n)

                repeated_eps = torch.randn((datas.shape[0] * self.cfg.nk, self.cfg.z_dim), device=self.cfg.device)
                all_z_p1, _, _ = self.model(
                    condition=padded_inputs_coeff,
                    repeated_eps=repeated_eps,
                    many_weights=None,
                    multi_modal_head=self.cfg.nk,
                    video_features=stage2_video_features,
                    temperature=self.cfg.temperature_p1,
                )

                all_outs_coeff = self.model_t1.inference(
                    condition=torch.repeat_interleave(padded_inputs_coeff, repeats=self.cfg.nk, dim=0),
                    z=all_z_p1,
                )
                all_outs_coeff = all_outs_coeff.reshape(self.cfg.nk, -1, self.cfg.dct_n)
                outputs = reverse_dct_torch(all_outs_coeff, self.i_dct_m, self.cfg.t_total)
                outputs = outputs.view(self.cfg.nk, -1, self.cfg.t_total)
                outputs = outputs[:, :, self.cfg.t_his :]
                gt_future = datas[:, :, self.cfg.t_his :]
            if sync_cuda_timing:
                torch.cuda.synchronize()
            inference_time_total += time.perf_counter() - inference_start
            inference_sample_count += datas.shape[0]

            with torch.no_grad():
                current_long_term = self._compute_eval_metrics_for_horizon(outputs, gt_future, similars)
                current_short_term = self._compute_eval_metrics_for_horizon(
                    outputs[:, :, :short_term_frames],
                    gt_future[:, :, :short_term_frames],
                    similars[:, :, :short_term_frames],
                )

            for key, value in current_long_term.items():
                long_term_sum[key] += value
            for key, value in current_short_term.items():
                short_term_sum[key] += value
            valid_count += 1

            if epoch == -1:
                print(
                    "Test {} > it {}: div {:.4f} | ade {:.4f} | fde {:.4f} | mmade {:.4f} | mmfde {:.4f}".format(
                        all_z_p1.shape[0],
                        i,
                        current_long_term["diversity"],
                        current_long_term["ade"],
                        current_long_term["fde"],
                        current_long_term["mmade"],
                        current_long_term["mmfde"],
                    )
                )

            if draw and i == draw_i:
                origin = datas[:1].detach().cpu().numpy()
                origin = self._add_root_for_draw(origin) * 1000.0

                all_outputs = outputs.detach().cpu().numpy()
                all_outputs = self._add_root_for_draw(all_outputs) * 1000.0
                all_outputs = np.concatenate(
                    (np.repeat(origin[:, :, :, : self.cfg.t_his], repeats=self.cfg.nk, axis=0), all_outputs),
                    axis=-1,
                )

                all_to_draw = np.concatenate((origin, all_outputs), axis=0)
                draw_acc = [acc for acc in range(0, all_to_draw.shape[-1], 2)]
                all_to_draw = all_to_draw[:, :, :, draw_acc]
                all_to_draw, x_period, vertical_period = self._project_oblique_view_for_draw(all_to_draw)

                draw_multi_seqs_2d(
                    all_to_draw,
                    gt_cnt=1,
                    t_his=self._draw_history_count(),
                    I=self.cfg.I_plot,
                    J=self.cfg.J_plot,
                    LR=self.cfg.LR_plot,
                    x_period=x_period,
                    z_period=vertical_period,
                    xlabel=f"view_x (yaw={self.cfg.draw_yaw_deg:.0f} deg)",
                    ylabel="y",
                    full_path=os.path.join(self.cfg.ckpt_dir, "images", "sample", f"test_epo{epoch}idx{draw_i}.png"),
                )

        if valid_count == 0:
            self.last_eval_metrics = {
                "short_term_frames": short_term_frames,
                "short_term": self._zero_eval_metrics(),
                "long_term": self._zero_eval_metrics(),
                "inference_time_ms_per_sample": 0.0,
            }
            return 0, 0, 0, 0, 0

        short_term_metrics = self._average_eval_metrics(short_term_sum, valid_count)
        long_term_metrics = self._average_eval_metrics(long_term_sum, valid_count)
        inference_time_ms_per_sample = (
            inference_time_total / max(1, inference_sample_count) * 1000.0
        )
        self.last_eval_metrics = {
            "short_term_frames": short_term_frames,
            "short_term": short_term_metrics,
            "long_term": long_term_metrics,
            "inference_time_ms_per_sample": inference_time_ms_per_sample,
        }

        self.summary.add_scalar("Test/div", long_term_metrics["diversity"], epoch)
        self.summary.add_scalar("Test/ade", long_term_metrics["ade"], epoch)
        self.summary.add_scalar("Test/fde", long_term_metrics["fde"], epoch)
        self.summary.add_scalar("Test/mmade", long_term_metrics["mmade"], epoch)
        self.summary.add_scalar("Test/mmfde", long_term_metrics["mmfde"], epoch)
        self.summary.add_scalar("Test/inference_time_ms_per_sample", inference_time_ms_per_sample, epoch)

        return (
            long_term_metrics["diversity"],
            long_term_metrics["ade"],
            long_term_metrics["fde"],
            long_term_metrics["mmade"],
            long_term_metrics["mmfde"],
        )

    def run(self):
        for epoch in range(self.start_epoch, self.cfg.epoch_t2 + 1):
            epoch_start_time = time.time()
            self.summary.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)

            (
                average_allloss,
                average_adeerrors,
                average_hinges,
                average_kls_p1,
                average_similar_loss,
                average_velocity_loss,
                average_acceleration_loss,
            ) = self.train(epoch, draw=False)
            self.scheduler.step()

            print(
                "Train ==> Epoch [{}/{}]: all {:.4f} | ades {:.4f} | hinges {:.4f} | klsp1 {:.4f} | similar {:.4f} | velocity {:.4f} | acceleration {:.4f} | epoch_time {:.2f}s".format(
                    epoch,
                    self.cfg.epoch_t2,
                    average_allloss,
                    average_adeerrors,
                    average_hinges,
                    average_kls_p1,
                    average_similar_loss,
                    average_velocity_loss,
                    average_acceleration_loss,
                    time.time() - epoch_start_time,
                )
            )

            eval_interval = 1 if self.is_debug else self.cfg.eval_interval
            should_eval = epoch % eval_interval == 0 or epoch == self.cfg.epoch_t2
            should_draw = epoch % self.draw_interval == 0 or epoch == self.cfg.epoch_t2
            if should_eval:
                diversity, ade, fde, mmade, mmfde = self.eval(epoch=epoch, draw=should_draw)
                print(
                    "Test --+ epo {}: div {:.4f} | ade {:.4f} | fde {:.4f} | mmade {:.4f} | mmfde {:.4f}".format(
                        epoch, diversity, ade, fde, mmade, mmfde
                    )
                )
                self._maybe_save_best(epoch, diversity, ade, fde, mmade, mmfde)

            if epoch == self.cfg.epoch_t2:
                self.save(
                    self._last_stage_checkpoint_path(),
                    epoch,
                    average_allloss,
                    metrics={
                        "train_loss": float(average_allloss),
                        "checkpoint_type": "last",
                    },
                )
