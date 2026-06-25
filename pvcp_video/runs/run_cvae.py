#!/usr/bin/env python
# encoding: utf-8

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
    loss_acceleration_t1,
    loss_kl_normal,
    loss_limb_length_t1,
    loss_recons,
    loss_recover_history_t1,
    loss_velocity_t1,
)

from ..configs import ConfigCVAE
from ..configs.common import build_video_settings_summary, validate_dct_checkpoint_settings
from ..datas import (
    MaoweiGSPS_Dynamic_Seq_PVCP_Video,
    MaoweiGSPS_Dynamic_Seq_PVCP_Video_ExtandDataset_T1,
    dct_transform_torch,
    draw_multi_seqs_2d,
    get_dct_matrix,
    reverse_dct_torch,
)
from ..nets import CVAE


class _NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None


class RunCVAE:
    def __init__(self, exp_name="pvcp_video_t1", device="cuda:0", num_works=0, is_debug=False, args=None):
        super().__init__()

        self.is_debug = is_debug
        self.start_epoch = 1
        self.best_accuracy = 1e15
        self.best_metrics = None
        self.best_metric_name = "ade+fde"
        self.cfg = ConfigCVAE(exp_name=exp_name, device=device, num_works=num_works, args=args)
        self.draw_interval = 50
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

        self.model = CVAE(
            node_n=self.cfg.node_n,
            hidden_dim=self.cfg.hidden_dim,
            z_dim=self.cfg.z_dim,
            dct_n=self.cfg.dct_n,
            dropout_rate=self.cfg.dropout_rate,
        )
        if self.cfg.device != "cpu":
            self.model.cuda(self.cfg.device)

        print(
            ">>> total params of {}: {:.8f}M\n".format(
                exp_name, sum(p.numel() for p in self.model.parameters()) / 1000000.0
            )
        )
        self.dct_m = torch.from_numpy(self.dct_m).float().to(self.cfg.device)
        self.i_dct_m = torch.from_numpy(self.i_dct_m).float().to(self.cfg.device)

        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.lr_t1)

        def lambda_rule(epoch):
            return 1.0 - max(0, epoch - self.cfg.epoch_fix_t1) / float(
                self.cfg.epoch_t1 - self.cfg.epoch_fix_t1 + 1
            )

        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)

        self.train_data = MaoweiGSPS_Dynamic_Seq_PVCP_Video_ExtandDataset_T1(
            data_path=self.cfg.base_data_dir,
            data_file=self.cfg.data_file,
            scene_split_path=self.cfg.scene_split_path,
            test_ratio=self.cfg.test_ratio,
            t_his=self.cfg.t_his,
            t_pred=self.cfg.t_pred,
            dynamic_sub_len=self.cfg.sub_len_train,
            batch_size=self.cfg.train_batch_size,
            joint_used=self.cfg.joint_used,
            parents=self.cfg.parents,
            mode="train",
            multimodal_threshold=self.cfg.multimodal_threshold,
            sample_step_train=self.cfg.sample_step_train,
            sample_step_test=self.cfg.sample_step_test,
            is_debug=self.is_debug,
            load_video=False,
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
            batch_size=self.cfg.test_batch_size,
            joint_used=self.cfg.joint_used,
            parents=self.cfg.parents,
            mode="test",
            multimodal_threshold=self.cfg.multimodal_threshold,
            sample_step_train=self.cfg.sample_step_train,
            sample_step_test=self.cfg.sample_step_test,
            is_debug=self.is_debug,
            load_video=False,
            semantic_input_dim=self.cfg.semantic_input_dim,
        )
        self.test_data.get_test_similat_gt_like_dlow()

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
        return pose

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

    def _validate_checkpoint(self, checkpoint_state):
        checkpoint_cfg = checkpoint_state.get("cfg_snapshot", {})
        if not checkpoint_cfg:
            validate_dct_checkpoint_settings(self.cfg, checkpoint_cfg, context="Stage 1 checkpoint")
            return
        validate_dct_checkpoint_settings(self.cfg, checkpoint_cfg, context="Stage 1 checkpoint")

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=self.cfg.device)
        self._validate_checkpoint(state)
        self.model.load_state_dict(state["model"])
        print(f"load from {checkpoint_path}")

    def train(self, epoch, draw=False):
        self.model.train()
        average_all_loss = 0.0
        average_recons_error = 0.0
        average_vec = 0.0
        average_kl = 0.0
        average_his = 0.0
        average_limblen = 0.0
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
            datas = self._batch_to_device(batch)
            b, _, _ = datas.shape
            if b == 1:
                continue

            self.global_step = (epoch - 1) * generator_len + i + 1

            with torch.no_grad():
                padded_inputs = datas[:, :, list(range(self.cfg.t_his)) + [self.cfg.t_his - 1] * self.cfg.t_pred]
                padded_inputs_coeff = dct_transform_torch(
                    padded_inputs,
                    self.dct_m,
                    dct_n=self.cfg.dct_n,
                )
                padded_inputs_coeff = padded_inputs_coeff.view(b, -1, 3 * self.cfg.dct_n)

                padded_gts_coeff = dct_transform_torch(datas, self.dct_m, dct_n=self.cfg.dct_n)
                padded_gts_coeff = padded_gts_coeff.view(b, -1, 3 * self.cfg.dct_n)

            out_coeff, posterior_mean, posterior_logvar = self.model(
                condition=padded_inputs_coeff,
                data=padded_gts_coeff,
            )
            out_coeff = out_coeff.reshape(b, -1, self.cfg.dct_n)
            outputs = reverse_dct_torch(out_coeff, self.i_dct_m, self.cfg.t_total)

            recons_error = loss_recons(gt=datas[:, :, self.cfg.t_his :], pred=outputs[:, :, self.cfg.t_his :])
            vec = (
                datas[:, :, self.cfg.t_his - 1 : self.cfg.t_his]
                - outputs[:, :, self.cfg.t_his : self.cfg.t_his + 1]
            ).pow(2).sum() / b
            kl = loss_kl_normal(posterior_mean, posterior_logvar)
            recoverhis = loss_recover_history_t1(
                pred=outputs[:, :, : self.cfg.t_his],
                gt=datas[:, :, : self.cfg.t_his],
            )
            limblen_err = loss_limb_length_t1(outputs, datas, parents=self.cfg.parents)
            motion_pred = outputs[:, :, self.cfg.t_his - 1 :]
            motion_gt = datas[:, :, self.cfg.t_his - 1 :]
            velocity_loss = loss_velocity_t1(pred=motion_pred, gt=motion_gt)
            acceleration_loss = loss_acceleration_t1(pred=motion_pred, gt=motion_gt)

            all_loss = (
                recons_error * self.cfg.t1_recons_weight
                + vec * self.cfg.t1_vec_weight
                + kl * self.cfg.t1_kl_weight
                + recoverhis * self.cfg.t1_recoverhis_weight
                + limblen_err * self.cfg.t1_limblen_weight
                + velocity_loss * self.cfg.t1_velocity_weight
                + acceleration_loss * self.cfg.t1_acceleration_weight
            )

            self.optimizer.zero_grad()
            all_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100)
            self.optimizer.step()

            average_all_loss += all_loss.item()
            average_recons_error += recons_error.item()
            average_vec += vec.item()
            average_kl += kl.item()
            average_his += recoverhis.item()
            average_limblen += limblen_err.item()
            average_velocity_loss += velocity_loss.item()
            average_acceleration_loss += acceleration_loss.item()

            if draw and i == draw_i:
                bidx = 0
                origin = datas[bidx : bidx + 1].detach().cpu().numpy()
                origin = self._add_root_for_draw(origin) * 1000.0

                output = outputs[bidx : bidx + 1].detach().cpu().numpy()
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
        average_all_loss /= denom
        average_recons_error /= denom
        average_vec /= denom
        average_kl /= denom
        average_his /= denom
        average_limblen /= denom
        average_velocity_loss /= denom
        average_acceleration_loss /= denom

        self.summary.add_scalar("loss/averageall", average_all_loss, epoch)
        self.summary.add_scalar("loss/averagerecons", average_recons_error, epoch)
        self.summary.add_scalar("loss/averagevec", average_vec, epoch)
        self.summary.add_scalar("loss/averagekl", average_kl, epoch)
        self.summary.add_scalar("loss/averagerhis", average_his, epoch)
        self.summary.add_scalar("loss/averagelimblen", average_limblen, epoch)
        self.summary.add_scalar("loss/average_velocity", average_velocity_loss, epoch)
        self.summary.add_scalar("loss/average_acceleration", average_acceleration_loss, epoch)

        return (
            average_all_loss,
            average_recons_error,
            average_kl,
            average_vec,
            average_his,
            average_limblen,
            average_velocity_loss,
            average_acceleration_loss,
        )

    def eval(self, epoch=-1, draw=False):
        self.model.eval()
        diversity = 0.0
        ade = 0.0
        fde = 0.0
        mmade = 0.0
        mmfde = 0.0
        valid_count = 0

        try:
            os.makedirs(os.path.join(self.cfg.ckpt_dir, "images", "sample"), exist_ok=True)
        except OSError:
            pass

        dg = self.test_data.onebyone_generator()
        generator_len = len(self.test_data.similat_gt_like_dlow) if not self.is_debug else min(90, len(self.test_data.similat_gt_like_dlow))
        if generator_len <= 0:
            return 0, 0, 0, 0, 0

        draw_i = random.randint(0, max(0, generator_len - 1))

        for i, batch in enumerate(dg):
            similars = self.test_data.similat_gt_like_dlow[i]
            if similars.shape[0] == 0:
                continue

            datas = self._batch_to_device(batch)
            similars = torch.from_numpy(similars).float().to(self.cfg.device)
            z = torch.randn(datas.shape[0] * self.cfg.nk, self.cfg.z_dim, device=self.cfg.device)

            with torch.no_grad():
                padded_inputs = datas[:, :, list(range(self.cfg.t_his)) + [self.cfg.t_his - 1] * self.cfg.t_pred]
                padded_inputs_coeff = dct_transform_torch(
                    padded_inputs,
                    self.dct_m,
                    dct_n=self.cfg.dct_n,
                )
                padded_inputs_coeff = padded_inputs_coeff.view(datas.shape[0], -1, 3 * self.cfg.dct_n)
                padded_inputs_coeff = torch.repeat_interleave(padded_inputs_coeff, repeats=self.cfg.nk, dim=0)

                out_coeff = self.model.inference(
                    condition=padded_inputs_coeff,
                    z=z,
                )
                out_coeff = out_coeff.reshape(self.cfg.nk, self.cfg.node_n * 3, -1)
                outputs = reverse_dct_torch(out_coeff, self.i_dct_m, self.cfg.t_total)
                outputs = outputs[:, :, self.cfg.t_his :]

                cdiv = compute_diversity(pred=outputs).mean()
                cade = compute_ade(outputs, datas[:, :, self.cfg.t_his :])
                cfde = compute_fde(outputs, datas[:, :, self.cfg.t_his :])
                cmmade = compute_mmade(outputs, datas[:, :, self.cfg.t_his :], similars)
                cmmfde = compute_mmfde(outputs, datas[:, :, self.cfg.t_his :], similars)

            diversity += cdiv.item()
            ade += cade.item()
            fde += cfde.item()
            mmade += cmmade.item()
            mmfde += cmmfde.item()
            valid_count += 1

            if epoch == -1:
                print(
                    "Test > it {}: div {:.4f} | ade {:.4f} | fde {:.4f} | mmade {:.4f} | mmfde {:.4f}".format(
                        i, cdiv, cade, cfde, cmmade, cmmfde
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
            return 0, 0, 0, 0, 0

        diversity /= valid_count
        ade /= valid_count
        fde /= valid_count
        mmade /= valid_count
        mmfde /= valid_count

        self.summary.add_scalar("Test/div", diversity, epoch)
        self.summary.add_scalar("Test/ade", ade, epoch)
        self.summary.add_scalar("Test/fde", fde, epoch)
        self.summary.add_scalar("Test/mmade", mmade, epoch)
        self.summary.add_scalar("Test/mmfde", mmfde, epoch)

        return diversity, ade, fde, mmade, mmfde

    def run(self):
        for epoch in range(self.start_epoch, self.cfg.epoch_t1 + 1):
            epoch_start_time = time.time()
            self.summary.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)

            (
                average_all_loss,
                average_recons_error,
                average_kl,
                average_vec,
                average_his,
                average_limblen,
                average_velocity_loss,
                average_acceleration_loss,
            ) = self.train(epoch, draw=False)
            self.scheduler.step()

            print(
                "Train ==> Epoch [{}/{}]: all {:.6f}, recons {:.6f}, kl {:.6f}, "
                "vec {:.6f}, losshis {:.6f}, losslimb {:.6f}, velocity {:.6f}, acceleration {:.6f}, "
                "epoch_time {:.2f}s".format(
                    epoch,
                    self.cfg.epoch_t1,
                    average_all_loss,
                    average_recons_error,
                    average_kl,
                    average_vec,
                    average_his,
                    average_limblen,
                    average_velocity_loss,
                    average_acceleration_loss,
                    time.time() - epoch_start_time,
                )
            )

            eval_interval = 1 if self.is_debug else self.cfg.eval_interval
            should_eval = epoch % eval_interval == 0 or epoch == self.cfg.epoch_t1
            should_draw = epoch % self.draw_interval == 0 or epoch == self.cfg.epoch_t1
            if should_eval:
                diversity, ade, fde, mmade, mmfde = self.eval(epoch=epoch, draw=should_draw)
                print(
                    "Test --+ epo {}: div {:.4f} | ade {:.4f} | fde {:.4f} | mmade {:.4f} | mmfde {:.4f}".format(
                        epoch, diversity, ade, fde, mmade, mmfde
                    )
                )
                self._maybe_save_best(epoch, diversity, ade, fde, mmade, mmfde)

            if epoch == self.cfg.epoch_t1:
                self.save(
                    self._last_stage_checkpoint_path(),
                    epoch,
                    average_all_loss,
                    metrics={
                        "train_loss": float(average_all_loss),
                        "checkpoint_type": "last",
                    },
                )
