#!/usr/bin/env python
# encoding: utf-8

import getpass
import json
import os

from .common import apply_common_video_config, build_ckpt_exp_name


class ConfigCVAE:
    def __init__(self, exp_name="", device="cuda:0", num_works=10, args=None):
        self.platform = getpass.getuser()
        self.exp_name = exp_name
        self.ckpt_exp_name = build_ckpt_exp_name(exp_name, args=args)
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        # >>> model
        self.nk = 50  # 评估/可视化时采样的未来姿态数量
        self.seperate_head = self.nk // 2  # 多样性相关分组大小，主要为兼容 stage2 的评估逻辑

        self.z_dim = 64  # 潜变量 z 的维度，控制 CVAE 隐空间表达容量
        self.dct_n = 20  # Number of DCT coefficients used by the temporal pose representation.
        self.hidden_dim = 256  # GCN 隐藏层维度，越大模型容量越强但更易过拟合

        # >>> data
        self.t_his = 10
        self.t_pred = 25
        self.t_total = self.t_his + self.t_pred

        self.sample_step_train = 1
        self.sample_step_test = self.t_his

        self.sub_len_train = 5000  # 每个 epoch 随机抽取的训练窗口数量，越大训练越充分但越慢
        self.multimodal_threshold = 0.5  # 查找相似历史姿态的距离阈值，用于构建多模态候选
        self.train_similar_cnt = 0  # stage1 是单目标 CVAE 训练，不采样相似未来候选

        self.min_frames = 35
        self.data_file = f"data_3d_pvcp_m{self.min_frames}.npz"
        self.test_ratio = 0.2
        self.scene_split_file = "pvcp_scene_splits_8_2.json"
        self.draw_yaw_deg = -20.0

        self.joint_used = [i for i in range(22)]
        self.num_joints = len(self.joint_used)
        self.root_idx = 0
        self.parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19]
        self.node_n = self.num_joints - 1

        self.I_plot = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19]
        self.J_plot = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        self.LR_plot = [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]

        # Backward-compatible aliases for any older PVCP scripts.
        self.I22_plot = list(self.I_plot)
        self.J22_plot = list(self.J_plot)
        self.LR22_plot = list(self.LR_plot)
        self.I17_plot = list(self.I_plot)
        self.J17_plot = list(self.J_plot)
        self.LR17_plot = list(self.LR_plot)

        # >>> runner
        self.lr_t1 = 3e-4  # 学习率
        self.train_batch_size = 32
        self.test_batch_size = 1
        self.epoch_t1 = 200  # stage1 CVAE 总训练轮数
        self.epoch_fix_t1 = 50  # 从该 epoch 后学习率开始线性衰减
        self.eval_interval = 10  # 每隔多少个 epoch 在测试集上评估一次
        self.snapshot_interval = self.epoch_t1 + 1

        self.t1_recons_weight = 1
        self.t1_kl_weight = 0.5
        self.t1_vec_weight = 1000
        self.t1_recoverhis_weight = 100
        self.t1_limblen_weight = 1000
        self.t1_velocity_weight = 0.1
        self.t1_acceleration_weight = 0.01

        self.dropout_rate = 0

        self.device = device
        self.num_works = num_works

        self.ckpt_dir = os.path.join(self.repo_root, "ckpt", self.ckpt_exp_name)
        for sub_dir in ("models", "images"):
            try:
                os.makedirs(os.path.join(self.ckpt_dir, sub_dir), exist_ok=True)
            except OSError:
                pass

        self.base_data_dir = os.path.join(self.repo_root, "dataset", "pvcp")
        self.data_path = os.path.join(self.base_data_dir, self.data_file)
        self.scene_split_path = os.path.join(self.base_data_dir, self.scene_split_file)

        self.scene_splits = self._load_scene_splits()
        self.subjects = self.scene_splits

        self.similar_idx_path = os.path.join(
            self.base_data_dir,
            "pvcp_multi_modal",
            f"t_his{self.t_his}_thre{self.multimodal_threshold:.3f}_t_pred{self.t_pred}_index.npz",
        )
        self.similar_pool_path = os.path.join(
            self.base_data_dir,
            "pvcp_multi_modal",
            f"data_candi_t_his{self.t_his}_t_pred{self.t_pred}.npz",
        )

        apply_common_video_config(self, args=args)

    def _load_scene_splits(self):
        if not os.path.exists(self.scene_split_path):
            return {"train": [], "test": [], "all": []}
        with open(self.scene_split_path, "r", encoding="utf-8") as f:
            scene_splits = json.load(f)
        return {
            "train": list(scene_splits.get("train", [])),
            "test": list(scene_splits.get("test", [])),
            "all": list(scene_splits.get("all", [])),
        }
