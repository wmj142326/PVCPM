#!/usr/bin/env python
# encoding: utf-8

import getpass
import json
import os

from .common import apply_common_video_config, build_ckpt_exp_name


class ConfigDiverseSampling:
    def __init__(self, exp_name="", device="cuda:0", num_works=0, args=None):
        self.platform = getpass.getuser()
        self.exp_name = exp_name
        self.ckpt_exp_name = build_ckpt_exp_name(exp_name, args=args)
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        self.class_num = 15

        # >>> model
        self.z_dim = 64  # 潜变量 z 的维度，控制每个预测未来的隐空间表达容量
        self.dct_n = 20  # Number of DCT coefficients used by the temporal pose representation.
        self.hidden_dim = 256  # GCN/MLP 隐藏层维度，越大模型容量越强但更易过拟合
        self.base_dim = 128  # stage2 中每个多样化 basis 的特征维度
        self.base_num_p1 = 40  # stage2 生成的 basis 数量，越多可组合的未来模式越丰富

        # >>> data
        self.t_his = 10
        self.t_pred = 25
        self.t_total = self.t_his + self.t_pred

        self.sample_step_train = 1
        self.sample_step_test = self.t_his

        self.sub_len_train = 5000  # 每个 epoch 随机抽取的训练窗口数量，越大训练越充分但越慢
        self.multimodal_threshold = 0.5  # 查找相似历史姿态的距离阈值，用于构建多模态未来候选
        self.train_similar_cnt = 10  # stage2 中每个训练窗口采样的相似未来候选数量

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
        self.nk = 50  # 每个输入序列采样的多样化未来数量
        self.seperate_head = self.nk // 2  # 计算多样性 hinge loss 时的采样结果分组大小

        self.lr_t2 = 3e-4  # 学习率
        self.train_batch_size = 16
        self.test_batch_size = 1
        self.epoch_t2 = 150  # stage2 多样化采样网络总训练轮数
        self.epoch_fix_t2 = 50  # 从该 epoch 后学习率开始线性衰减
        self.eval_interval = 10  # 每隔多少个 epoch 在测试集上评估一次
        self.snapshot_interval = self.epoch_t2 + 1

        self.temperature_p1 = 0.70  # Gumbel-Softmax 温度，越低选择越尖锐，越高采样越随机/多样
        self.minthreshold = 25  # 多样性 hinge loss 鼓励的预测未来之间的最小距离

        self.t2_kl_p1_weight = 0.5
        self.t2_ade_weight = 40
        self.t2_diversity_weight = 20  # 多样性损失权重
        self.t2_similar_weight = 0.1  # 相似未来候选的 MM-ADE-like 监督权重
        self.t2_velocity_weight = 0.1  # 未来段速度一致性损失权重
        self.t2_acceleration_weight = 0.01  # 未来段加速度一致性损失权重；0 表示不启用

        self.dropout_rate = 0
        self.stage2_video_condition_mode = "film"

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
        self.model_path_t1 = os.path.join(self.repo_root, "ckpt", "pvcp_video_t1", "models", "pvcp_video_t1_best.pth")

        apply_common_video_config(self, args=args)
        default_t1_path = os.path.join(self.repo_root, "ckpt", "pvcp_video_t1", "models", "pvcp_video_t1_best.pth")
        configured_t1_path = getattr(args, "model_path_t1", "") if args is not None else ""
        self.model_path_t1 = configured_t1_path or default_t1_path

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
