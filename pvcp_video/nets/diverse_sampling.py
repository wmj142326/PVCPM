#!/usr/bin/env python
# encoding: utf-8

import torch
from torch import nn
from torch.nn import BatchNorm1d, Linear, Module, Sequential, Tanh

from .gcn_layers import GraphConvBlock, ResGCB


class DiverseSampling(Module):
    def __init__(
        self,
        node_n=16,
        hidden_dim=256,
        base_dim=64,
        z_dim=64,
        dct_n=10,
        base_num_p1=10,
        dropout_rate=0,
        use_video=False,
        video_feat_dim=128,
        semantic_input_dim=512,
        video_in_condition=False,
        video_in_gaussian=False,
        video_condition_mode="cat",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.base_dim = base_dim
        self.base_num_p1 = base_num_p1
        self.node_n = node_n
        self.hidden_dim = hidden_dim
        requested_use_video = bool(use_video)
        self.video_feat_dim = int(video_feat_dim)
        self.semantic_input_dim = int(semantic_input_dim)
        self.video_in_condition = requested_use_video and bool(video_in_condition)
        self.video_in_gaussian = requested_use_video and bool(video_in_gaussian)
        self.video_condition_mode = str(video_condition_mode or "cat").strip().lower()
        if self.video_condition_mode not in {"cat", "film"}:
            raise ValueError(f"Unsupported video_condition_mode={video_condition_mode!r}. Expected 'cat' or 'film'.")
        self.use_video = bool(self.video_in_condition or self.video_in_gaussian)

        condition_input_dim = 3 * dct_n
        bases_input_dim = node_n * hidden_dim
        self.semantic_encoder = None
        self.condition_video_proj = None
        self.condition_video_mod = None
        self.gaussian_video_mod = None
        if self.use_video:
            self.semantic_encoder = Sequential(
                Linear(self.semantic_input_dim, self.video_feat_dim),
                Tanh(),
            )
            if self.video_in_condition:
                if self.video_condition_mode == "cat":
                    self.condition_video_proj = Sequential(Linear(self.video_feat_dim, self.video_feat_dim), Tanh())
                    condition_input_dim += self.video_feat_dim
                else:
                    self.condition_video_mod = Sequential(
                        Linear(self.video_feat_dim, self.hidden_dim * 2),
                        Tanh(),
                    )
            if self.video_in_gaussian:
                self.gaussian_video_mod = Sequential(
                    Linear(self.video_feat_dim, self.base_dim * 2),
                    Tanh(),
                )

        self.condition_enc = Sequential(
            GraphConvBlock(
                condition_input_dim,
                hidden_dim,
                node_n,
                node_n,
                dropout_rate=dropout_rate,
                bias=True,
                residual=False,
            ),
            ResGCB(hidden_dim, hidden_dim, node_n, node_n, dropout_rate=dropout_rate, bias=True, residual=True),
            ResGCB(hidden_dim, hidden_dim, node_n, node_n, dropout_rate=dropout_rate, bias=True, residual=True),
        )
        self.bases_p1 = Sequential(
            Linear(bases_input_dim, self.base_num_p1 * self.base_dim),
            BatchNorm1d(self.base_num_p1 * self.base_dim),
            Tanh(),
        )

        self.mean_p1 = Sequential(
            Linear(self.base_dim, 64),
            BatchNorm1d(64),
            Tanh(),
            Linear(64, self.z_dim),
        )
        self.logvar_p1 = Sequential(
            Linear(self.base_dim, 64),
            BatchNorm1d(64),
            Tanh(),
            Linear(64, self.z_dim),
        )

    def _expand_video_feature(self, video_feature):
        return video_feature.unsqueeze(dim=1).repeat([1, self.node_n, 1])

    def _encode_video(self, video_features=None):
        if not self.use_video:
            return None
        if video_features is None:
            raise ValueError("Stage 2 video conditioning is enabled, but no semantic video features were provided.")
        if video_features.ndim != 2 or video_features.shape[-1] != self.semantic_input_dim:
            raise ValueError(
                f"Expected semantic video feature shape (B, {self.semantic_input_dim}), got {tuple(video_features.shape)}."
            )
        return self.semantic_encoder(video_features)

    def _sample_weight_gumbel_softmax(self, logits, temperature=1.0, eps=1e-20):
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")
        uniform = torch.rand(logits.shape, device=logits.device)
        gumbel = -torch.log(-torch.log(uniform + eps) + eps)
        return torch.softmax((logits + gumbel) / temperature, dim=-1)

    def _build_many_weights(self, base_input, video_feature, multi_modal_head, temperature):
        b = base_input.shape[0]
        logits = torch.ones((b, self.base_num_p1), dtype=base_input.dtype, device=base_input.device)
        logits = logits / self.base_num_p1
        logits = torch.repeat_interleave(logits[:, None, :], repeats=multi_modal_head, dim=0)
        return self._sample_weight_gumbel_softmax(logits, temperature=temperature)

    def forward(
        self,
        condition,
        repeated_eps=None,
        many_weights=None,
        multi_modal_head=10,
        video_features=None,
        temperature=1.0,
    ):
        b, _, _ = condition.shape
        condition_input = condition
        video_feature = self._encode_video(video_features=video_features)
        if self.video_in_condition and self.video_condition_mode == "cat":
            conditioned_video = self.condition_video_proj(video_feature)
            condition_input = torch.cat((condition_input, self._expand_video_feature(conditioned_video)), dim=-1)

        condition_enced = self.condition_enc(condition_input)
        if self.video_in_condition and self.video_condition_mode == "film":
            scale_shift = self.condition_video_mod(video_feature).view(b, self.hidden_dim, 2)
            scale = scale_shift[..., 0].unsqueeze(1)
            shift = scale_shift[..., 1].unsqueeze(1)
            condition_enced = condition_enced * (1.0 + scale) + shift

        base_input = condition_enced.view(b, -1)
        bases = self.bases_p1(base_input).view(b, self.base_num_p1, self.base_dim)

        if many_weights is None:
            many_weights = self._build_many_weights(
                base_input=base_input,
                video_feature=video_feature,
                multi_modal_head=multi_modal_head,
                temperature=temperature,
            )
        repeat_many_bases = torch.repeat_interleave(bases, repeats=multi_modal_head, dim=0)
        many_bases_blending = torch.matmul(many_weights, repeat_many_bases).view(-1, self.base_dim)
        if self.video_in_gaussian:
            scale_shift = self.gaussian_video_mod(video_feature).view(b, self.base_dim, 2)
            scale = torch.repeat_interleave(scale_shift[..., 0], repeats=multi_modal_head, dim=0)
            shift = torch.repeat_interleave(scale_shift[..., 1], repeats=multi_modal_head, dim=0)
            many_bases_blending = many_bases_blending * (1.0 + scale) + shift

        gaussian_input = many_bases_blending

        all_mean = self.mean_p1(gaussian_input)
        all_logvar = self.logvar_p1(gaussian_input)
        all_z = torch.exp(0.5 * all_logvar) * repeated_eps + all_mean
        return all_z, all_mean, all_logvar
