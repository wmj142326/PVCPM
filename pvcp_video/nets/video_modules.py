#!/usr/bin/env python
# encoding: utf-8

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


VIDEO_FRAME_FEATURE_DIM = 128
OFFLINE_VIDEO_EXTRACTOR_INIT_SEED = 20260413


def format_bbox_scale_token(scale):
    scale_text = f"{float(scale):.3f}".rstrip("0").rstrip(".")
    return scale_text or "1"


def compute_scaled_bbox_xyxy(bbox, width, height, bbox_scale=1.0):
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        return 0, 0, 1, 1

    if bbox is None:
        return 0, 0, width, height

    x, y, w, h = [float(v) for v in bbox]
    w = max(w, 1.0)
    h = max(h, 1.0)
    scale = max(float(bbox_scale), 1e-6)

    center_x = x + 0.5 * w
    center_y = y + 0.5 * h
    scaled_w = max(w * scale, 1.0)
    scaled_h = max(h * scale, 1.0)

    x1 = max(0, int(math.floor(center_x - 0.5 * scaled_w)))
    y1 = max(0, int(math.floor(center_y - 0.5 * scaled_h)))
    x2 = min(width, int(math.ceil(center_x + 0.5 * scaled_w)))
    y2 = min(height, int(math.ceil(center_y + 0.5 * scaled_h)))

    if x2 <= x1 or y2 <= y1:
        return 0, 0, width, height
    return x1, y1, x2, y2


def _build_frame_encoder(backbone):
    if backbone not in {"tiny_cnn", "frame_cnn"}:
        raise ValueError(
            f"Unsupported video backbone: {backbone}. "
            'The current implementation supports "tiny_cnn" and "frame_cnn".'
        )

    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, VIDEO_FRAME_FEATURE_DIM, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(VIDEO_FRAME_FEATURE_DIM),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )


def _initialize_offline_frame_encoder(module):
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(OFFLINE_VIDEO_EXTRACTOR_INIT_SEED)
        for submodule in module.modules():
            if isinstance(submodule, nn.Conv2d):
                nn.init.kaiming_uniform_(submodule.weight, a=math.sqrt(5))
                if submodule.bias is not None:
                    nn.init.zeros_(submodule.bias)
            elif isinstance(submodule, nn.BatchNorm2d):
                nn.init.ones_(submodule.weight)
                nn.init.zeros_(submodule.bias)
                submodule.running_mean.zero_()
                submodule.running_var.fill_(1.0)
    finally:
        torch.random.set_rng_state(rng_state)


def build_offline_frame_feature_extractor(backbone="tiny_cnn"):
    extractor = _build_frame_encoder(backbone=backbone)
    _initialize_offline_frame_encoder(extractor)
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad = False
    return extractor


class VideoInputAdapter(nn.Module):
    def __init__(self, output_size=112):
        super().__init__()
        self.output_size = int(output_size)

    def _crop_single_frame(self, frame, bbox, bbox_scale=1.0):
        _, height, width = frame.shape
        x1, y1, x2, y2 = compute_scaled_bbox_xyxy(bbox=bbox, width=width, height=height, bbox_scale=bbox_scale)

        if x2 <= x1 or y2 <= y1:
            cropped = frame
        else:
            cropped = frame[:, y1:y2, x1:x2]
        resized = F.interpolate(
            cropped.unsqueeze(0),
            size=(self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

    def prepare_frames(self, frames, bbox=None, use_bbox=False, bbox_scale=1.0):
        if frames is None:
            raise ValueError("frames must be provided when video conditioning is enabled.")

        if not use_bbox:
            return frames

        if bbox is None:
            raise ValueError("bbox must be provided when use_bbox=True.")

        batch_size, clip_len, _, _, _ = frames.shape
        cropped = []
        for batch_idx in range(batch_size):
            clip_frames = []
            for time_idx in range(clip_len):
                clip_frames.append(
                    self._crop_single_frame(
                        frames[batch_idx, time_idx],
                        bbox[batch_idx, time_idx],
                        bbox_scale=bbox_scale,
                    )
                )
            cropped.append(torch.stack(clip_frames, dim=0))
        return torch.stack(cropped, dim=0)

    def forward(self, frames, bbox=None, use_bbox=False, bbox_scale=1.0):
        return self.prepare_frames(frames=frames, bbox=bbox, use_bbox=use_bbox, bbox_scale=bbox_scale)


class VideoFeatureEncoder(nn.Module):
    def __init__(self, backbone="tiny_cnn", feat_dim=128, image_size=112):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = int(feat_dim)
        self.image_size = int(image_size)
        self.frame_feature_dim = VIDEO_FRAME_FEATURE_DIM
        self.input_adapter = VideoInputAdapter(output_size=self.image_size)
        self.frame_encoder = _build_frame_encoder(backbone=self.backbone)
        self.temporal_head = nn.Sequential(
            nn.Linear(self.frame_feature_dim, self.feat_dim),
            nn.Tanh(),
        )

    def extract_frame_features(self, frames, bbox=None, use_bbox=False, bbox_scale=1.0):
        prepared_frames = self.input_adapter(
            frames=frames,
            bbox=bbox,
            use_bbox=use_bbox,
            bbox_scale=bbox_scale,
        )
        batch_size, clip_len, channels, height, width = prepared_frames.shape
        encoded = self.frame_encoder(prepared_frames.reshape(batch_size * clip_len, channels, height, width))
        return encoded.reshape(batch_size, clip_len, -1)

    def encode_frame_features(self, frame_features):
        if frame_features is None:
            raise ValueError("frame_features must be provided when encoding cached video features.")
        if frame_features.ndim != 3:
            raise ValueError(
                f"Expected frame_features shape (B, T, C), got {tuple(frame_features.shape)}."
            )
        if frame_features.shape[-1] != self.frame_feature_dim:
            raise ValueError(
                f"Expected cached frame feature dim {self.frame_feature_dim}, got {frame_features.shape[-1]}."
            )

        clip_feature = frame_features.mean(dim=1)
        return self.temporal_head(clip_feature)

    def encode_video(self, frames=None, bbox=None, use_bbox=False, frame_features=None, bbox_scale=1.0):
        if frame_features is None:
            frame_features = self.extract_frame_features(
                frames=frames,
                bbox=bbox,
                use_bbox=use_bbox,
                bbox_scale=bbox_scale,
            )
        return self.encode_frame_features(frame_features)

    def forward(self, frames=None, bbox=None, use_bbox=False, frame_features=None, bbox_scale=1.0):
        return self.encode_video(
            frames=frames,
            bbox=bbox,
            use_bbox=use_bbox,
            frame_features=frame_features,
            bbox_scale=bbox_scale,
        )
