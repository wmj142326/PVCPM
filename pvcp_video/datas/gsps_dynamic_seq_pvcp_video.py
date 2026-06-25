#!/usr/bin/env python
# encoding: utf-8

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _WindowRef:
    scene: str
    segment: str
    start: int


class MaoweiGSPS_Dynamic_Seq_PVCP_Video:
    _SEMANTIC_CACHE_MEMORY = {}

    def __init__(
        self,
        t_his=10,
        t_pred=25,
        similar_cnt=10,
        dynamic_sub_len=5000,
        batch_size=8,
        data_path=r"./dataset/pvcp",
        data_file="data_3d_pvcp_video_m35.npz",
        scene_splits=None,
        scene_split_path=None,
        test_ratio=0.2,
        joint_used=None,
        parents=None,
        mode="train",
        multimodal_threshold=0.5,
        sample_step_train=1,
        sample_step_test=None,
        similar_pool_step=1,
        is_debug=False,
        load_video=False,
        semantic_feature_file="",
        stage2_semantic_feature_file="",
        semantic_input_dim=512,
        **_ignored_kwargs,
    ):
        self.t_his = int(t_his)
        self.t_pred = int(t_pred)
        self.t_total = self.t_his + self.t_pred
        self.similar_cnt = int(similar_cnt)
        self.dynamic_sub_len = int(dynamic_sub_len)
        self.batch_size = int(batch_size)
        self.mode = mode
        self.multimodal_threshold = float(multimodal_threshold)
        self.sample_step_train = int(sample_step_train)
        self.sample_step_test = self.t_his if sample_step_test is None else int(sample_step_test)
        self.similar_pool_step = int(similar_pool_step)
        self.is_debug = bool(is_debug)

        fallback_semantic_feature_file = str(semantic_feature_file or "")
        self.stage2_semantic_feature_file = str(stage2_semantic_feature_file or "")
        if bool(load_video) and fallback_semantic_feature_file:
            if not self.stage2_semantic_feature_file:
                self.stage2_semantic_feature_file = fallback_semantic_feature_file
        self.load_stage2_video = bool(self.stage2_semantic_feature_file)
        self.load_video = bool(load_video) and self.load_stage2_video
        self.semantic_feature_file = (
            fallback_semantic_feature_file
            or self.stage2_semantic_feature_file
        )
        self.semantic_input_dim = int(semantic_input_dim)
        self.stage2_semantic_feature_cache = None

        self.parents = (
            [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 12, 12, 13, 14, 16, 17, 18, 19]
            if parents is None
            else list(parents)
        )
        joint_num = len(self.parents)
        self.joint_used = list(range(joint_num)) if joint_used is None else list(joint_used)
        if len(self.joint_used) < 2:
            raise ValueError("joint_used must contain at least 2 joints.")
        if len(self.parents) != len(self.joint_used):
            raise ValueError(
                "For PVCP video loader, parents and joint_used must describe the same selected skeleton. "
                f"Got len(parents)={len(self.parents)} and len(joint_used)={len(self.joint_used)}."
            )

        self.data_file = os.path.join(data_path, data_file)
        loaded_npz = np.load(self.data_file, allow_pickle=True)
        data_o = loaded_npz["positions_3d"].item()
        loaded_npz.close()

        self.all_segment_lengths = {}
        for scene_name, scene_segments in data_o.items():
            for segment_name, seq in scene_segments.items():
                seq = np.asarray(seq)
                if seq.ndim != 3 or seq.shape[-1] != 3:
                    raise ValueError(
                        f"Expected sequence shape (T, J, 3), got {seq.shape} for {scene_name}/{segment_name}."
                    )
                self.all_segment_lengths[(scene_name, segment_name)] = int(seq.shape[0])

        if scene_split_path is None:
            scene_split_path = os.path.join(data_path, "pvcp_scene_splits_8_2.json")
        elif not os.path.isabs(scene_split_path):
            scene_split_path = os.path.join(data_path, scene_split_path)
        self.scene_split_path = scene_split_path
        self.scene_splits = self._resolve_scene_splits(
            all_scene_names=sorted(data_o.keys()),
            scene_splits=scene_splits,
            scene_split_path=self.scene_split_path,
            test_ratio=test_ratio,
        )

        if mode not in self.scene_splits:
            raise ValueError(f"Unsupported mode={mode}. Available modes: {list(self.scene_splits.keys())}")

        self.scene_names = self.scene_splits[mode]
        self.data = {}
        self.segment_order = []
        self.usable_segment_order = []
        self.segment_lookup = {}
        self.feature_dim = (len(self.joint_used) - 1) * 3

        for scene_name in self.scene_names:
            if scene_name not in data_o:
                continue

            scene_segments = {}
            for segment_name in sorted(data_o[scene_name].keys()):
                seq = np.asarray(data_o[scene_name][segment_name], dtype=np.float32)
                if seq.ndim != 3 or seq.shape[-1] != 3:
                    raise ValueError(
                        f"Expected sequence shape (T, J, 3), got {seq.shape} for {scene_name}/{segment_name}."
                    )

                seq = seq[:, self.joint_used, :].copy()
                seq[:, 1:] -= seq[:, :1]

                scene_segments[segment_name] = seq
                self.segment_lookup[(scene_name, segment_name)] = seq
                self.segment_order.append((scene_name, segment_name))
                if seq.shape[0] >= self.t_total:
                    self.usable_segment_order.append((scene_name, segment_name))

            if scene_segments:
                self.data[scene_name] = scene_segments

        if not self.usable_segment_order:
            raise ValueError(
                f"No usable PVCP segments found for mode={mode} with t_total={self.t_total}. "
                "Try reducing t_his/t_pred or using a different split."
            )

        if self.load_video:
            if self.load_stage2_video:
                self.stage2_semantic_feature_cache = self._load_semantic_feature_cache(
                    self.stage2_semantic_feature_file
                )

        self.similat_gt_like_dlow = []
        self._similar_refs = []
        self._similar_start_poses = None
        if self.mode == "train" and self.similar_cnt > 0:
            self._build_similarity_pool()

        loaded_segment_count = sum(len(v) for v in self.data.values())
        print(
            f"{mode} PVCP-video Data Loaded, segments: {loaded_segment_count}, "
            f"usable_segments: {len(self.usable_segment_order)}, "
            f"batch_size: {batch_size}, t_total: {self.t_total}, "
            f"load_video: {self.load_video}, "
            f"stage2_semantic_feature_file: {os.path.basename(self.stage2_semantic_feature_file) if self.stage2_semantic_feature_file else ''}"
        )

    def _resolve_scene_splits(self, all_scene_names, scene_splits, scene_split_path, test_ratio):
        if scene_splits is not None:
            resolved = {k: list(v) for k, v in scene_splits.items()}
            resolved.setdefault("all", list(all_scene_names))
            return resolved

        if scene_split_path is not None and os.path.exists(scene_split_path):
            with open(scene_split_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError(f"Scene split json must be an object, got {type(loaded).__name__}.")
            known_scenes = set(all_scene_names)
            resolved = {
                "train": [scene for scene in loaded.get("train", []) if scene in known_scenes],
                "test": [scene for scene in loaded.get("test", []) if scene in known_scenes],
            }
            resolved["all"] = [scene for scene in loaded.get("all", all_scene_names) if scene in known_scenes]
            if not resolved["all"]:
                resolved["all"] = list(all_scene_names)
            return resolved

        if len(all_scene_names) == 1:
            return {"train": list(all_scene_names), "test": list(all_scene_names), "all": list(all_scene_names)}

        num_test = max(1, int(round(len(all_scene_names) * test_ratio)))
        num_test = min(num_test, len(all_scene_names) - 1)
        split_idx = len(all_scene_names) - num_test
        return {
            "train": list(all_scene_names[:split_idx]),
            "test": list(all_scene_names[split_idx:]),
            "all": list(all_scene_names),
        }

    def _build_similarity_pool(self):
        refs = []
        start_poses = []

        for scene_name, segment_name in self.usable_segment_order:
            seq = self.segment_lookup[(scene_name, segment_name)]
            max_start = seq.shape[0] - self.t_total
            for start in range(0, max_start + 1, self.similar_pool_step):
                refs.append(_WindowRef(scene=scene_name, segment=segment_name, start=start))
                start_pose = seq[start + self.t_his - 1, 1:, :].reshape(-1)
                start_poses.append(start_pose)

        self._similar_refs = refs
        self._similar_start_poses = np.stack(start_poses, axis=0).astype(np.float32)

    def _window_to_model_array(self, window):
        return window[:, 1:, :].reshape(1, self.t_total, -1).transpose(0, 2, 1).astype(np.float32)

    def _sample_window(self):
        scene_name, segment_name = self.usable_segment_order[np.random.randint(len(self.usable_segment_order))]
        seq = self.segment_lookup[(scene_name, segment_name)]
        max_start = seq.shape[0] - self.t_total
        start = 0 if max_start == 0 else np.random.randint(max_start + 1)
        window = seq[start : start + self.t_total]
        return scene_name, segment_name, start, window

    def _build_similar_batch(self, anchor_scene, anchor_segment, anchor_start, anchor_window):
        if self._similar_start_poses is None or len(self._similar_refs) == 0:
            zeros = np.zeros((1, self.similar_cnt, self.feature_dim, self.t_total), dtype=np.float32)
            return zeros

        anchor_pose = anchor_window[self.t_his - 1, 1:, :].reshape(-1)
        distances = np.linalg.norm(self._similar_start_poses - anchor_pose[None, :], axis=1)
        candidate_indices = np.where((distances < self.multimodal_threshold) & (distances > 0))[0].tolist()

        anchor_ref = _WindowRef(anchor_scene, anchor_segment, anchor_start)
        candidate_indices = [idx for idx in candidate_indices if self._similar_refs[idx] != anchor_ref]

        if candidate_indices:
            choose_num = min(len(candidate_indices), self.similar_cnt)
            chosen = np.random.choice(candidate_indices, choose_num, replace=False)
            similar_arrays = []
            for idx in chosen:
                ref = self._similar_refs[idx]
                seq = self.segment_lookup[(ref.scene, ref.segment)]
                sim_window = seq[ref.start : ref.start + self.t_total].copy()
                sim_window[: self.t_his] = anchor_window[: self.t_his]
                similar_arrays.append(self._window_to_model_array(sim_window)[0])
            similar_arrays = np.stack(similar_arrays, axis=0)
        else:
            similar_arrays = np.zeros((0, self.feature_dim, self.t_total), dtype=np.float32)

        if similar_arrays.shape[0] < self.similar_cnt:
            pad = np.zeros(
                (self.similar_cnt - similar_arrays.shape[0], self.feature_dim, self.t_total),
                dtype=np.float32,
            )
            similar_arrays = np.concatenate([similar_arrays, pad], axis=0)

        return similar_arrays[None, ...]

    def _read_semantic_feature_cache(self, cache_path):
        loaded_npz = np.load(cache_path, allow_pickle=True)
        try:
            if "feature_data" not in loaded_npz.files:
                raise ValueError(f"{cache_path} does not contain feature_data.")
            feature_data = loaded_npz["feature_data"].item()
            metadata = loaded_npz["metadata"].item() if "metadata" in loaded_npz.files else {}
        finally:
            loaded_npz.close()
        return feature_data, metadata

    def _load_semantic_feature_cache(self, cache_path):
        if not cache_path:
            raise FileNotFoundError(
                "Video conditioning is enabled, but semantic_feature_file is empty. "
                "Please generate an .rfeat.npz cache first."
            )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Semantic feature cache not found: {cache_path}")

        memory_key = (cache_path, self.semantic_input_dim, self.t_his)
        if memory_key in self._SEMANTIC_CACHE_MEMORY:
            return self._SEMANTIC_CACHE_MEMORY[memory_key]

        feature_data, metadata = self._read_semantic_feature_cache(cache_path)
        cached_dim = int(metadata.get("feature_dim", metadata.get("export_feature_dim", self.semantic_input_dim)))
        if cached_dim != self.semantic_input_dim:
            raise ValueError(
                f"Semantic cache feature dim mismatch: expected {self.semantic_input_dim}, got {cached_dim}."
            )
        cached_history = int(metadata.get("history_frames", metadata.get("t_his", self.t_his)))
        if cached_history != self.t_his:
            raise ValueError(
                f"Semantic cache history length mismatch: expected {self.t_his}, got {cached_history}."
            )

        start_lookup = {}
        for scene_name, scene_segments in feature_data.items():
            start_lookup.setdefault(scene_name, {})
            for segment_name, segment_entry in scene_segments.items():
                window_features = np.asarray(segment_entry.get("window_features"), dtype=np.float32)
                start_indices = np.asarray(segment_entry.get("start_indices"), dtype=np.int32)
                if window_features.ndim != 2 or window_features.shape[-1] != self.semantic_input_dim:
                    raise ValueError(
                        f"Invalid semantic feature shape for {scene_name}/{segment_name}: {window_features.shape}"
                    )
                if start_indices.ndim != 1 or start_indices.shape[0] != window_features.shape[0]:
                    raise ValueError(
                        f"Invalid semantic start index shape for {scene_name}/{segment_name}: {start_indices.shape}"
                    )
                start_lookup[scene_name][segment_name] = {
                    int(start): window_features[idx] for idx, start in enumerate(start_indices.tolist())
                }

        cache = {"feature_data": feature_data, "metadata": metadata, "start_lookup": start_lookup}
        self._SEMANTIC_CACHE_MEMORY[memory_key] = cache
        return cache

    def _load_semantic_features_from_cache(self, cache, cache_path, scene_name, segment_name, start):
        try:
            feature = cache["start_lookup"][scene_name][segment_name][int(start)]
        except KeyError as exc:
            raise KeyError(
                f"Missing semantic feature for {scene_name}/{segment_name} window_start={start} "
                f"in {cache_path}."
            ) from exc
        return np.asarray(feature, dtype=np.float32)

    def _load_stage2_semantic_features(self, scene_name, segment_name, start):
        return self._load_semantic_features_from_cache(
            cache=self.stage2_semantic_feature_cache,
            cache_path=self.stage2_semantic_feature_file,
            scene_name=scene_name,
            segment_name=segment_name,
            start=start,
        )

    def _build_sample_bundle(self, scene_name, segment_name, start, window):
        sample = {
            "pose": self._window_to_model_array(window),
            "history_semantic_features": None,
            "stage2_semantic_features": None,
            "scene_name": scene_name,
            "segment_name": segment_name,
            "window_start": int(start),
        }
        if self.load_video:
            if self.load_stage2_video:
                sample["stage2_semantic_features"] = self._load_stage2_semantic_features(
                    scene_name=scene_name,
                    segment_name=segment_name,
                    start=start,
                )
            if sample["stage2_semantic_features"] is not None:
                sample["history_semantic_features"] = sample["stage2_semantic_features"]
        return sample

    def _collate_sample_bundles(self, sample_bundles, similar_bundles=None):
        batch = {
            "pose": np.concatenate([sample["pose"] for sample in sample_bundles], axis=0),
            "history_semantic_features": None,
            "stage2_semantic_features": None,
            "scene_name": [sample["scene_name"] for sample in sample_bundles],
            "segment_name": [sample["segment_name"] for sample in sample_bundles],
            "window_start": np.asarray([sample["window_start"] for sample in sample_bundles], dtype=np.int32),
        }
        if self.load_video:
            if self.load_stage2_video:
                batch["stage2_semantic_features"] = np.stack(
                    [sample["stage2_semantic_features"] for sample in sample_bundles],
                    axis=0,
                )
                batch["history_semantic_features"] = batch["stage2_semantic_features"]
        if similar_bundles is not None:
            batch["similar_pose"] = np.concatenate(similar_bundles, axis=0)
        return batch

    def sample(self):
        scene_name, segment_name, start, window = self._sample_window()
        sample_bundle = self._build_sample_bundle(scene_name, segment_name, start, window)
        if self.mode == "train" and self.similar_cnt > 0:
            similars = self._build_similar_batch(
                anchor_scene=scene_name,
                anchor_segment=segment_name,
                anchor_start=start,
                anchor_window=window,
            )
            return sample_bundle, similars
        return sample_bundle

    def batch_generator(self):
        if self.is_debug:
            self.dynamic_sub_len = 200

        for _ in range(self.dynamic_sub_len // self.batch_size):
            sample_bundles = []
            similar_bundles = []
            for _ in range(self.batch_size):
                sample_i = self.sample()
                if self.mode == "train" and self.similar_cnt > 0:
                    sample_bundles.append(sample_i[0])
                    similar_bundles.append(sample_i[1])
                else:
                    sample_bundles.append(sample_i)

            if self.mode == "train" and self.similar_cnt > 0:
                yield self._collate_sample_bundles(sample_bundles, similar_bundles=similar_bundles)
            else:
                yield self._collate_sample_bundles(sample_bundles)

    def onebyone_generator(self):
        ordered_segments = self.usable_segment_order
        if self.is_debug:
            ordered_segments = ordered_segments[:1]

        step = self.sample_step_train if self.mode == "train" else self.sample_step_test
        for scene_name, segment_name in ordered_segments:
            seq = self.segment_lookup[(scene_name, segment_name)]
            seq_len = seq.shape[0]
            for start in range(0, seq_len - self.t_total + 1, step):
                window = seq[start : start + self.t_total]
                sample_bundle = self._build_sample_bundle(scene_name, segment_name, start, window)
                yield self._collate_sample_bundles([sample_bundle])

    def get_all_test_sample_inorder(self):
        all_data = []
        for sample in self.onebyone_generator():
            all_data.append(sample["pose"])

        if not all_data:
            return np.zeros((0, self.feature_dim, self.t_total), dtype=np.float32)
        return np.concatenate(all_data, axis=0)

    def get_test_similat_gt_like_dlow(self):
        all_data = self.get_all_test_sample_inorder()
        if all_data.shape[0] == 0:
            self.similat_gt_like_dlow = []
            print("#0 future: 0/0")
            print("#<=5 future: 0/0")
            print("#>5 future: 0/0")
            return

        all_start_pose = all_data[:, :, self.t_his - 1]
        similar_gts = []
        num_mult = []

        for i in range(all_data.shape[0]):
            distances = np.linalg.norm(all_start_pose - all_start_pose[i : i + 1], axis=1)
            ind = np.nonzero(np.logical_and(distances < self.multimodal_threshold, distances > 0))[0]
            choosed_pseudo = all_data[ind][:, :, self.t_his :]
            similar_gts.append(choosed_pseudo)
            num_mult.append(len(ind))

        num_mult = np.asarray(num_mult, dtype=np.int32)
        print(f"#0 future: {len(np.where(num_mult == 0)[0])}/{all_data.shape[0]}")
        print(f"#<=5 future: {len(np.where(num_mult <= 5)[0])}/{all_data.shape[0]}")
        print(f"#>5 future: {len(np.where(num_mult > 5)[0])}/{all_data.shape[0]}")
        self.similat_gt_like_dlow = similar_gts


class MaoweiGSPS_Dynamic_Seq_PVCP_Video_ExpandDataset_T1(MaoweiGSPS_Dynamic_Seq_PVCP_Video):
    def __init__(self, similar_cnt=0, *args, **kwargs):
        super().__init__(similar_cnt=similar_cnt, *args, **kwargs)

    def sample(self):
        scene_name, segment_name, start, window = self._sample_window()
        return self._build_sample_bundle(scene_name, segment_name, start, window)

    def batch_generator(self):
        if self.is_debug:
            self.dynamic_sub_len = 200

        for _ in range(self.dynamic_sub_len // self.batch_size):
            sample_bundles = [self.sample() for _ in range(self.batch_size)]
            yield self._collate_sample_bundles(sample_bundles)


MaoweiGSPS_Dynamic_Seq_PVCP_Video_ExtandDataset_T1 = MaoweiGSPS_Dynamic_Seq_PVCP_Video_ExpandDataset_T1
